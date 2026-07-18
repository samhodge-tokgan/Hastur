#!/usr/bin/env python3
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# export_detector.py -- export a clean, commercial-licensed ONNX person detector
# for the SAM 3D Body pipeline (Track A / M1).
#
# The SAM-3D-Body pipeline only consumes person BOUNDING BOXES, so any accurate
# person detector works. This replaces the export-hostile Detectron2 ViTDet-H
# cascade Mask R-CNN with a small torchvision detector (BSD-3-Clause code +
# COCO-trained weights redistributed by torchvision -- commercial-friendly).
#
# We deliberately DO NOT use Ultralytics YOLO (AGPL-3.0).
#
# Output:
#   person_detector.onnx   -- fixed square input [1,3,S,S], NCHW, RGB in [0,1],
#                             ImageNet normalization + NMS baked into the graph.
#                             Outputs post-NMS: boxes[n,4] (xyxy, in SxS letterbox
#                             pixels), labels[n] (COCO 1-based, person==1),
#                             scores[n].
#   person_detector.json   -- sidecar describing preprocessing / output semantics
#                             so DetectorEngine.cpp can consume the graph.
#
# Candidate models (see tools/DETECTOR_REPORT.md for the parity study):
#   ssdlite         torchvision ssdlite320_mobilenet_v3_large  (~13 MB, S=320)
#   frcnn_mnv3      torchvision fasterrcnn_mobilenet_v3_large_fpn (~74 MB, S=800)
#   retinanet_v2    torchvision retinanet_resnet50_fpn_v2      (~146 MB, S=800)
#   frcnn_r50_v2    torchvision fasterrcnn_resnet50_fpn_v2     (~167 MB, S=800)
#
# Usage:
#   python tools/export_detector.py --model ssdlite --out models/person_detector.onnx

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torchvision
from torchvision.models import detection as D

# Per-model factory + default fixed square input size + torchvision weights enum.
# The weights license below is torchvision's redistribution license (BSD-3-Clause
# for the code/weights packaging; COCO images/annotations underlie training).
MODELS = {
    "ssdlite": dict(
        ctor=D.ssdlite320_mobilenet_v3_large,
        weights=D.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        size=320,
    ),
    "frcnn_mnv3": dict(
        ctor=D.fasterrcnn_mobilenet_v3_large_fpn,
        weights=D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        size=800,
    ),
    "retinanet_v2": dict(
        ctor=D.retinanet_resnet50_fpn_v2,
        weights=D.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        size=800,
    ),
    "frcnn_r50_v2": dict(
        ctor=D.fasterrcnn_resnet50_fpn_v2,
        weights=D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        size=800,
    ),
}

# ImageNet normalization (baked into the exported graph by torchvision's
# GeneralizedRCNNTransform; documented in the sidecar for reference).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COCO_PERSON_CLASS = 1  # torchvision COCO models: index 0 == __background__.


def build_model(name, score_thresh, nms_thresh):
    spec = MODELS[name]
    weights = spec["weights"]
    kwargs = dict(weights=weights, box_score_thresh=score_thresh)
    # RetinaNet uses `score_thresh`/`nms_thresh`; the R-CNN family uses
    # `box_score_thresh`/`box_nms_thresh`. Pass what each accepts.
    ctor = spec["ctor"]
    import inspect

    params = inspect.signature(ctor).parameters
    kwargs = dict(weights=weights)
    if "box_score_thresh" in params:
        kwargs["box_score_thresh"] = score_thresh
        kwargs["box_nms_thresh"] = nms_thresh
    if "score_thresh" in params:
        kwargs["score_thresh"] = score_thresh
        kwargs["nms_thresh"] = nms_thresh
    model = ctor(**kwargs)
    model.eval()

    # Force the internal GeneralizedRCNNTransform to be a strict no-op on a square
    # SxS input: shortest edge == S == longest edge, so no rescale happens and the
    # output boxes live in the exact SxS letterbox pixel space we feed. The graph
    # still normalizes internally, so C++ feeds RGB in [0,1].
    S = spec["size"]
    model.transform.min_size = (S,)
    model.transform.max_size = S
    return model, S, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ssdlite", choices=list(MODELS.keys()))
    ap.add_argument("--out", default="models/person_detector.onnx")
    ap.add_argument("--size", type=int, default=0, help="override fixed square input size")
    ap.add_argument("--score-thresh", type=float, default=0.05,
                    help="low export threshold; real thresholding is done in C++")
    ap.add_argument("--nms-thresh", type=float, default=0.5)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    model, S, weights = build_model(args.model, args.score_thresh, args.nms_thresh)
    if args.size:
        S = args.size
        model.transform.min_size = (S,)
        model.transform.max_size = S

    dummy = [torch.rand(3, S, S)]
    print(f"Exporting {args.model} at fixed {S}x{S} -> {args.out}")
    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        opset_version=args.opset,
        dynamo=False,  # legacy TorchScript exporter: proven for tv detection + in-graph NMS
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        # Input is fixed [3,S,S]; only the (data-dependent) detection count is dynamic.
        dynamic_axes={"boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
    )

    size_mb = os.path.getsize(args.out) / 1e6

    # Determine the TRUE output order by tensor dtype. torchvision SSD emits
    # [boxes, scores, labels] while the R-CNN/RetinaNet families emit
    # [boxes, labels, scores]; the output_names above are positional and can
    # therefore mislabel SSD. DetectorEngine.cpp dispatches by dtype the same way:
    #   labels == int64 ; scores == float 1-D ; boxes == float N×4.
    import onnxruntime as _ort

    _sess = _ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    _bi = _li = _si = None
    for _i, _o in enumerate(_sess.get_outputs()):
        if "int" in _o.type:
            _li = _i
        elif len(_o.shape) == 2:
            _bi = _i
        else:
            _si = _i
    out_order = {"boxes_index": _bi, "labels_index": _li, "scores_index": _si,
                 "identify_by": "dtype (labels=int64, scores=float 1-D, boxes=float Nx4)"}

    sidecar = {
        "model": args.model,
        "framework": "torchvision",
        "torchvision_version": torchvision.__version__,
        "onnx_opset": args.opset,
        "onnx_size_mb": round(size_mb, 2),
        "license": "BSD-3-Clause (torchvision code + redistributed COCO-trained weights)",
        "license_commercial_ok": True,
        "input": {
            "name": "images",
            "layout": "NCHW",
            "shape": [1, 3, S, S],
            "channel_order": "RGB",
            "value_range": [0.0, 1.0],
            "note": "Feed letterboxed RGB in [0,1]. ImageNet normalization is "
                    "BAKED INTO the graph (do NOT normalize in C++).",
        },
        "normalization_baked_in": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "letterbox": {
            "target": S,
            "aspect_preserving": True,
            "pad_value": 0.0,
            "pad_location": "centered",
            "scale": "min(S/W, S/H)",
            "note": "Boxes come back in SxS letterbox pixels; undo with "
                    "x_full=(x_letterbox - pad_x)/scale.",
        },
        "outputs": {
            "boxes": {"index": out_order["boxes_index"], "shape": ["n", 4],
                      "format": "xyxy", "space": f"{S}x{S} letterbox pixels",
                      "dtype": "float32"},
            "labels": {"index": out_order["labels_index"], "shape": ["n"],
                       "dtype": "int64",
                       "note": "COCO 1-based class ids (0==background dropped)."},
            "scores": {"index": out_order["scores_index"], "shape": ["n"],
                       "dtype": "float32"},
            "identify_by": out_order["identify_by"],
        },
        "nms_in_graph": True,
        "nms_iou_thresh": args.nms_thresh,
        "export_score_thresh": args.score_thresh,
        "person_class_index": COCO_PERSON_CLASS,
    }
    sidecar_path = os.path.splitext(args.out)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Wrote {sidecar_path}  ({size_mb:.1f} MB ONNX)")

    # Smoke-test under onnxruntime.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"images": np.random.rand(3, S, S).astype(np.float32)})
        print("onnxruntime smoke test OK; output shapes:", [o.shape for o in outs])
    except Exception as e:  # noqa: BLE001
        print("WARNING: onnxruntime smoke test failed:", e)


if __name__ == "__main__":
    main()
