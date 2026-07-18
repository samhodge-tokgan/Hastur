#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# validate_hand.py -- numeric parity for the exported hand-decoder pass (M7).
# Writes HAND_EXPORT_REPORT.md.
#
# Checks (mirrors validate_body.py):
#   1. torch hand wrapper (HandMeshModule refinement + condition_info projection)
#      vs the PyTorch reference hand pass (unmodified forward_decoder_hand + real
#      TorchScript MHR head) -- proves the substitutions on the HAND params.
#   2. onnxruntime CPU (fp16 ONNX) vs the torch wrapper -- proves the export.
#   3. graph op histogram + CoreML fallback-op audit.
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_hand as H  # noqa: E402
import export_sam3dbody as B  # noqa: E402

REPORT = os.path.join(HERE, "HAND_EXPORT_REPORT.md")


def stats(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    mad = float(np.max(np.abs(a - b)))
    rel = float(np.mean(np.abs(a - b) / (np.abs(b) + 1e-6)))
    p = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
    return mad, rel, p


def _all_nodes(graph):
    for n in graph.node:
        yield n
        for a in n.attribute:
            if a.HasField("g"):
                yield from _all_nodes(a.g)
            for sub in a.graphs:
                yield from _all_nodes(sub)


# The 519-d slices we care about for the merge.
HAND_OFF = 6 + 260 + 45 + 28       # start of the 108 hand PCA params
SCALE_OFF = 6 + 260 + 45
SHAPE_OFF = 6 + 260


def slice_report(name, wrap_pred, ref_pred, log):
    """Report parity on the merge-relevant slices of pred[519]."""
    def sl(p, a, b):
        return np.asarray(p).ravel()[a:b]
    rows = [
        ("hand PCA[108]", HAND_OFF, HAND_OFF + 108),
        ("scale[28]", SCALE_OFF, SCALE_OFF + 28),
        ("shape[45]", SHAPE_OFF, SHAPE_OFF + 45),
        ("pred[519] full", 0, 519),
    ]
    log(f"### {name}")
    log("| slice | max_abs | mean_rel | pearson |")
    log("|---|---|---|---|")
    for label, a, b in rows:
        mad, rel, p = stats(sl(wrap_pred, a, b), sl(ref_pred, a, b))
        log(f"| {label} | {mad:.3e} | {rel:.3e} | {p:.7f} |")
    log("")


def run_parity(img_path=None, onnx_path=None, write=True):
    torch.manual_seed(0)
    lines = []
    log = lambda s="": (print(s), lines.append(s))
    onnx_path = onnx_path or H.DEFAULT_OUT

    # Build a representative right-hand crop from the body decoder's hand box.
    model, cfg = B.load_model()
    if img_path is None:
        img_path = os.path.join(B.REF_ROOT, "notebook/images/dancing.jpg")
    try:
        boxes = H._hand_box_from_body(model, cfg, img_path)
    except Exception as e:
        log(f"(hand-box probe failed: {e}; using center box)")
        import cv2
        im = cv2.imread(img_path); Hh, Ww = im.shape[:2]
        boxes = np.array([[Ww*.4, Hh*.4, Ww*.6, Hh*.6],
                          [Ww*.4, Hh*.4, Ww*.6, Hh*.6]], np.float32)

    # ---- 1: torch wrapper vs reference hand pass (both hand crops) ----
    log("## Numeric parity: reference hand pass vs faithful torch hand wrapper\n")
    log("Reference = unmodified `forward_decoder_hand` + the real shipped TorchScript")
    log("MHR head (`head_pose_hand`, enable_hand_model) -- the exact per-crop hand")
    log("building block `run_inference()` runs for each hand. Wrapper = the exported")
    log("graph's torch module (condition_info hand projection + HandMeshModule).\n")

    faith_feeds = None
    faith_pred = None
    faith_wrist = None
    for side, label in ((1, "right-hand crop"), (0, "left-hand crop (flipped->right)")):
        box = boxes[side:side + 1]

        model, cfg = B.load_model()
        batch = H.build_hand_batch(box, img_path=img_path)[0]
        ref = H.run_reference_hand(model, cfg, batch)

        model, cfg = B.load_model()
        mesh = H.build_hand_mesh(model)
        wrap = H.HandWrapper(model, cfg, mesh).eval()
        image, ray32, cond = H.make_onnx_inputs(model, cfg,
                                                H.build_hand_batch(box, img_path=img_path)[0])
        with torch.no_grad():
            wp, wwrist = wrap(image, ray32, cond)
        slice_report(label, wp.numpy(), ref["pred"].numpy(), log)
        mad, rel, p = stats(wwrist.numpy(), ref["wrist_global"].numpy())
        log(f"wrist_global[2,3,3] vs reference: max_abs={mad:.3e}, "
            f"mean_rel={rel:.3e}, pearson={p:.7f}\n")
        if side == 1:
            faith_feeds = {"image": image.numpy().astype(np.float16),
                           "ray_cond": ray32.numpy().astype(np.float16),
                           "condition_info": cond.numpy().astype(np.float16)}
            faith_pred = wp.numpy()
            faith_wrist = wwrist.numpy()

    # ---- 2+3: ONNX CPU parity + graph audit ----
    if onnx_path and os.path.exists(onnx_path) and faith_feeds is not None:
        import onnx
        import onnxruntime as ort
        sz = os.path.getsize(onnx_path) / 1e6
        log(f"## ONNX runtime parity + graph audit\n")
        log(f"ONNX file: `{os.path.basename(onnx_path)}`  ({sz:.1f} MB, fp16)\n")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outs = sess.run(H.OUTPUT_NAMES, faith_feeds)
        opred, owrist = outs[0], outs[1]
        log("### onnxruntime CPU (fp16) vs torch hand wrapper (fp32), right-hand crop")
        slice_report("pred slices", opred, faith_pred, log)
        mad, rel, p = stats(owrist, faith_wrist)
        log(f"wrist_global vs wrapper: max_abs={mad:.3e}, pearson={p:.6f}\n")

        m = onnx.load(onnx_path, load_external_data=False)
        from collections import Counter
        opc = Counter(nd.op_type for nd in _all_nodes(m.graph))
        log(f"Total graph nodes: {sum(opc.values())}. Op histogram (top 15):")
        log("```")
        for op, n in opc.most_common(15):
            log(f"  {op:20s} {n}")
        log("```")
        COREML_WEAK = {"GridSample", "Atan2", "ScatterND", "ScatterElements",
                       "GatherND", "ArgMax", "CumSum", "Einsum", "NonZero",
                       "Range", "Mod", "Round"}
        present = [(op, opc[op]) for op in sorted(COREML_WEAK) if op in opc]
        if present:
            log("Fallback-prone ops present (CoreML -> CPU), same family as the body graph:")
            log("```")
            for op, n in present:
                log(f"  {op:20s} {n}")
            log("```")
    else:
        log("## ONNX runtime parity\n")
        log(f"(skipped -- {onnx_path} not found; run export_hand.py first)")

    if write:
        with open(REPORT, "w") as f:
            f.write(SUMMARY + "\n")
            f.write("\n".join(lines) + "\n")
        print(f"\n[validate-hand] wrote {REPORT}")
    return lines


SUMMARY = """\
# SAM 3D Body -- hand-decoder export report (M7)

## Summary

`models/sam3dbody_hand.onnx` -- a fixed-shape (batch=1, 512x512) fp16 ONNX of the
SAM-3D-Body **hand-decoder** 2nd pass, the direct analogue of the body export:

    (image[1,3,512,512], ray_cond[1,2,32,32], condition_info[1,3])   # a HAND crop
      -> pred[1,519]              (hand-decoder MHR-head param vector)
         wrist_global[1,2,3,3]    (global wrist rotmats, joints [78 L, 42 R])

Graph: DINOv3-vith16plus backbone (on the hand crop) -> +no-mask embed ->
`decoder_hand` (6 layers, dummy keypoint prompt, per-layer refinement driven by
`HandMeshModule`) -> `head_pose_hand` 519 vector + wrist global rotmats.

The C++ refiner slices the hand PCA[108] / scale[28] / shape[45] out of `pred`
and merges them into the body `pred[519]` (see the flip/merge conventions in
HandRefinerEngine / Sam3dBodyPipeline), gating on the wrist-angle criterion that
compares `wrist_global` against the body decoder's own FK wrist rotation.

Method / substitutions (identical family to the body export):
  * device-pinned `.to("mps")` helpers replaced with clean CPU/fp32 versions;
    `forward_decoder_hand` + the `*_hand` update fns are driven unmodified.
  * `camera_project_hand` reimplemented from `condition_info` alone (same CLIFF
    derivation validated to ~1e-7 for the body path).
  * the in-graph refinement + the wrist rotmats use `HandMeshModule` -- MhrMeshModule
    extended with the hand-head `mhr_forward` semantics (enable_hand_model): the
    `local_to_world_wrist` global-rot transfer, non-hand model-param zeroing, and
    right-hand-only (21:42) keypoints. Same scatter-add LBS + recursive fp16 fold.
  * CoreML: same interleaved MHR-refinement ops (GridSample / Scatter / GatherND /
    NonZero / Range) fragment the partition -> runs essentially on CPU under CoreML,
    NeuralNetwork format + CPU fallback (as the body engine).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=H.DEFAULT_OUT)
    ap.add_argument("--img", default=None)
    args = ap.parse_args()
    run_parity(args.img, args.onnx)


if __name__ == "__main__":
    main()
