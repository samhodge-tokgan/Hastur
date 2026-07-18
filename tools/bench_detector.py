#!/usr/bin/env python3
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# bench_detector.py -- bbox-parity benchmark for the M1 person detector.
#
# Runs BOTH the reference Detectron2 ViTDet-H cascade detector (ground truth for
# this study) and one or more candidate ONNX detectors on the reference's
# demo/test images, then reports per-image person-count match and Hungarian-
# matched IoU / recall / precision of each candidate vs ViTDet-H. Also reports
# each model's on-disk size (MB) and mean onnxruntime latency so the
# size/accuracy tradeoff is explicit.
#
# The ONNX preprocessing here (letterbox into a fixed square, RGB in [0,1],
# normalization baked into the graph) mirrors src/DetectorEngine.cpp exactly.
#
# Usage:
#   env/bin/python tools/bench_detector.py \
#       --onnx ssdlite=models/person_detector.onnx \
#       --onnx frcnn_r50_v2=models/frcnn_r50_v2.onnx \
#       --report tools/DETECTOR_REPORT.md

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sam/Documents/github/sam-3d-body")

PERSON_CLASS = 1  # torchvision COCO models


def find_images():
    imgs = sorted(
        glob.glob(
            "/Users/sam/Documents/github/sam-3d-body/assets/"
            "qualitative_comparisons/*/input_bbox.png"
        )
    )
    dancing = "/Users/sam/Documents/github/sam-3d-body/notebook/images/dancing.jpg"
    if os.path.exists(dancing):
        imgs.append(dancing)
    return imgs


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def hungarian_match(gt, pred, iou_thr=0.5):
    """Return (matched_pairs, ious) using optimal IoU assignment."""
    if len(gt) == 0 or len(pred) == 0:
        return [], []
    cost = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            cost[i, j] = iou(g, p)
    from scipy.optimize import linear_sum_assignment

    ri, ci = linear_sum_assignment(-cost)
    pairs, ious = [], []
    for i, j in zip(ri, ci):
        if cost[i, j] >= iou_thr:
            pairs.append((i, j))
            ious.append(cost[i, j])
    return pairs, ious


# --- letterbox preprocessing, identical to DetectorEngine.cpp -----------------
def letterbox(rgb01, S):
    import cv2

    H, W = rgb01.shape[:2]
    scale = min(S / W, S / H)
    nw, nh = round(W * scale), round(H * scale)
    px, py = (S - nw) // 2, (S - nh) // 2
    resized = cv2.resize(rgb01, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((S, S, 3), np.float32)
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, scale, px, py


def unletterbox(box, scale, px, py, W, H):
    x0 = (box[0] - px) / scale
    y0 = (box[1] - py) / scale
    x1 = (box[2] - px) / scale
    y1 = (box[3] - py) / scale
    return [max(0, min(W, x0)), max(0, min(H, y0)),
            max(0, min(W, x1)), max(0, min(H, y1))]


def identify_outputs(sess):
    """torchvision SSD returns [boxes,scores,labels] but R-CNN/RetinaNet return
    [boxes,labels,scores]. Identify by tensor type: labels==int64, scores==float
    1-D, boxes==float N×4. Mirrors the dtype-based dispatch in DetectorEngine.cpp."""
    bi = li = si = None
    for i, o in enumerate(sess.get_outputs()):
        if "int" in o.type:
            li = i
        elif len(o.shape) == 2:
            bi = i
        else:
            si = i
    return bi, li, si


def run_onnx(sess, S, bgr, score_thr, idx):
    import cv2

    bi, li, si = idx
    H, W = bgr.shape[:2]
    rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    canvas, scale, px, py = letterbox(rgb01, S)
    nchw = np.transpose(canvas, (2, 0, 1)).copy()  # [3,S,S]
    t = time.time()
    outs = sess.run(None, {sess.get_inputs()[0].name: nchw})
    dt = time.time() - t
    boxes, labels, scores = outs[bi], outs[li], outs[si]
    out = []
    for b, l, s in zip(boxes, labels, scores):
        if int(l) == PERSON_CLASS and s >= score_thr:
            out.append(unletterbox(b, scale, px, py, W, H))
    return out, dt


def get_gt(images, cache):
    if cache and os.path.exists(cache):
        data = json.load(open(cache))
        print(f"Loaded GT cache: {cache}")
        return {p: np.asarray(v["boxes"], float).reshape(-1, 4) for p, v in data.items()}
    import cv2

    from tools.build_detector import HumanDetector

    det = HumanDetector(name="vitdet", device="cpu")
    gt, dump = {}, {}
    for p in images:
        im = cv2.imread(p)
        boxes = np.asarray(
            det.run_human_detection(im, det_cat_id=0, bbox_thr=0.5,
                                    default_to_full_image=False), float
        ).reshape(-1, 4)
        gt[p] = boxes
        dump[p] = {"boxes": boxes.tolist(), "hw": [im.shape[0], im.shape[1]]}
        print(f"ViTDet-H {os.path.basename(os.path.dirname(p))}: {len(boxes)} boxes")
    if cache:
        json.dump(dump, open(cache, "w"), indent=2)
    return gt


def main():
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="append", default=[],
                    help="label=path.onnx (repeatable)")
    ap.add_argument("--gt-cache", default="")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--report", default="tools/DETECTOR_REPORT.md")
    args = ap.parse_args()

    images = find_images()
    print(f"{len(images)} benchmark images")
    gt = get_gt(images, args.gt_cache)
    total_gt = sum(len(gt[p]) for p in images)

    results = {}
    for spec in args.onnx:
        label, path = spec.split("=", 1)
        sidecar = os.path.splitext(path)[0] + ".json"
        S = json.load(open(sidecar))["input"]["shape"][-1] if os.path.exists(sidecar) else 800
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        idx = identify_outputs(sess)
        size_mb = os.path.getsize(path) / 1e6

        per_img, lat, tp, fp, sum_iou, matched = [], [], 0, 0, 0.0, 0
        pred_total = 0
        for p in images:
            preds, dt = run_onnx(sess, S, __import__("cv2").imread(p), args.score_thr, idx)
            lat.append(dt)
            g = gt[p]
            pairs, ious = hungarian_match(g, preds, args.iou_thr)
            tp += len(pairs)
            fp += len(preds) - len(pairs)
            sum_iou += sum(ious)
            matched += len(pairs)
            pred_total += len(preds)
            per_img.append((os.path.basename(os.path.dirname(p)) or os.path.basename(p),
                            len(g), len(preds), len(preds) == len(g),
                            np.mean(ious) if ious else 0.0))
        recall = tp / total_gt if total_gt else 0.0
        precision = tp / pred_total if pred_total else 0.0
        results[label] = dict(
            S=S, size_mb=size_mb, mean_lat=float(np.mean(lat)),
            recall=recall, precision=precision,
            mean_iou=(sum_iou / matched if matched else 0.0),
            count_match=sum(1 for r in per_img if r[3]), n_img=len(images),
            per_img=per_img,
        )
        print(f"[{label}] S={S} size={size_mb:.1f}MB recall={recall:.3f} "
              f"IoU={results[label]['mean_iou']:.3f} lat={np.mean(lat)*1000:.0f}ms")

    write_report(args.report, images, gt, total_gt, results, args)


def write_report(path, images, gt, total_gt, results, args):
    lines = []
    lines.append("# M1 Person-Detector Parity Report\n")
    lines.append("Track A / Milestone M1 — replacing the export-hostile Detectron2 "
                 "ViTDet-H person detector with a clean, commercial-licensed ONNX detector.\n")
    lines.append(f"- Reference (ground truth): **Detectron2 cascade_mask_rcnn_vitdet_h** "
                 f"(person boxes, score>0.5)\n")
    lines.append(f"- Benchmark images: {len(images)} multi-person images from the "
                 f"sam-3d-body clone (`assets/qualitative_comparisons/*/input_bbox.png`, "
                 f"`notebook/images/dancing.jpg`)\n")
    lines.append(f"- Total ViTDet-H person boxes across images: **{total_gt}**\n")
    lines.append(f"- Matching: Hungarian (optimal IoU), match threshold IoU>={args.iou_thr}, "
                 f"candidate score>={args.score_thr}\n")

    lines.append("\n## Summary (candidate vs ViTDet-H)\n")
    lines.append("| Model | Input | Size (MB) | Latency (ms, CPU) | Count-match | "
                 "Recall | Precision | Mean IoU |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for label, r in results.items():
        lines.append(f"| {label} | {r['S']}² | {r['size_mb']:.1f} | "
                     f"{r['mean_lat']*1000:.0f} | {r['count_match']}/{r['n_img']} | "
                     f"{r['recall']:.3f} | {r['precision']:.3f} | {r['mean_iou']:.3f} |\n")

    for label, r in results.items():
        lines.append(f"\n## {label} — per-image\n")
        lines.append("| Image | ViTDet-H | Candidate | Count match | Mean IoU(matched) |\n")
        lines.append("|---|---|---|---|---|\n")
        for name, ng, npd, ok, miou in r["per_img"]:
            lines.append(f"| {name} | {ng} | {npd} | {'yes' if ok else 'NO'} | {miou:.3f} |\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print("Wrote", path)


if __name__ == "__main__":
    main()
