# M1 Person-Detector Parity Report

**Track A / Milestone M1** — replace the export-hostile Detectron2 ViTDet-H
person detector with a clean, commercial-licensed ONNX detector. The SAM-3D-Body
pipeline only consumes person **bounding boxes**, so any accurate person detector
works; the goal is bbox parity with ViTDet-H, a small footprint, and a clean ONNX
export.

## Recommendation

**Ship `ssdlite320_mobilenet_v3_large` (torchvision), exported to
`person_detector.onnx` (14 MB, fixed 320×320, in-graph NMS).**

- **License: BSD-3-Clause** (torchvision code + redistributed COCO-trained
  weights) — commercial-friendly. No AGPL (Ultralytics YOLO was *not* used).
- Matches the heavyweight ceiling on **recall (0.714)** at **1/12th the size**
  (14 MB vs 175 MB) and **~100× lower latency**, with high localization quality
  (mean IoU 0.90 vs ViTDet-H) — well above the crop threshold the downstream HMR
  stage needs.
- Clean, self-contained ONNX: normalization + NMS baked into the graph, runs
  unmodified under onnxruntime with the CoreML EP.

The exporter (`tools/export_detector.py`) also supports a heavier
`--model frcnn_r50_v2` profile if a deployment needs maximum crowd/small-person
recall (see risks).

## Setup

- **Reference / ground truth:** Detectron2 `cascade_mask_rcnn_vitdet_h`
  (auto-downloaded `model_final_f05665.pkl`), person boxes at score > 0.5, run on
  CPU via the reference `tools/build_detector.py`.
- **Benchmark images (5):** the reference clone's own demo/test images —
  `assets/qualitative_comparisons/sample{1..4}/input_bbox.png` and
  `notebook/images/dancing.jpg`. Total ViTDet-H person boxes: **7**.
- **Matching:** Hungarian (optimal IoU) assignment, match threshold IoU ≥ 0.5,
  candidate score ≥ 0.5.
- **Preprocessing (identical in `bench_detector.py` and `src/DetectorEngine.cpp`):**
  aspect-preserving letterbox into the fixed square, RGB in [0,1], ImageNet
  normalization baked into the graph.
- Latency is mean single-image onnxruntime **CPU** wall-time (the shipping engine
  uses CoreML/CUDA); measured on the M1 host.

## Results (candidate vs ViTDet-H)

| Model | Input | Size (MB) | Latency (ms, CPU) | Count-match | Recall | Precision | Mean IoU |
|---|---|---|---|---|---|---|---|
| **ssdlite320_mobilenet_v3_large** | 320² | **14.1** | **~135** | 3/5 | **0.714** | **1.000** | 0.899 |
| fasterrcnn_resnet50_fpn_v2 (ceiling) | 800² | 175.0 | ~13240 | 4/5 | 0.714 | 0.625 | 0.957 |

### ssdlite320 — per-image

| Image | ViTDet-H | ssdlite | Count match | Mean IoU (matched) |
|---|---|---|---|---|
| sample1 | 2 | 1 | NO | 0.837 |
| sample2 | 2 | 1 | NO | 0.871 |
| sample3 | 1 | 1 | yes | 0.943 |
| sample4 | 1 | 1 | yes | 0.938 |
| dancing | 1 | 1 | yes | 0.904 |

### fasterrcnn_resnet50_fpn_v2 — per-image

| Image | ViTDet-H | frcnn_r50_v2 | Count match | Mean IoU (matched) |
|---|---|---|---|---|
| sample1 | 2 | 2 | yes | 0.963 |
| sample2 | 2 | 3 | NO | 0.909 |
| sample3 | 1 | 1 | yes | 0.986 |
| sample4 | 1 | 1 | yes | 0.953 |
| dancing | 1 | 1 | yes | 0.976 |

Both candidates land at the **same recall (5/7)** vs ViTDet-H. The two boxes
neither small model reproduces exactly are ViTDet-H's *secondary* people; ssdlite
trades a little localization IoU (0.90 vs 0.96) and a larger model trades
precision (extra false positives), for a **12× smaller / ~100× faster** model.

## Candidates considered

| Candidate | License | Verdict |
|---|---|---|
| **ssdlite320_mobilenet_v3_large** (torchvision) | BSD-3 | **Chosen** — tiny, fast, clean ONNX, recall matches the ceiling. |
| fasterrcnn_resnet50_fpn_v2 (torchvision) | BSD-3 | Accuracy ceiling / optional heavy profile; 175 MB, 13 s CPU — too heavy for the default. |
| fasterrcnn_mobilenet_v3_large_fpn (torchvision) | BSD-3 | **Rejected** — exports but its ONNX **crashes at runtime** (Reshape `{N,363}`→`{-1,4}`, a torchvision mobilenet-FRCNN export bug). Export-fragile. |
| RTMDet-tiny/s/m (mmdetection) | Apache-2.0 | **Not attempted** — best size/accuracy on paper, but `mmcv` 2.2.0 has no build for the env's **torch 2.10** (would need risky source compilation) and disk was tight (~11 GB free). Revisit if the env is pinned to a torch/mmcv-supported combo. |
| Ultralytics YOLO | AGPL-3.0 | **Forbidden** (commercial plugin). |

## Risks / caveats

1. **Small / secondary-person recall at 320².** ssdlite misses 2 of 7 ViTDet-H
   people. Diagnosis: (a) sample1's second person (17% of frame) *is* detected but
   at score **0.11** — recoverable by lowering the score threshold at some
   precision cost; (b) sample2's second person is **31×61 px (0.2% of frame)** and
   becomes ~9×17 px after the 320² letterbox — effectively invisible to this model.
   For SAM-3D-Body this is largely benign (a 3D body mesh can't be fit to a
   ~10-px blob anyway), but **for crowded scenes / distant people, use the
   `frcnn_r50_v2` profile or lower the score threshold.**
2. **Bbox distribution shift vs ViTDet-H.** ssdlite boxes are close (mean IoU
   0.90) but not identical; the downstream crop uses `GetBBoxCenterScale` with
   padding, which absorbs box-size differences, so risk to the HMR crop is low.
   Worth a visual spot-check once the crop stage is wired (M2+).
3. **Weights provenance.** torchvision COCO weights are redistributed under
   torchvision's BSD-3 license; underlying COCO images are Flickr/CC. This is the
   standard commercial-use posture but should get a final legal sign-off before
   release.

## Reproduce

```bash
PY=/Users/sam/Documents/github/sam-3d-body/env/bin/python
# export the shipping detector + JSON sidecar
$PY tools/export_detector.py --model ssdlite --out models/person_detector.onnx
# parity benchmark vs ViTDet-H (runs the reference detector too; first run downloads it)
$PY tools/bench_detector.py \
    --onnx ssdlite=models/person_detector.onnx \
    --onnx frcnn_r50_v2=/path/to/frcnn_r50_v2.onnx \
    --report tools/DETECTOR_REPORT.md
```
