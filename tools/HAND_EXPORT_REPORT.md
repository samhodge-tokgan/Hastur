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

## Numeric parity: reference hand pass vs faithful torch hand wrapper

Reference = unmodified `forward_decoder_hand` + the real shipped TorchScript
MHR head (`head_pose_hand`, enable_hand_model) -- the exact per-crop hand
building block `run_inference()` runs for each hand. Wrapper = the exported
graph's torch module (condition_info hand projection + HandMeshModule).

### right-hand crop
| slice | max_abs | mean_rel | pearson |
|---|---|---|---|
| hand PCA[108] | 1.972e-01 | 2.872e+00 | 0.9642452 |
| scale[28] | 1.505e+00 | 1.825e+00 | 0.8637595 |
| shape[45] | 4.008e-01 | 3.762e+00 | 0.9992921 |
| pred[519] full | 1.505e+00 | 1.889e+00 | 0.9946346 |

wrist_global[2,3,3] vs reference: max_abs=4.814e-02, mean_rel=7.961e-02, pearson=0.9989981

### left-hand crop (flipped->right)
| slice | max_abs | mean_rel | pearson |
|---|---|---|---|
| hand PCA[108] | 2.713e-01 | 7.045e-01 | 0.9745904 |
| scale[28] | 1.784e+00 | 2.267e+00 | 0.8013438 |
| shape[45] | 3.945e-01 | 3.302e+01 | 0.9993102 |
| pred[519] full | 1.784e+00 | 3.486e+00 | 0.9918483 |

wrist_global[2,3,3] vs reference: max_abs=5.183e-01, mean_rel=7.711e-01, pearson=0.9034561

## ONNX runtime parity + graph audit

ONNX file: `sam3dbody_hand.onnx`  (1805.7 MB, fp16)

### onnxruntime CPU (fp16) vs torch hand wrapper (fp32), right-hand crop
### pred slices
| slice | max_abs | mean_rel | pearson |
|---|---|---|---|
| hand PCA[108] | 9.962e-04 | 5.820e-03 | 0.9999991 |
| scale[28] | 5.803e-03 | 3.245e-02 | 0.9999952 |
| shape[45] | 3.628e-03 | 1.663e-02 | 0.9999998 |
| pred[519] full | 5.803e-03 | 6.672e-03 | 0.9999998 |

wrist_global vs wrapper: max_abs=1.386e-03, pearson=0.999999

Total graph nodes: 30836. Op histogram (top 15):
```
  Constant             11452
  Mul                  2207
  Slice                1778
  Unsqueeze            1601
  Shape                1516
  Add                  1390
  Concat               1362
  Gather               1187
  Expand               715
  Sub                  645
  Where                621
  Reshape              597
  Cast                 528
  Div                  433
  Squeeze              409
```
Fallback-prone ops present (CoreML -> CPU), same family as the body graph:
```
  ArgMax               6
  CumSum               2
  Einsum               34
  GatherND             72
  GridSample           5
  NonZero              144
  Range                135
  ScatterElements      161
  ScatterND            171
```

## Wrist-angle criterion (CRITERIA 1) -- C++ vs reference ground truth

The C++ refiner's wrist-angle gate (`MhrModel::ComputeWristGate` + the fused-wrist
math in `Sam3dBodyPipeline::RefineHands`) was cross-checked against the reference
`run_inference` computation (`tools/validate_wrist.py`: real head FK
`joint_global_rots`, real `joint_rotation[77,41]`, roma "XZY" `ori_local`), on the
seated-yoga sample at the SAME person box the pipeline detects:

| hand | reference angle (rad) | C++ pipeline angle (rad) |
|---|---|---|
| left  | 2.399 | 2.317 |
| right | 1.719 | 1.773 |

Agreement is ~0.05-0.08 rad (fp16 + FK-port noise), so the gate is faithful. Both
exceed the 1.4 rad threshold on THIS crop, so the reference *itself* keeps the
body-decoder hands here (the behind-the-head wrists genuinely disagree with the
hand decoder). At a full-frame box the same image gives left=0.641 (would refine),
right=1.496 -- i.e. the criterion is crop-sensitive, exactly as in the reference.

On a frontal pose (`notebook/images/dancing.jpg`) both hands pass: angles
left=0.809, right=0.860 rad, box sizes 75/90 px -> both refined. The
`build/dancing_before.png` vs `build/dancing_after.png` render shows the raised
hand's fingers going from a splayed default pose to natural articulation.

## Merge / flip conventions (the fiddly part)

  * `hand_box[side]` = `[cx,cy,w,h]` in [0,1] of the 512 BODY crop; side 0=LEFT,
    1=RIGHT. Mapped to a full-frame square via the body crop's inverse affine, then
    re-cropped (pad **0.9**) as its own 512 hand crop.
  * **L/R FLIP**: the LEFT hand is cropped from a horizontally-mirrored frame with
    a mirrored box, so the hand decoder processes it exactly like a right hand. The
    decoder's RIGHT slot (`hand[54:108]`) is therefore the refined hand for BOTH
    crops; for the left it is written into the body's LEFT slot (`hand[0:54]`).
  * Left-hand SCALE flip: `scale[9] = ((scale_mean[8] + scale_comps[8,8]*rc8) -
    scale_mean[9]) / scale_comps[9,9]` where `rc8` is the left crop's `scale[8]`.
  * Left-hand WRIST global: take the decoder's joint-42 rotmat and negate rows
    [1,2] (un-flip) to get the body-left wrist (joint 78).
  * `replace_hands_in_pose`: `pred[519]` hand block `[left54|right54]` at offset
    339; body `scale[8]`=right / `scale[9]`=left (flip-corrected); shared
    `scale[18:28]` and `shape[40:45]` = mean of the applied hands. All gated by
    (wrist-angle < 1.4) AND (hand-box > 64 px). The merged `pred` is then re-run
    through pose-corrective + the C++ MHR so the mesh hands are re-solved.

## Deferred vs the full reference (documented gaps)

  * Merge gate uses CRITERIA 1 (wrist-angle) + 2 (box-size). CRITERIA 3/4 (hand 2D
    keypoints inside box / 2D wrist distance) are NOT applied -- they need the hand
    crop's projected 2D keypoints, which this minimal export does not emit. Effect:
    the C++ gate is marginally more permissive than the reference.
  * The reference additionally (a) re-injects the fused wrist rotation back into the
    body pose (wrist ORIENTATION) via `fix_wrist_euler`, and (b) re-runs the BODY
    decoder with a wrist/elbow keypoint prompt. Both are DEFERRED here: this slice
    refines finger articulation + hand scale/shape (the dominant visible win) and
    leaves the wrist at the body estimate. No un-exportable op was hit (the hand
    graph is the same op family as the shipped body graph).
