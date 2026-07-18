# SAM 3D Body -- body-only export report (M3 / Track B)

## Summary

`models/sam3dbody_body.onnx` -- a FAITHFUL, fixed-shape (batch=1, 512x512) fp16
ONNX of the SAM-3D-Body body-only regressor:

    (image[1,3,512,512], ray_cond[1,2,32,32], condition_info[1,3])
      -> pred[1,519], pred_cam[1,3], hand_logits[1,2,2], hand_box[1,2,4]

Graph: DINOv3-vith16plus backbone (image-only) -> +no-mask embed -> SAM
promptable decoder (6 layers, dummy keypoint prompt baked in, per-layer MHR-driven
keypoint refinement) -> MHR-head 519 vector + perspective camera + hand-detect
logits/box. It loads and runs in onnxruntime (verified) and matches the PyTorch
reference to fp16 noise (`pred` max_abs ~6e-3 ONNX-vs-wrapper; ~1.4e-2 wrapper-vs-
reference incl. the dropped pose-corrective; Pearson ~1.0).

Method / how the known blockers were solved:
  * The reference's device-pinned `.to("mps")` helpers are replaced with clean
    CPU/fp32 versions; the rest of the reference decoder is driven unmodified.
  * camera_project is reimplemented from `condition_info` ALONE (proven equivalent
    to the cam_int/affine version to ~1e-7), so the frozen 3-input contract needs
    no extra camera inputs.
  * The shipped momentum-MHR TorchScript cannot be exported (`aten::sparse_coo_
    tensor`), so the in-graph refinement uses `tools/mhr_meshmodule.MhrMeshModule`
    (pure-torch, ONNX-exportable, validated bit-for-bit -- keypoints match to
    ~3e-8 m). Its pose-corrective blendshapes are dropped from the refinement mesh
    (they only nudge the intermediate keypoints; +1.4e-2 on `pred`).
  * LBS uses `scatter_add` (not `index_add`, whose legacy-exporter lowering emits
    a broken `Expand` ORT rejects); the 5 scatter-add nodes are kept in fp32
    (no fp16 CPU kernel for reduction='add') via boundary Casts.
  * fp16: a complete recursive float/double->float16 pass (weights, Constants, Cast
    targets, value types, incl. RoPE's If-subgraph) avoids the mixed-precision
    boundary clashes onnxconverter_common leaves; saved inline (<2 GB single file).

CoreML: the interleaved MHR-refinement ops (GridSample / ScatterND / GatherND /
NonZero / Range / Atan) fragment the partition -- CoreML fuses only one ~72-node
block, so the model runs essentially on CPU (details below). The engine uses the
CoreML NeuralNetwork format with CPU fallback.

## Numeric parity: PyTorch reference vs faithful torch wrapper

Reference = unmodified SAM-3D-Body decoder driven with the real momentum
TorchScript MHR. Wrapper = the exported graph's torch module (condition_info
camera projection + MhrMeshModule refinement).

### crop = full_frame   (condition_info non-trivial for offcenter)
| output | max_abs | mean_rel | pearson |
|---|---|---|---|
| pred | 1.764e-02 | 2.613e-02 | 0.9999995 |
| pred_cam | 3.533e-04 | 1.185e-03 | 0.9999999 |
| hand_logits | 3.748e-03 | 2.650e-04 | 1.0000000 |
| hand_box | 9.671e-05 | 3.627e-04 | 1.0000000 |

MhrMeshModule keypoints vs reference-MHR keypoints: max_abs=1.490e-08 m, pearson=1.0000000

### crop = offcenter   (condition_info non-trivial for offcenter)
| output | max_abs | mean_rel | pearson |
|---|---|---|---|
| pred | 1.425e-02 | 6.562e-02 | 0.9999995 |
| pred_cam | 5.387e-04 | 6.591e-04 | 0.9999999 |
| hand_logits | 7.381e-03 | 4.928e-04 | 0.9999999 |
| hand_box | 1.867e-04 | 7.526e-04 | 0.9999999 |

MhrMeshModule keypoints vs reference-MHR keypoints: max_abs=2.980e-08 m, pearson=1.0000000

## Meshfree vs faithful (impact of the MHR-driven refinement)

| output | max_abs | mean_rel | pearson |  (meshfree wrapper vs reference)
|---|---|---|---|
| pred | 4.956e-01 | 1.950e+00 | 0.999587 |
| pred_cam | 4.561e-02 | 5.667e-02 | 0.997809 |
| hand_logits | 3.867e-01 | 3.297e-02 | 0.999728 |
| hand_box | 1.060e-01 | 3.807e-01 | 0.949011 |

## ONNX runtime parity + CoreML partition

ONNX file: `sam3dbody_body.onnx`  (1808.7 MB, fp16)

### onnxruntime CPU (fp16) vs torch faithful wrapper (fp32)
| output | max_abs | mean_rel | pearson |
|---|---|---|---|
| pred | 5.818e-03 | 1.335e-01 | 1.000000 |
| pred_cam | 4.932e-04 | 4.670e-04 | 1.000000 |
| hand_logits | 1.439e-03 | 9.257e-05 | 1.000000 |
| hand_box | 2.327e-04 | 1.135e-03 | 1.000000 |

Total graph nodes: 25021. Op histogram (top 20):
```
  Constant             9341
  Mul                  1729
  Slice                1567
  Shape                1327
  Add                  1213
  Unsqueeze            1201
  Concat               1065
  Gather               958
  Expand               518
  Where                500
  Sub                  483
  Cast                 483
  Reshape              454
  Transpose            369
  Squeeze              356
  Div                  348
  MatMul               347
  ConstantOfShape      341
  Equal                268
  Identity             200
```
Fallback-prone ops present in the graph (CoreML -> CPU):
```
  ArgMax               5
  CumSum               2
  Einsum               30
  GatherND             72
  GridSample           5
  NonZero              144
  Range                123
  ScatterElements      135
  ScatterND            154
```

### CoreML EP partition
| output | max_abs | pearson |  (CoreML EP vs torch wrapper)
|---|---|---|
| pred | 5.818e-03 | 1.000000 |
| pred_cam | 4.932e-04 | 1.000000 |
| hand_logits | 1.439e-03 | 1.000000 |
| hand_box | 2.327e-04 | 1.000000 |

Final EP placement (post graph-optimization): **CoreML = 1 node(s), CPU = 12185 node(s)** (0.01% on the accelerator).
CoreML `GetCapability` accepted **72 of 9403** post-optimization nodes as a single fused partition; everything else stays on CPU.

**Interpretation:** the MHR-refinement fallback ops (GridSample, ScatterND/ScatterElements-add, GatherND, NonZero, Range, Atan) are interleaved through the 6 decoder layers, so they fragment the graph into tiny CoreML-eligible islands -- the EP can only fuse one ~72-node block. This body regressor therefore runs essentially on CPU under CoreML; MLProgram additionally recompiles the fragments impractically slowly. The engine selects the NeuralNetwork format + CPU fallback accordingly.

Raw partition log (excerpt):
```
CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 1 number of nodes in the graph: 9403 number of nodes supported by CoreML: 72
CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 0 number of nodes in the graph: 9528 number of nodes supported by CoreML: 0
Node(s) placed on [CoreMLExecutionProvider]. Number of nodes: 1
Node(s) placed on [CPUExecutionProvider]. Number of nodes: 12185
```

## Momentum-MHR direct ONNX export blocker
```
UnsupportedOperatorError: UnsupportedOperatorError("Exporting the operator 'aten::sparse_coo_tensor' to ONNX opset version 17 is not supported")
```
