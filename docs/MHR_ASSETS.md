# MHR mesh assets — provenance & format (Track C / M2)

This documents how Hastur's C++ MHR body model (`src/MhrModel.{h,cpp}`) replaces
the shipped 664 MB TorchScript `mhr_model.pt` with a small static asset binary
plus one ONNX for the single learned nonlinearity.

## Did clean extraction succeed?

**Yes — fully, from the TorchScript's own named buffers.** No fabrication, no
fallback. The shipped `checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt` is a
`torch.jit` trace of the momentum/MHR *character* model, and every static buffer
we need is exposed cleanly as a `named_buffer` / `named_parameter` (verified via
`torch.jit.load(...).named_buffers()`). The momentum asset format maps 1:1 onto
these buffers (`character_torch.{skeleton,mesh,blend_shape,parameter_transform,
linear_blend_skinning}`), so we did **not** need the separate facebookresearch/MHR
or facebookresearch/momentum asset release — the TorchScript *is* the asset source.

The MHR-head buffers that live outside the TorchScript (scale/hand PCA bases,
`keypoint_mapping`) come from the training checkpoint `model.ckpt`, keys
`head_pose.*` (the body head; `head_pose_hand.*` is the wrist-centric hand variant
and is not used on the body mesh path).

Extraction is done by `tools/extract_mhr_assets.py` -> `mhr_assets.bin` (~35 MB).

## The one thing that is NOT a static buffer

The **pose-corrective MLP** (`pose_correctives_model`) is the only learned
nonlinearity in the mesh path and is ~663 MB of the 664 MB TorchScript (a dense
`Linear(3000 -> 55317)`). It ships separately as `pose_corrective.onnx`
(`tools/export_pose_corrective.py`): a Hastur ORT session runs it. Input is the
889-d `joint_parameters` (computed in C++ by `MhrModel::JointParameters`); output
is the per-vertex offset (cm, pre-flip) added to the unposed vertices.

## `mhr_assets.bin` blocks

Flat versioned binary (writer/format: `tools/mhr_binfmt.py`; reader:
`src/MeshAssets.cpp`). Header: magic `MHRA`, u32 version, u32 nblocks, then
nblocks × 104-byte block headers `{char name[48]; u32 dtype; u32 ndim;
i64 shape[4]; u64 offset; u64 nbytes}`, then 8-byte-aligned data.

| block | shape | dtype | source buffer |
|---|---|---|---|
| `base_shape` | (18439,3) | f32 | `character_torch.blend_shape.base_shape` |
| `shape_vectors` | (45,18439,3) | f32 | `character_torch.blend_shape.shape_vectors` |
| `faces` | (36874,3) | i32 | `character_torch.mesh.faces` |
| `uv` | (18439,2) | f32 | mesh UV set if present, else cylindrical unwrap of `base_shape` (see `extract_uv`) |
| `parameter_transform` | (889,249) | f32 | `character_torch.parameter_transform.parameter_transform` |
| `joint_translation_offsets` | (127,3) | f32 | `character_torch.skeleton.joint_translation_offsets` |
| `joint_prerotations` | (127,4) | f32 | `character_torch.skeleton.joint_prerotations` (quat xyzw) |
| `joint_parents` | (127,) | i32 | `character_torch.skeleton.joint_parents` |
| `pmi` | (2,266) | i64 | `character_torch.skeleton.pmi` (FK prefix-multiply src/tgt) |
| `pmi_buffer_sizes` | (4,) | i32 | `skeleton._pmi_buffer_sizes` = [65,56,62,83] |
| `inverse_bind_pose` | (127,8) | f32 | `linear_blend_skinning.inverse_bind_pose` (skel-state) |
| `skin_vert_indices` | (51337,) | i64 | `linear_blend_skinning.vert_indices_flattened` |
| `skin_joint_indices` | (51337,) | i32 | `linear_blend_skinning.skin_indices_flattened` |
| `skin_weights` | (51337,) | f32 | `linear_blend_skinning.skin_weights_flattened` (Σ/vert = 1.0) |
| `scale_mean` | (68,) | f32 | `head_pose.scale_mean` |
| `scale_comps` | (28,68) | f32 | `head_pose.scale_comps` |
| `hand_pose_mean` | (54,) | f32 | `head_pose.hand_pose_mean` |
| `hand_pose_comps` | (54,54) | f32 | `head_pose.hand_pose_comps` |
| `hand_joint_idxs_left/right` | (27,) | i32 | `head_pose.hand_joint_idxs_*` |
| `keypoint_mapping` | (308,18566) | f32 | `head_pose.keypoint_mapping` |

## Mesh math (skel-state = [tx,ty,tz, qx,qy,qz,qw, s])

Decoded exactly from the TorchScript submodule `.code` (see
`tools/mhr_meshmodule.py`, a pure-torch reimplementation validated bit-for-bit):

1. identity rest = `base_shape + Σ_n shape_vectors[n]·shape[n]`
2. `joint_parameters[889] = parameter_transform · [model_params[204], 0₄₅]`
3. per-joint local skel-state: `t = jp[:3] + offset`, `q = prerotation ⊗
   euler_xyz→quat(jp[3:6])`, `s = exp(jp[6]·ln2)`
4. FK: parallel prefix-multiply over the 4 pmi levels (double precision),
   `global[src] = multiply(global[tgt], global[src])`
5. `+ pose_corrective(joint_parameters)` (from `pose_corrective.onnx`)
6. LBS: `joint_state = multiply(global, inverse_bind_pose)`; per skin entry
   `skinned[v] += w · (t + q·(s·p))`
7. keypoints = `keypoint_mapping · [verts, joints]`, first 70
8. `/100` (cm→m) and `verts/joints[...,{1,2}] *= -1` (camera-frame flip)

## Oracle & validation

- `tools/export_mesh_oracle.py` → `mesh_oracle.onnx` (+ `.onnx.data`): ONE ONNX
  graph `pred[519] → (verts, joints, keypoints)` (dynamo/torch.export path; the
  legacy exporter emits an invalid graph on the scatter ops). Matches TorchScript
  to ≤7e-4 mm. NOT shipped — validation only. Also dumps `.npz` fixtures.
- `tests/mhr_validate.cpp` / `tools/validate_e2e.py`: C++ `MhrModel` vs the oracle.
  **Worst per-vertex RMSE = 0.0002 mm, max = 0.0006 mm** (float32 round-off; the
  target was < 1 mm). `pose_corrective.onnx` via ORT matches TorchScript to 4e-7 cm.
