#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# extract_mhr_assets.py -- extract the STATIC MHR buffers needed for the C++ LBS
# into a versioned flat binary `mhr_assets.bin` (+ mhr_assets.manifest.json).
#
# SOURCE OF ASSETS
# ----------------
# Every buffer below is a *clean named buffer* of the shipped TorchScript
# (checkpoints/.../assets/mhr_model.pt) or of the body head in the training
# checkpoint (model.ckpt, keys head_pose.*). Nothing is fabricated. The
# TorchScript is the momentum/MHR "character_torch" model; its buffers map 1:1
# onto the momentum asset format (skeleton, blend_shape, linear_blend_skinning,
# parameter_transform). See docs/MHR_ASSETS.md for the field-by-field provenance.
#
# The ONE large learned nonlinearity -- the pose-corrective MLP
# (pose_correctives_model, ~663 MB of the 664 MB TorchScript) -- is intentionally
# NOT packed here; it ships separately as pose_corrective.onnx
# (see export_pose_corrective.py). mhr_assets.bin is the small (~34 MB) static core.
import argparse
import os

import numpy as np
import torch

from mhr_binfmt import write_assets
from mhr_common import MHR_PT, load_head_buffers, load_mhr_ts


def np_f32(t):
    return t.detach().cpu().float().numpy()


def np_i32(t):
    return t.detach().cpu().to(torch.int32).numpy()


def np_i64(t):
    return t.detach().cpu().to(torch.int64).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mhr_assets.bin")
    ap.add_argument("--manifest", default="mhr_assets.manifest.json")
    args = ap.parse_args()

    m = load_mhr_ts()
    ct = m.character_torch
    B = dict(m.named_buffers())

    def buf(name):
        return B[name]

    def extract_uv(base_shape):
        """Per-vertex texture (u,v) for the ST AOV, shape (18439,2) f32.

        Prefers a real UV set from the MHR mesh if the checkpoint carries one;
        otherwise falls back to a deterministic cylindrical unwrap of the
        canonical `base_shape` (azimuth about the vertical axis -> u, normalized
        height -> v). The fallback is a "simple ST" good enough for texture
        projection / DensePose-style conditioning, not a seam-free atlas.
        """
        for name in ("character_torch.mesh.texcoords",
                     "character_torch.mesh.uvs",
                     "character_torch.mesh.uv"):
            if name in B:
                uv = np_f32(B[name])
                if uv.ndim == 2 and uv.shape == (18439, 2):
                    print(f"uv: using mesh UV set '{name}'")
                    return uv
        for attr in ("texcoords", "uvs", "uv"):
            t = getattr(getattr(ct, "mesh", object()), attr, None)
            if t is not None:
                uv = np_f32(t)
                if uv.ndim == 2 and uv.shape == (18439, 2):
                    print(f"uv: using mesh.{attr}")
                    return uv
        # Fallback: cylindrical unwrap of base_shape (x,y,z in cm, pre-flip).
        v = base_shape.reshape(-1, 3).astype(np.float64)
        u = (np.arctan2(v[:, 0], v[:, 2]) / (2.0 * np.pi)) + 0.5   # azimuth
        h = v[:, 1]
        t = (h - h.min()) / max(1e-6, (h.max() - h.min()))         # height
        print("uv: no mesh UV set found -> cylindrical unwrap fallback")
        return np.stack([u, t], axis=1).astype(np.float32)

    # ---- pmi buffer sizes (FK prefix-multiply level split) ----
    pmi_sizes = list(m.character_torch.skeleton._pmi_buffer_sizes)

    head = load_head_buffers()

    blocks = {
        # ---- mesh + identity blendshape (blend_shape) ----
        "base_shape":        np_f32(buf("character_torch.blend_shape.base_shape")),      # (18439,3)
        "shape_vectors":     np_f32(buf("character_torch.blend_shape.shape_vectors")),   # (45,18439,3)
        "faces":             np_i32(buf("character_torch.mesh.faces")),                  # (36874,3)
        "uv":                extract_uv(np_f32(buf("character_torch.blend_shape.base_shape"))),  # (18439,2) ST AOV
        # ---- parameter transform (model_params[249] -> joint_params[889]) ----
        "parameter_transform": np_f32(buf("character_torch.parameter_transform.parameter_transform")),  # (889,249)
        # ---- skeleton (local skel-state build + FK) ----
        "joint_translation_offsets": np_f32(buf("character_torch.skeleton.joint_translation_offsets")),  # (127,3)
        "joint_prerotations":        np_f32(buf("character_torch.skeleton.joint_prerotations")),         # (127,4) xyzw
        "joint_parents":             np_i32(buf("character_torch.skeleton.joint_parents")),              # (127,)
        "pmi":                       np_i64(buf("character_torch.skeleton.pmi")),                        # (2,266)
        "pmi_buffer_sizes":          np.asarray(pmi_sizes, dtype=np.int32),                              # (4,)
        # ---- linear blend skinning (sparse) ----
        "inverse_bind_pose":   np_f32(buf("character_torch.linear_blend_skinning.inverse_bind_pose")),        # (127,8)
        "skin_vert_indices":   np_i64(buf("character_torch.linear_blend_skinning.vert_indices_flattened")),   # (51337,)
        "skin_joint_indices":  np_i32(buf("character_torch.linear_blend_skinning.skin_indices_flattened")),   # (51337,)
        "skin_weights":        np_f32(buf("character_torch.linear_blend_skinning.skin_weights_flattened")),   # (51337,)
        # ---- MHRHead body-head buffers (from model.ckpt: head_pose.*) ----
        "scale_mean":          np_f32(head["scale_mean"]),        # (68,)
        "scale_comps":         np_f32(head["scale_comps"]),       # (28,68)
        "hand_pose_mean":      np_f32(head["hand_pose_mean"]),    # (54,)
        "hand_pose_comps":     np_f32(head["hand_pose_comps"]),   # (54,54)
        "hand_joint_idxs_left":  np_i32(head["hand_joint_idxs_left"]),   # (27,)
        "hand_joint_idxs_right": np_i32(head["hand_joint_idxs_right"]),  # (27,)
        "keypoint_mapping":    np_f32(head["keypoint_mapping"]),  # (308,18566)
    }

    # sanity: skin weights sum to ~1 per vertex
    vidx = blocks["skin_vert_indices"]
    w = blocks["skin_weights"]
    wsum = np.zeros(18439, dtype=np.float64)
    np.add.at(wsum, vidx, w)
    print(f"skin weight/vertex sum: min={wsum.min():.4f} max={wsum.max():.4f} "
          f"nnz={len(w)} verts_covered={(wsum>0).sum()}")

    manifest = write_assets(args.out, blocks, args.manifest)
    total = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({total/1e6:.1f} MB, {len(blocks)} blocks)")
    for b in manifest["blocks"]:
        print(f"  {b['name']:26s} {str(b['shape']):20s} {b['dtype']}")


if __name__ == "__main__":
    main()
