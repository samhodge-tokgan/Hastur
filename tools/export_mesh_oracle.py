#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# export_mesh_oracle.py -- the NUMERICAL ORACLE for the MHR mesh path.
#
#   pred[519] -> (verts[18439,3], joints[127,3], keypoints[70,3])   in metres,
#   with the /100 unit conversion and the [1,2] camera-frame flip baked in.
#
# Produces:
#   * mesh_oracle.onnx      -- ONE ONNX graph (from MhrMeshModule, a pure-torch
#                              reimplementation validated bit-for-bit vs the
#                              shipped TorchScript). NOT shipped; validation only.
#   * fixtures/*.npz        -- (pred519 -> verts/joints/keypoints) pairs PLUS the
#                              pose-corrective vertex offset (cm, pre-flip) so the
#                              C++ LBS test can feed it through the "hook".
#
# Ground-truth values come from MhrReference (the real TorchScript path); the
# ONNX/MhrMeshModule are checked against it.
import argparse
import os

import numpy as np
import torch

from mhr_common import MhrReference, compact_cont_to_model_params_body
from mhr_meshmodule import MhrMeshModule
from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body


def zero_pose():
    p = torch.zeros(1, 519)
    p[:, :6] = torch.tensor([1.0, 0, 0, 0, 1, 0])
    p[:, 6:266] = compact_model_params_to_cont_body(torch.zeros(1, 133))
    return p


def make_preds(n, seed=0):
    torch.manual_seed(seed)
    preds = [zero_pose()]  # canonical zero pose
    for i in range(n - 1):
        p = zero_pose()
        p[:, 6:266] += 0.06 * torch.randn(1, 260)
        p[:, 266:311] = 0.6 * torch.randn(1, 45)     # shape
        p[:, 311:339] = 0.3 * torch.randn(1, 28)     # scale
        p[:, 339:447] = 0.4 * torch.randn(1, 108)    # hands
        preds.append(p)
    return preds


def pose_corrective_cm(ref, mod, pred):
    """The pose-corrective vertex offset (cm, pre-flip) the C++ hook consumes."""
    model_params, shape = mod.decode(pred)
    full249 = torch.cat([model_params, torch.zeros(1, 45)], dim=1)
    jp = torch.einsum("dn,bn->bd", mod.parameter_transform, full249)
    return mod.pose_corrective(jp)[0]  # (V,3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../test-assets")
    ap.add_argument("--nfix", type=int, default=6)
    ap.add_argument("--onnx", action="store_true", help="also export mesh_oracle.onnx")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.outdir, "fixtures"), exist_ok=True)

    ref = MhrReference()
    mod = MhrMeshModule(ref).eval()

    from mhr_binfmt import write_assets

    preds = make_preds(args.nfix)
    maxv = 0.0
    fix_blocks = {"nfix": np.asarray([len(preds)], dtype=np.int32)}
    for i, p in enumerate(preds):
        o = ref.forward(p)
        v, j, k = mod(p)
        dv = (v - o["verts"]).abs().max().item() * 1000
        maxv = max(maxv, dv)
        corr = pose_corrective_cm(ref, mod, p)
        arrays = dict(pred=p[0].numpy().astype(np.float32),
                      verts=o["verts"][0].numpy().astype(np.float32),
                      joints=o["joints"][0].numpy().astype(np.float32),
                      keypoints=o["keypoints"][0].numpy().astype(np.float32),
                      pose_corrective=corr.numpy().astype(np.float32))
        np.savez(os.path.join(args.outdir, "fixtures", f"fix{i:02d}.npz"), **arrays)
        for name, arr in arrays.items():
            fix_blocks[f"{name}_{i:02d}"] = arr
        print(f"fixture {i}: max |mesh-onnx| = {dv:.6f} mm  (oracle=TorchScript)")
    # Flat binary the C++ validation test reads (MHRA format).
    write_assets(os.path.join(args.outdir, "fixtures.bin"), fix_blocks)
    print(f"wrote fixtures.bin ({len(preds)} fixtures)")
    print(f"MhrMeshModule vs TorchScript: max vert err = {maxv:.6f} mm")

    if args.onnx:
        onnx_path = os.path.join(args.outdir, "mesh_oracle.onnx")
        example = preds[1]
        # The dynamo (torch.export) path handles the scatter/gather mesh ops that
        # the legacy TorchScript exporter emits as invalid graphs. Requires onnxscript.
        torch.onnx.export(
            mod, (example,), onnx_path,
            input_names=["pred"], output_names=["verts", "joints", "keypoints"],
            opset_version=18, dynamo=True)
        print(f"wrote {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")
        _check_onnx(onnx_path, preds, ref)


def _check_onnx(path, preds, ref):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[info] onnxruntime not installed; skipping ONNX numeric check")
        return
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    mx = 0.0
    for p in preds:
        vo = ref.forward(p)["verts"][0].numpy()
        vv = sess.run(None, {"pred": p.numpy().astype(np.float32)})[0][0]
        mx = max(mx, np.abs(vv - vo).max() * 1000)
    print(f"ONNX oracle vs TorchScript: max vert err = {mx:.6f} mm")


if __name__ == "__main__":
    main()
