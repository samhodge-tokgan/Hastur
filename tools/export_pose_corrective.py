#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# export_pose_corrective.py -- isolate the MHR pose-corrective MLP (the only
# learned nonlinearity in the mesh path) into a small standalone ONNX that a
# Hastur ORT session runs.
#
#   joint_parameters[889] -> pose_corrective_offset[18439,3]  (cm, pre-flip)
#
# Graph: reshape(127,7) -> take joints[2:] euler[3:6] -> batch6DFromXYZ (750) ->
#        [-1 on the two diag feats] -> Linear(750->3000, dense from the sparse
#        layer, no bias) -> ReLU -> Linear(3000->55317, no bias) -> reshape.
#
# This block is ~663 MB of the 664 MB TorchScript; shipping it as ONNX (fp32, or
# fp16 with --fp16) is what lets mhr_assets.bin stay ~34 MB.
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mhr_common import MhrReference
from mhr_meshmodule import MhrMeshModule, batch6DFromXYZ


class PoseCorrective(nn.Module):
    def __init__(self, mod: MhrMeshModule):
        super().__init__()
        self.register_buffer("W0", mod.pc_W0.clone())   # (3000,750)
        self.register_buffer("W2", mod.pc_W2.clone())   # (55317,3000)
        self.J = mod.J
        self.V = mod.V

    def forward(self, joint_parameters):
        B = joint_parameters.shape[0]
        j = joint_parameters.reshape(B, self.J, 7)
        eul = j[:, 2:, 3:6]
        feat = batch6DFromXYZ(eul)
        # subtract 1 from the two near-diagonal features (identity -> 0)
        off = torch.zeros_like(feat)
        off[:, :, 0] = 1.0
        off[:, :, 4] = 1.0
        feat = feat - off
        x = feat.flatten(1, 2)
        h = F.relu(x @ self.W0.t())
        o = h @ self.W2.t()
        return o.reshape(B, self.V, 3)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../test-assets/pose_corrective.onnx")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    ref = MhrReference()
    mod = MhrMeshModule(ref).eval()
    pc = PoseCorrective(mod).eval()

    # reference offset via the shipped TorchScript pose_correctives_model
    jp = torch.einsum("dn,bn->bd", mod.parameter_transform,
                      torch.cat([torch.zeros(1, 204), torch.zeros(1, 45)], 1))
    # use a non-trivial joint_parameters for the check
    torch.manual_seed(3)
    jp = torch.randn(2, 889) * 0.1
    ref_out = ref.mhr.pose_correctives_model(jp)      # (2,V,3) TorchScript
    my_out = pc(jp)
    err = (ref_out - my_out).abs().max().item()
    print(f"PoseCorrective vs TorchScript pose_correctives_model: max err = {err:.3e} cm")

    dummy = torch.zeros(1, 889)
    torch.onnx.export(
        pc, (dummy,), args.out,
        input_names=["joint_parameters"], output_names=["pose_corrective"],
        dynamic_axes={"joint_parameters": {0: "B"}, "pose_corrective": {0: "B"}},
        opset_version=17, dynamo=False)
    sz = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({sz:.1f} MB)")

    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16
            m = onnx.load(args.out)
            m16 = float16.convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m16, args.out)
            print(f"converted to fp16 -> {os.path.getsize(args.out)/1e6:.1f} MB")
        except Exception as e:
            print(f"[warn] fp16 conversion skipped: {e}")

    # numeric check against ORT if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        oo = sess.run(None, {"joint_parameters": jp.numpy().astype(np.float32)})[0]
        print(f"ONNX vs TorchScript: max err = {np.abs(oo - ref_out.numpy()).max():.3e} cm")
    except ImportError:
        print("[info] onnxruntime not installed; skipping ORT check")


if __name__ == "__main__":
    main()
