#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# validate_e2e.py -- TRUE end-to-end validation of the shipping pipeline:
#
#   pred[519] --C++ decode--> joint_parameters[889] --pose_corrective.onnx (ORT)-->
#   corrective offsets --C++ LBS (mhr_assets.bin)--> posed mesh   vs   oracle.
#
# The pose-corrective offsets here come from the REAL ONNX session (not torch),
# so this exercises exactly what Hastur ships: mhr_assets.bin + pose_corrective.onnx
# driving the C++ MhrModel. Ground truth = the TorchScript oracle stored in the
# fixtures. Also writes joint_params so the C++ test cross-checks its own decode.
import argparse
import glob
import os
import subprocess

import numpy as np
import onnxruntime as ort
import torch

from mhr_binfmt import write_assets
from mhr_common import MhrReference
from mhr_meshmodule import MhrMeshModule


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(here, "..", "test-assets"))
    ap.add_argument("--build-dir", default=os.path.join(here, "..", "build"))
    args = ap.parse_args()

    ref = MhrReference()
    mod = MhrMeshModule(ref).eval()
    pc_path = os.path.join(args.assets, "pose_corrective.onnx")
    sess = ort.InferenceSession(pc_path, providers=["CPUExecutionProvider"])
    print(f"loaded {os.path.basename(pc_path)} "
          f"({os.path.getsize(pc_path)/1e6:.0f} MB) via ONNX Runtime {ort.__version__}")

    files = sorted(glob.glob(os.path.join(args.assets, "fixtures", "fix*.npz")))
    blocks = {"nfix": np.asarray([len(files)], dtype=np.int32)}
    worst_corr = 0.0
    for i, f in enumerate(files):
        d = np.load(f)
        pred = torch.from_numpy(d["pred"])[None]
        # joint_parameters[889] (same math the C++ MhrModel::JointParameters runs)
        model_params, shape = mod.decode(pred)
        full249 = torch.cat([model_params, torch.zeros(1, 45)], dim=1)
        jp = torch.einsum("dn,bn->bd", mod.parameter_transform, full249)  # (1,889)
        # pose-corrective offsets from the REAL ONNX session
        corr_ort = sess.run(None, {"joint_parameters": jp.numpy().astype(np.float32)})[0][0]
        worst_corr = max(worst_corr, np.abs(corr_ort - d["pose_corrective"]).max())
        blocks[f"pred_{i:02d}"] = d["pred"].astype(np.float32)
        blocks[f"verts_{i:02d}"] = d["verts"].astype(np.float32)
        blocks[f"joints_{i:02d}"] = d["joints"].astype(np.float32)
        blocks[f"keypoints_{i:02d}"] = d["keypoints"].astype(np.float32)
        blocks[f"pose_corrective_{i:02d}"] = corr_ort.astype(np.float32)  # ORT, not torch
        blocks[f"joint_params_{i:02d}"] = jp[0].numpy().astype(np.float32)

    print(f"pose_corrective.onnx (ORT) vs torch corrective: max = {worst_corr:.3e} cm")
    e2e = os.path.join(args.assets, "fixtures_e2e.bin")
    write_assets(e2e, blocks)

    binp = os.path.join(args.build_dir, "mhr_validate")
    if not os.path.exists(binp):
        print(f"[skip] build the C++ test first: cmake --build {args.build_dir} --target mhr_validate")
        return
    print("\n=== C++ MhrModel (mhr_assets.bin) + pose_corrective.onnx (ORT) vs oracle ===")
    r = subprocess.run([binp, os.path.join(args.assets, "mhr_assets.bin"), e2e],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)


if __name__ == "__main__":
    main()
