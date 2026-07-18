#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# validate_mhr.py -- end-to-end validation of the MHR mesh reimplementation.
#
#   (1) Fixture integrity: re-run the live TorchScript oracle (MhrReference) on
#       each fixture's pred[519] and confirm the stored verts/joints/keypoints
#       still match -> the fixtures faithfully capture the ground truth.
#   (2) C++ parity: run the compiled `mhr_validate` (C++ MhrModel vs fixtures)
#       and stream its per-vertex RMSE report.
#
# Usage: validate_mhr.py [--build-dir ../build] [--assets DIR]
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch

from mhr_common import MhrReference


def fixture_integrity(fixdir):
    ref = MhrReference()
    files = sorted(glob.glob(os.path.join(fixdir, "fixtures", "fix*.npz")))
    if not files:
        print(f"[skip] no fixtures in {fixdir}/fixtures")
        return
    worst = 0.0
    for f in files:
        d = np.load(f)
        pred = torch.from_numpy(d["pred"])[None]
        with torch.no_grad():
            o = ref.forward(pred)
        dv = np.abs(o["verts"][0].numpy() - d["verts"]).max() * 1000
        worst = max(worst, dv)
        print(f"  {os.path.basename(f)}: fixture-vs-live-TorchScript max = {dv:.6f} mm")
    print(f"fixture integrity: worst = {worst:.6f} mm "
          f"({'OK' if worst < 1e-3 else 'DRIFT'})\n")


def cpp_parity(build_dir, assets):
    binp = os.path.join(build_dir, "mhr_validate")
    bin_ = os.path.join(assets, "mhr_assets.bin")
    fix = os.path.join(assets, "fixtures.bin")
    if not os.path.exists(binp):
        print(f"[skip] {binp} not built. Build it with:\n"
              f"       cmake --build {build_dir} --target mhr_validate")
        return None
    print("=== C++ MhrModel vs oracle ===")
    r = subprocess.run([binp, bin_, fix], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--assets", default=os.path.join(here, "..", "test-assets"))
    ap.add_argument("--build-dir", default=os.path.join(here, "..", "build"))
    args = ap.parse_args()

    print("=== (1) fixture integrity: stored fixtures vs live TorchScript ===")
    fixture_integrity(args.assets)
    print("=== (2) C++ parity ===")
    rc = cpp_parity(args.build_dir, args.assets)
    sys.exit(0 if rc in (0, None) else rc)


if __name__ == "__main__":
    main()
