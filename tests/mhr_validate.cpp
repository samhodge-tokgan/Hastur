// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// mhr_validate.cpp -- feeds identical pred[519] vectors to the C++ MhrModel and
// compares the posed mesh to the ONNX/TorchScript oracle fixtures (fixtures.bin),
// reporting per-vertex RMSE (mm) and max error. Target: RMSE well under 1 mm.
//
//   Usage: mhr_validate <mhr_assets.bin> <fixtures.bin>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "../src/MeshAssets.h"
#include "../src/MhrModel.h"

using namespace hastur;

namespace {
struct Stats {
  double rmse_mm, max_mm;
};

Stats compare(const std::vector<float>& a, const float* b, int n) {
  double se = 0.0, mx = 0.0;
  for (int i = 0; i < n * 3; ++i) {
    double d = (static_cast<double>(a[i]) - b[i]) * 1000.0;  // m -> mm
    se += d * d;
    mx = std::max(mx, std::abs(d));
  }
  return {std::sqrt(se / n), mx};  // RMSE over points (3 coords summed)
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <mhr_assets.bin> <fixtures.bin>\n", argv[0]);
    return 2;
  }
  auto assets = MeshAssets::Load(argv[1]);
  auto fixtures = MeshAssets::Load(argv[2]);
  MhrModel model(assets);

  const int nfix = static_cast<int>(fixtures->i32("nfix")[0]);
  std::printf("Loaded %d fixtures. Validating C++ MhrModel vs oracle:\n\n", nfix);
  std::printf("%-8s %14s %14s %14s %14s\n", "fixture", "verts_rmse", "verts_max",
              "joints_max", "kpts_max");

  double worst_rmse = 0.0, worst_max = 0.0;
  for (int i = 0; i < nfix; ++i) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "pred_%02d", i);
    const float* pred_p = fixtures->f32(nm);
    std::array<float, kParamDim> pred{};
    for (int k = 0; k < kParamDim; ++k) pred[k] = pred_p[k];

    std::snprintf(nm, sizeof(nm), "pose_corrective_%02d", i);
    const float* corr = fixtures->f32(nm);

    // Prove the C++ decode independently: compare C++ joint_parameters[889] to
    // the reference (this is the exact input pose_corrective.onnx consumes).
    std::snprintf(nm, sizeof(nm), "joint_params_%02d", i);
    if (fixtures->has(nm)) {
      auto jp = model.JointParameters(pred);
      const float* jp_ref = fixtures->f32(nm);
      double jpmax = 0.0;
      for (int k = 0; k < 889; ++k)
        jpmax = std::max(jpmax, std::abs(static_cast<double>(jp[k]) - jp_ref[k]));
      if (jpmax > 1e-4)
        std::printf("  [warn] fixture %d joint_params max err = %.6g\n", i, jpmax);
    }

    Mesh m = model.Run(pred, corr);

    std::snprintf(nm, sizeof(nm), "verts_%02d", i);
    Stats sv = compare(m.verts, fixtures->f32(nm), kNumVerts);
    std::snprintf(nm, sizeof(nm), "joints_%02d", i);
    Stats sj = compare(m.joints, fixtures->f32(nm), kNumJoints);
    std::snprintf(nm, sizeof(nm), "keypoints_%02d", i);
    Stats sk = compare(m.keypoints, fixtures->f32(nm), kNumKeypoints);

    std::printf("%-8d %11.5f mm %11.5f mm %11.5f mm %11.5f mm\n", i,
                sv.rmse_mm, sv.max_mm, sj.max_mm, sk.max_mm);
    worst_rmse = std::max(worst_rmse, sv.rmse_mm);
    worst_max = std::max(worst_max, sv.max_mm);
  }

  std::printf("\nWORST verts RMSE = %.5f mm,  WORST verts max = %.5f mm\n",
              worst_rmse, worst_max);
  const bool ok = worst_rmse < 1.0;
  std::printf("%s (target: RMSE < 1 mm)\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
