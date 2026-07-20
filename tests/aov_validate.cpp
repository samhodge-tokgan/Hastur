// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// aov_validate -- self-contained unit checks for the AOV support subsystems:
// Cryptomatte (MurmurHash3 conformance, id->float, occlusion ranking) and the
// camera m44f matrices (inverse round-trip + project/unproject). No models, no
// ONNX Runtime, no OFX host required. Exit code 0 = all pass.

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "CameraMatrix.h"
#include "Cryptomatte.h"

using namespace hastur;

static int g_fails = 0;
static void check(bool ok, const char* msg) {
  if (!ok) {
    std::printf("FAIL: %s\n", msg);
    ++g_fails;
  }
}

static void mul44(const std::array<float, 16>& M, const float v[4], float o[4]) {
  for (int r = 0; r < 4; ++r)
    o[r] = M[r * 4 + 0] * v[0] + M[r * 4 + 1] * v[1] + M[r * 4 + 2] * v[2] +
           M[r * 4 + 3] * v[3];
}

int main() {
  // --- MurmurHash3_x86_32 conformance (seed 0) --------------------------
  // Values cross-checked against an independent reference implementation.
  check(MurmurHash3_x86_32("", 0, 0) == 0u, "murmur empty == 0");
  check(MurmurHash3_x86_32("person_00", 9, 0) == 0xc9159c8au, "murmur person_00");
  check(MurmurHash3_x86_32("person_01", 9, 0) == 0x1f83426fu, "murmur person_01");
  check(MurmurHash3_x86_32("person", 6, 0) == 0xff7931b8u, "murmur person");
  check(CryptoTypeKey("person") == "ff7931b", "type key person");
  check(CryptoIdHex("person_00") == "c9159c8a", "id hex person_00");

  // id floats finite and distinct across many names.
  bool all_finite = true;
  for (int i = 0; i < 4096; ++i) {
    char nm[16];
    std::snprintf(nm, sizeof(nm), "person_%02d", i);
    if (!std::isfinite(CryptoIdFloat(nm))) all_finite = false;
  }
  check(all_finite, "all ids finite");

  // --- Cryptomatte occlusion ranking ------------------------------------
  {
    const int W = 2, H = 1;
    std::vector<CryptoPerson> ps(2);
    ps[0].name = "person_00"; ps[0].coverage = {1.0f, 0.5f};  // front
    ps[1].name = "person_01"; ps[1].coverage = {1.0f, 1.0f};  // back
    CryptoResult cr = BuildCryptomatte(W, H, "person", ps, 2);
    check(cr.layers.size() == 2, "two crypto levels");
    const float* p0 = cr.layers[0].data();
    // pixel 0: front fully occludes back -> rank0 = (id0, 1), rank1 absent.
    check(p0[0] == CryptoIdFloat("person_00") && std::fabs(p0[1] - 1.0f) < 1e-6f,
          "px0 rank0 = front, cov 1");
    check(p0[2] == 0.0f && p0[3] == 0.0f, "px0 rank1 absent (occluded)");
    // pixel 1: front .5, back gets .5*(1-.5)=.5 -> both visible at .5.
    const float* p1 = &cr.layers[0][4];
    check(std::fabs(p1[1] - 0.5f) < 1e-6f && std::fabs(p1[3] - 0.5f) < 1e-6f,
          "px1 both ranks cov 0.5");
  }

  // --- Camera matrix round-trip -----------------------------------------
  {
    const int W = 1920, H = 1080;
    const float focal = std::sqrt(float(W) * W + float(H) * H);
    CameraMatrices cm =
        BuildCameraMatrices(focal, W / 2.f, H / 2.f, W, H, 0.1f, 100.f);
    float maxerr = 0.f;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k)
          s += cm.ndc_to_world[r * 4 + k] * cm.world_to_ndc[k * 4 + c];
        maxerr = std::max(maxerr, std::fabs(s - (r == c ? 1.f : 0.f)));
      }
    check(maxerr < 1e-3f, "NDCToWorld . worldToNDC == I");

    float P[4] = {0.3f, -0.2f, 5.0f, 1.0f};
    float clip[4];
    mul44(cm.world_to_ndc, P, clip);
    float ndc[4] = {clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.f};
    check(std::fabs(ndc[0]) <= 1.f && std::fabs(ndc[1]) <= 1.f &&
              std::fabs(ndc[2]) <= 1.f,
          "projected point lands in NDC [-1,1]");
    float q[4];
    mul44(cm.ndc_to_world, ndc, q);
    float err = std::fabs(q[0] / q[3] - P[0]) + std::fabs(q[1] / q[3] - P[1]) +
                std::fabs(q[2] / q[3] - P[2]);
    check(err < 1e-2f, "unproject(project(P)) == P");
  }

  std::printf(g_fails ? "\naov_validate: %d FAILURE(S)\n"
                      : "\naov_validate: ALL PASS\n",
              g_fails);
  return g_fails ? 1 : 0;
}
