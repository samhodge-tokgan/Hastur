// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone numerical validation harness for Track F geometry. Reads a cases
// file produced by tools/validate_geometry.py, computes every ported quantity
// via CameraSolver / CropAffine, and writes results for the Python side to diff
// against the sam-3d-body reference. Built directly with clang++ (see the
// header of tools/validate_geometry.py); it does NOT touch CMake.
//
// Cases file: one case per line, whitespace separated:
//   W H x0 y0 x1 y1 pad camflag f cx cy p0 p1 p2 defscale
// camflag==1 -> use (f,cx,cy) as cam_int; camflag==0 -> DefaultCamInt(W,H).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "CameraSolver.h"
#include "CropAffine.h"

using namespace hastur;

// Deterministic synthetic image shared with the Python reference so the crop
// warp can be validated without transferring pixels.
static std::vector<float> SynthImage(int W, int H) {
  std::vector<float> img(static_cast<size_t>(W) * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      float* p = img.data() + (static_cast<size_t>(y) * W + x) * 3;
      // Smooth low-frequency gradient so float32 affine LSB differences in the
      // sampler do not get amplified by high-frequency content.
      const float fx = W > 1 ? x / (W - 1.f) : 0.f;
      const float fy = H > 1 ? y / (H - 1.f) : 0.f;
      p[0] = fx;
      p[1] = fy;
      p[2] = 0.5f * (fx + fy);
    }
  return img;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <cases.txt> <out.txt>\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  out.setf(std::ios::scientific);
  out.precision(9);

  std::string line;
  int idx = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double W, H, x0, y0, x1, y1, pad, camflag, f, cx, cy, p0, p1, p2, defscale;
    if (!(ss >> W >> H >> x0 >> y0 >> x1 >> y1 >> pad >> camflag >> f >> cx >> cy >>
          p0 >> p1 >> p2 >> defscale))
      continue;

    BBox box{static_cast<float>(x0), static_cast<float>(y0), static_cast<float>(x1),
             static_cast<float>(y1), 1.f};
    CamInt K = camflag > 0.5
                   ? CamInt{static_cast<float>(f), 0, static_cast<float>(cx),
                            0, static_cast<float>(f), static_cast<float>(cy), 0, 0, 1}
                   : DefaultCamInt(static_cast<int>(W), static_cast<int>(H));

    BBoxCS cs = ProcessBBox(box, static_cast<float>(pad));
    std::array<float, 6> fwd, inv;
    MakeAffine(cs.center, cs.square, kBodyImageSize, fwd, inv);
    auto cond = ConditionInfo(box, static_cast<float>(pad), K);
    std::array<float, 2 * kTokenGrid * kTokenGrid> ray{};
    RayCond(box, static_cast<float>(pad), K, ray);
    Camera cam = PerspectiveProjection(
        {static_cast<float>(p0), static_cast<float>(p1), static_cast<float>(p2)}, box,
        static_cast<float>(pad), K, static_cast<float>(defscale));

    // Crop warp: build the synthetic image, warp, emit a 16x16x3 sample grid.
    std::vector<float> img = SynthImage(static_cast<int>(W), static_cast<int>(H));
    CropInputs crop =
        MakeCrop(img.data(), static_cast<int>(W), static_cast<int>(H), box, K,
                 static_cast<float>(pad));

    out << "CASE " << idx << "\n";
    out << "CENTER " << cs.center[0] << " " << cs.center[1] << "\n";
    out << "SCALE " << cs.scale[0] << " " << cs.scale[1] << "\n";
    out << "SQUARE " << cs.square << "\n";
    out << "CAMINT " << K[0] << " " << K[2] << " " << K[5] << "\n";
    out << "AFF " << fwd[0] << " " << fwd[1] << " " << fwd[2] << " " << fwd[3] << " "
        << fwd[4] << " " << fwd[5] << "\n";
    out << "AFFINV " << inv[0] << " " << inv[1] << " " << inv[2] << " " << inv[3] << " "
        << inv[4] << " " << inv[5] << "\n";
    out << "COND " << cond[0] << " " << cond[1] << " " << cond[2] << "\n";
    out << "PERSP " << cam.focal << " " << cam.cam_t[0] << " " << cam.cam_t[1] << " "
        << cam.cam_t[2] << "\n";
    out << "RAY";
    for (float v : ray) out << " " << v;
    out << "\n";
    // 16x16 grid of crop coords over [0,512), 3 channels (CHW), from crop.image.
    const int S = kBodyImageSize;
    const size_t plane = static_cast<size_t>(S) * S;
    out << "CROP";
    for (int gy = 0; gy < 16; ++gy)
      for (int gx = 0; gx < 16; ++gx) {
        int v = gy * (S / 16) + (S / 32);
        int u = gx * (S / 16) + (S / 32);
        size_t pix = static_cast<size_t>(v) * S + u;
        for (int ch = 0; ch < 3; ++ch)
          out << " " << crop.image[ch * plane + pix];
      }
    out << "\n";
    ++idx;
  }
  std::fprintf(stderr, "geometry_validate: processed %d cases\n", idx);
  return 0;
}
