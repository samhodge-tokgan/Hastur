// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// leotard_preview -- renders the garment (leotard) mask on the canonical rest
// pose with NO inference (loads mhr_assets.bin, evaluates the identity MHR mesh,
// computes the mask, rasterizes it). Sub-second, so it is the fast iteration /
// regression tool for tuning the mask coverage. Writes an RGB PNG over white.
//
//   leotard_preview <mhr_assets.bin | model_dir> <out.png>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "LeotardMask.h"
#include "MeshAssets.h"
#include "MhrModel.h"
#include "SoftwareRasterizer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

using namespace hastur;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <mhr_assets.bin|dir> <out.png>\n", argv[0]);
    return 2;
  }
  std::string apath = argv[1];
  if (apath.find("mhr_assets.bin") == std::string::npos) apath += "/mhr_assets.bin";

  std::shared_ptr<MeshAssets> assets;
  try {
    assets = MeshAssets::Load(apath);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "load %s failed: %s\n", apath.c_str(), e.what());
    return 1;
  }
  MhrModel mhr(assets);
  std::array<float, kParamDim> zero{};
  Mesh rest = mhr.Run(zero, nullptr);
  std::vector<float> mask = ComputeLeotardMask(rest);
  if (mask.empty()) {
    std::fprintf(stderr, "ComputeLeotardMask returned empty (degenerate)\n");
    return 1;
  }
  double cover = 0.0;
  for (float m : mask) cover += m;
  std::fprintf(stderr, "leotard coverage: %.1f%% of %d verts\n",
               100.0 * cover / mask.size(), kNumVerts);

  // Frame the rest mesh frontally: centre it and set focal to fill ~85%.
  const int W = 640, H = 800;
  float lo[3] = {1e9f, 1e9f, 1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
  for (int v = 0; v < kNumVerts; ++v)
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], rest.verts[v * 3 + k]);
      hi[k] = std::max(hi[k], rest.verts[v * 3 + k]);
    }
  const float cen[3] = {(lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f,
                        (lo[2] + hi[2]) * 0.5f};
  const float size = std::max(hi[1] - lo[1], hi[0] - lo[0]);
  const float D = 3.0f;
  Camera cam;
  cam.focal = 0.85f * H * D / std::max(1e-3f, size);
  cam.center = {W * 0.5f, H * 0.5f};
  cam.cam_t = {-cen[0], -cen[1], D - cen[2]};

  RasterOptions opt;
  opt.ssaa = 2;
  opt.premultiply = false;
  opt.garment = true;
  opt.leotard_rgb[0] = 0.08f; opt.leotard_rgb[1] = 0.15f; opt.leotard_rgb[2] = 0.6f;
  opt.skin_rgb[0] = opt.skin_rgb[1] = opt.skin_rgb[2] = 0.6f;
  VertAttrib attrib;
  attrib.leotardness = mask.data();

  RgbaImage img = Render(rest, cam, W, H, opt, nullptr, &attrib);

  // The rest mesh is Y-flipped in the camera frame (head projects to the top of
  // image space is DOWN), so flip vertically on write to get an upright figure.
  std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t src = (static_cast<size_t>(H - 1 - y) * W + x) * 4;
      const size_t dst = (static_cast<size_t>(y) * W + x) * 3;
      const float a = img.data[src + 3];
      for (int k = 0; k < 3; ++k) {
        float c = img.data[src + k] * a + 1.0f * (1.0f - a);  // over white
        c = c < 0 ? 0 : (c > 1 ? 1 : c);
        rgb[dst + k] = static_cast<uint8_t>(c * 255.0f + 0.5f);
      }
    }
  }
  if (!stbi_write_png(argv[2], W, H, 3, rgb.data(), W * 3)) {
    std::fprintf(stderr, "write %s failed\n", argv[2]);
    return 1;
  }
  std::fprintf(stderr, "wrote %s (%dx%d)\n", argv[2], W, H);
  return 0;
}
