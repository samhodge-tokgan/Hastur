// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// pipeline_smoke — offline end-to-end exercise of Sam3dBodyPipeline (no OFX host).
//
//   pipeline_smoke <input_image> <output.png> [model_search_path]
//
// Loads a real photo (stb_image), runs the BODY-ONLY pipeline directly with
// HASTUR_MODEL_DIR (or the 3rd arg) pointing at the models/assets, writes the
// composited grey-mesh RGBA to a PNG, and prints sanity stats (alpha coverage,
// silhouette bbox). Set HASTUR_PIPELINE_TIMING=1 for per-stage timing on stderr.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include "Sam3dBodyPipeline.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <input_image> <output.png> [model_search_path]\n",
                 argv[0]);
    return 2;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];
  const std::string model_dir = argc > 3 ? argv[3] : "";

  int W = 0, H = 0, ncomp = 0;
  unsigned char* px = stbi_load(in_path.c_str(), &W, &H, &ncomp, 3);  // force RGB
  if (!px) {
    std::fprintf(stderr, "failed to load image: %s\n", in_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "loaded %s : %dx%d (%d ch)\n", in_path.c_str(), W, H, ncomp);

  // Optional downsize so the 1.8 GB body ONNX stays quick on CPU. The crop is
  // always 512² regardless, so this only affects detection/warp resolution.
  int maxdim = 1280;
  if (const char* m = std::getenv("HASTUR_SMOKE_MAXDIM")) maxdim = std::atoi(m);

  std::vector<float> rgb;
  int fw = W, fh = H;
  if (maxdim > 0 && (W > maxdim || H > maxdim)) {
    const float sc = static_cast<float>(maxdim) / (W > H ? W : H);
    fw = static_cast<int>(W * sc);
    fh = static_cast<int>(H * sc);
    rgb.assign(static_cast<size_t>(fw) * fh * 3, 0.f);
    for (int y = 0; y < fh; ++y) {
      const int sy = static_cast<int>(y / sc);
      for (int x = 0; x < fw; ++x) {
        const int sx = static_cast<int>(x / sc);
        const unsigned char* s = px + (static_cast<size_t>(sy) * W + sx) * 3;
        float* d = &rgb[(static_cast<size_t>(y) * fw + x) * 3];
        d[0] = s[0] / 255.f; d[1] = s[1] / 255.f; d[2] = s[2] / 255.f;
      }
    }
    std::fprintf(stderr, "downsized to %dx%d (maxdim=%d)\n", fw, fh, maxdim);
  } else {
    rgb.assign(static_cast<size_t>(W) * H * 3, 0.f);
    for (size_t i = 0, n = static_cast<size_t>(W) * H * 3; i < n; ++i)
      rgb[i] = px[i] / 255.f;
  }
  stbi_image_free(px);

  hastur::PipelineParams p;
  p.model_dir = model_dir;
  p.units = hastur::ComputeUnits::All;
  p.detector_score_thresh = 0.4f;
  p.max_people = 1;
  p.grey = 0.6f;
  p.ssaa = 2;
  p.premultiply = false;

  hastur::Sam3dBodyPipeline pipe;
  hastur::FrameResult fr = pipe.Run(rgb.data(), fw, fh, p);

  if (static_cast<int>(fr.render.data.size()) != fw * fh * 4) {
    std::fprintf(stderr, "pipeline produced no render (%s)\n",
                 pipe.last_error().c_str());
    return 1;
  }
  std::fprintf(stderr, "people detected/meshed: %zu   last_error: '%s'\n",
               fr.people.size(), pipe.last_error().c_str());
  for (size_t i = 0; i < fr.people.size(); ++i) {
    const auto& pr = fr.people[i];
    std::fprintf(stderr,
                 "  person %zu: box=[%.0f,%.0f,%.0f,%.0f] score=%.3f "
                 "verts=%zu cam_t=[%.3f,%.3f,%.3f] focal=%.1f has_hands=%d\n",
                 i, pr.box.x0, pr.box.y0, pr.box.x1, pr.box.y1, pr.box.score,
                 pr.mesh.verts.size() / 3, pr.cam.cam_t[0], pr.cam.cam_t[1],
                 pr.cam.cam_t[2], pr.cam.focal, pr.has_hands ? 1 : 0);
  }

  // Stats: alpha coverage + silhouette bbox (alpha > 0.5).
  const std::vector<float>& R = fr.render.data;
  long covered = 0;
  int bx0 = fw, by0 = fh, bx1 = -1, by1 = -1;
  for (int y = 0; y < fh; ++y) {
    for (int x = 0; x < fw; ++x) {
      const float a = R[(static_cast<size_t>(y) * fw + x) * 4 + 3];
      if (a > 0.5f) {
        ++covered;
        if (x < bx0) bx0 = x;
        if (y < by0) by0 = y;
        if (x > bx1) bx1 = x;
        if (y > by1) by1 = y;
      }
    }
  }
  const double cov_pct = 100.0 * covered / (static_cast<double>(fw) * fh);

  // Write PNG (straight RGBA -> 8-bit). Render is top-down, HWC.
  std::vector<unsigned char> png(static_cast<size_t>(fw) * fh * 4);
  for (size_t i = 0, n = static_cast<size_t>(fw) * fh * 4; i < n; ++i) {
    float v = R[i];
    if (v < 0.f) v = 0.f;
    if (v > 1.f) v = 1.f;
    png[i] = static_cast<unsigned char>(v * 255.f + 0.5f);
  }
  if (!stbi_write_png(out_path.c_str(), fw, fh, 4, png.data(), fw * 4)) {
    std::fprintf(stderr, "failed to write PNG: %s\n", out_path.c_str());
    return 1;
  }

  std::printf("SMOKE OK\n");
  std::printf("  output      : %s (%dx%d)\n", out_path.c_str(), fw, fh);
  std::printf("  alpha cover : %.3f%% (%ld px > 0.5)\n", cov_pct, covered);
  if (bx1 >= 0)
    std::printf("  silhouette  : bbox [%d,%d,%d,%d]  (%dx%d)\n", bx0, by0, bx1, by1,
                bx1 - bx0 + 1, by1 - by0 + 1);
  else
    std::printf("  silhouette  : EMPTY\n");
  return 0;
}
