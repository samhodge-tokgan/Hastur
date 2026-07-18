// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// parity_dump -- M5 numeric-parity harness. Runs the REAL Sam3dBodyPipeline on an
// input image and dumps, per detected person, the full inter-stage contract
// (pred[519], pred_cam[3], solved camera, verts[18439,3], keypoints[70,3]) plus
// the composited silhouette alpha, to a compact binary blob. It also re-emits the
// EXACT 8-bit RGB image it fed the pipeline as a lossless PNG, so the Python
// reference (tools/validate_e2e_parity.py) compares against byte-identical pixels
// (no JPEG-decoder mismatch between stb and cv2).
//
//   parity_dump <input_image> <out_prefix> [model_search_path]
//
// Writes:  <out_prefix>.png   -- the exact RGB frame the pipeline consumed
//          <out_prefix>.bin   -- HPDUMP2 binary (see layout below)
//
// This is a STANDALONE build (mirrors tools/pipeline_smoke.cpp): it compiles the
// engine .cpp files directly and links the private ORT dylib; it does NOT modify
// any pipeline source. Body-only parity is obtained by pointing the model search
// path at a directory WITHOUT sam3dbody_hand.onnx (e.g. build/smoke_models_nohand),
// which disables the M7 hand refiner so the dump matches the reference body path.
//
// Binary layout (little-endian, host float32/int32):
//   char    magic[8] = "HPDUMP2\0"
//   int32   W, H
//   int32   n_people
//   repeat n_people:
//     float box[5]           (x0,y0,x1,y1,score)   full-frame px
//     float pred[519]
//     float pred_cam[3]
//     float cam[6]           (focal, cx, cy, tx, ty, tz)
//     int32 n_verts          (== 18439)
//     float verts[n_verts*3] (m, MHR camera frame, [1,2] flipped)
//     int32 n_kpts           (== 70)
//     float kpts[n_kpts*3]   (m, same frame)
//   float alpha[W*H]         (composited silhouette coverage, straight)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include "Sam3dBodyPipeline.h"

namespace {

void WriteF32(std::FILE* f, const float* p, size_t n) {
  std::fwrite(p, sizeof(float), n, f);
}
void WriteI32(std::FILE* f, int32_t v) { std::fwrite(&v, sizeof(int32_t), 1, f); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <input_image> <out_prefix> [model_search_path]\n",
                 argv[0]);
    return 2;
  }
  const std::string in_path = argv[1];
  const std::string prefix = argv[2];
  const std::string model_dir = argc > 3 ? argv[3] : "";

  int W = 0, H = 0, ncomp = 0;
  unsigned char* px = stbi_load(in_path.c_str(), &W, &H, &ncomp, 3);  // force RGB
  if (!px) {
    std::fprintf(stderr, "failed to load image: %s\n", in_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "loaded %s : %dx%d (%d ch)\n", in_path.c_str(), W, H, ncomp);

  // Full resolution, no downsize: the reference loads the re-emitted PNG so both
  // see byte-identical pixels. (The body ONNX cost is crop-bound, not frame-bound.)
  std::vector<float> rgb(static_cast<size_t>(W) * H * 3);
  for (size_t i = 0, n = rgb.size(); i < n; ++i) rgb[i] = px[i] / 255.f;

  // Re-emit the EXACT 8-bit RGB frame as a lossless PNG for the reference.
  const std::string png_path = prefix + ".png";
  if (!stbi_write_png(png_path.c_str(), W, H, 3, px, W * 3)) {
    std::fprintf(stderr, "failed to write %s\n", png_path.c_str());
    stbi_image_free(px);
    return 1;
  }
  stbi_image_free(px);

  hastur::PipelineParams p;
  p.model_dir = model_dir;
  p.units = hastur::ComputeUnits::All;
  p.detector_score_thresh = 0.4f;
  p.max_people = 1;
  p.grey = 0.6f;
  p.ssaa = 2;
  p.premultiply = false;  // straight alpha == pure coverage silhouette

  hastur::Sam3dBodyPipeline pipe;
  hastur::FrameResult fr = pipe.Run(rgb.data(), W, H, p);

  std::fprintf(stderr, "people detected/meshed: %zu   last_error: '%s'\n",
               fr.people.size(), pipe.last_error().c_str());
  for (size_t i = 0; i < fr.people.size(); ++i) {
    const auto& pr = fr.people[i];
    std::fprintf(stderr,
                 "  person %zu: box=[%.1f,%.1f,%.1f,%.1f] score=%.3f "
                 "verts=%zu cam_t=[%.4f,%.4f,%.4f] focal=%.2f\n",
                 i, pr.box.x0, pr.box.y0, pr.box.x1, pr.box.y1, pr.box.score,
                 pr.mesh.verts.size() / 3, pr.cam.cam_t[0], pr.cam.cam_t[1],
                 pr.cam.cam_t[2], pr.cam.focal);
  }

  const std::string bin_path = prefix + ".bin";
  std::FILE* f = std::fopen(bin_path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "failed to open %s\n", bin_path.c_str());
    return 1;
  }
  char magic[8] = {'H', 'P', 'D', 'U', 'M', 'P', '2', '\0'};
  std::fwrite(magic, 1, 8, f);
  WriteI32(f, W);
  WriteI32(f, H);
  WriteI32(f, static_cast<int32_t>(fr.people.size()));

  for (const auto& pr : fr.people) {
    float box[5] = {pr.box.x0, pr.box.y0, pr.box.x1, pr.box.y1, pr.box.score};
    WriteF32(f, box, 5);
    WriteF32(f, pr.pred.pred.data(), pr.pred.pred.size());       // 519
    WriteF32(f, pr.pred.pred_cam.data(), pr.pred.pred_cam.size());  // 3
    float cam[6] = {pr.cam.focal, pr.cam.center[0], pr.cam.center[1],
                    pr.cam.cam_t[0], pr.cam.cam_t[1], pr.cam.cam_t[2]};
    WriteF32(f, cam, 6);
    WriteI32(f, static_cast<int32_t>(pr.mesh.verts.size() / 3));
    WriteF32(f, pr.mesh.verts.data(), pr.mesh.verts.size());
    WriteI32(f, static_cast<int32_t>(pr.mesh.keypoints.size() / 3));
    WriteF32(f, pr.mesh.keypoints.data(), pr.mesh.keypoints.size());
  }

  // Silhouette: the composited coverage alpha (straight), W*H.
  std::vector<float> alpha(static_cast<size_t>(W) * H, 0.f);
  const std::vector<float>& R = fr.render.data;
  if (static_cast<int>(R.size()) == W * H * 4) {
    for (size_t i = 0; i < alpha.size(); ++i) alpha[i] = R[i * 4 + 3];
  }
  WriteF32(f, alpha.data(), alpha.size());
  std::fclose(f);

  std::fprintf(stderr, "wrote %s and %s\n", bin_path.c_str(), png_path.c_str());
  return 0;
}
