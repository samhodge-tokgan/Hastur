// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// raster_validate.cpp -- loads the raster fixtures (test-assets/raster_fixtures.bin,
// produced by tools/dump_test_meshes.py), runs the C++ SoftwareRasterizer, and
// compares against the reference RGBA:
//   * alpha IoU     -- coverage-mask agreement (silhouette), the geometrically
//                      meaningful check any correct rasterizer must pass.
//   * shaded-L1     -- mean abs RGB diff over pixels covered in BOTH renders.
// It also reports the anti-aliased (ss=2) silhouette IoU and checks the
// premultiplied-alpha invariant (premult.rgb == straight.rgb * alpha).
//
//   Usage: raster_validate <raster_fixtures.bin>
//
// Build standalone (no ORT / OFX needed):
//   clang++ -std=c++17 -O2 -I src tests/raster_validate.cpp \
//           src/SoftwareRasterizer.cpp src/MeshAssets.cpp -o build/raster_validate

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../src/MeshAssets.h"
#include "../src/MeshTypes.h"
#include "../src/SoftwareRasterizer.h"

using namespace hastur;

namespace {

struct Cmp {
  double iou;      // alpha IoU (mask@0.5)
  double l1;       // mean |rgb diff| over both-covered pixels
  long both;       // #pixels covered in both
};

Cmp CompareToRef(const RgbaImage& img, const float* ref, int W, int H) {
  long inter = 0, uni = 0, both = 0;
  double l1 = 0.0;
  for (int i = 0; i < W * H; ++i) {
    const float* c = &img.data[i * 4];
    const float* r = &ref[i * 4];
    const bool ca = c[3] > 0.5f, ra = r[3] > 0.5f;
    if (ca || ra) ++uni;
    if (ca && ra) {
      ++inter;
      ++both;
      l1 += std::fabs(c[0] - r[0]) + std::fabs(c[1] - r[1]) + std::fabs(c[2] - r[2]);
    }
  }
  return {uni ? static_cast<double>(inter) / uni : 1.0,
          both ? l1 / (both * 3.0) : 0.0, both};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <raster_fixtures.bin>\n", argv[0]);
    return 2;
  }
  auto fx = MeshAssets::Load(argv[1]);
  auto faces = fx->faces();
  if (!faces) {
    std::fprintf(stderr, "no faces block in fixtures\n");
    return 2;
  }
  const int nfix = static_cast<int>(fx->i32("nfix")[0]);
  std::printf("Loaded %d raster fixtures (faces=%zu tris)\n\n",
              nfix, faces->size() / 3);

  std::printf("%-6s %-10s %8s %8s %10s %12s %10s\n", "fix", "res", "ss1_IoU",
              "ss2_IoU", "shadedL1", "both_px", "premult");

  double worst_iou = 1.0, worst_l1 = 0.0;
  bool premult_ok = true;
  for (int i = 0; i < nfix; ++i) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "cam_%02d", i);
    const float* cam = fx->f32(nm);
    const float focal = cam[0];
    const int W = static_cast<int>(std::lround(cam[6]));
    const int H = static_cast<int>(std::lround(cam[7]));

    std::snprintf(nm, sizeof(nm), "verts_%02d", i);
    const MeshAssets::Block& vb = fx->block(nm);
    const float* vp = fx->f32(nm);
    const int nv = static_cast<int>(vb.numel() / 3);

    std::snprintf(nm, sizeof(nm), "rgba_%02d", i);
    const float* ref = fx->f32(nm);

    Mesh mesh;
    mesh.verts.assign(vp, vp + nv * 3);
    mesh.faces = faces;

    Camera c;
    c.focal = focal;
    c.center = {cam[1], cam[2]};
    c.cam_t = {cam[3], cam[4], cam[5]};

    RasterOptions o1;
    o1.ssaa = 1;
    o1.premultiply = false;
    RgbaImage img1 = Render(mesh, c, W, H, o1);
    Cmp c1 = CompareToRef(img1, ref, W, H);

    RasterOptions o2;
    o2.ssaa = 2;
    o2.premultiply = false;
    RgbaImage img2 = Render(mesh, c, W, H, o2);
    Cmp c2 = CompareToRef(img2, ref, W, H);

    // premultiplied invariant: premult.rgb ~= straight.rgb * alpha (same ss=1).
    RasterOptions op;
    op.ssaa = 1;
    op.premultiply = true;
    RgbaImage imgp = Render(mesh, c, W, H, op);
    double pmax = 0.0;
    for (int p = 0; p < W * H; ++p) {
      const float a = img1.data[p * 4 + 3];
      for (int k = 0; k < 3; ++k) {
        double expect = img1.data[p * 4 + k] * a;
        pmax = std::max(pmax, std::fabs(expect - imgp.data[p * 4 + k]));
      }
    }
    if (pmax > 1e-5) premult_ok = false;

    char res[16];
    std::snprintf(res, sizeof(res), "%dx%d", W, H);
    std::printf("%-6d %-10s %8.5f %8.5f %10.6f %12ld %10.2g\n", i, res, c1.iou,
                c2.iou, c1.l1, c1.both, pmax);
    worst_iou = std::min(worst_iou, c1.iou);
    worst_l1 = std::max(worst_l1, c1.l1);
  }

  std::printf("\nWORST ss1 alpha IoU = %.5f,  WORST shaded-L1 = %.6f\n", worst_iou,
              worst_l1);
  std::printf("premult invariant: %s\n", premult_ok ? "OK" : "FAIL");
  // Thresholds: an exact-semantics port should reproduce the reference nearly
  // bit-for-bit at ss=1 (IoU very close to 1, L1 ~ 0).
  const bool ok = worst_iou > 0.999 && worst_l1 < 1e-3 && premult_ok;
  std::printf("%s (target: ss1 IoU > 0.999, shaded-L1 < 1e-3)\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
