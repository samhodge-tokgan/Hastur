// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// matte_partition_validate — self-contained checks for PartitionCryptomatte: the
// Matte Partition CORE. Builds a synthetic input Cryptomatte from two hard person
// boxes, a pooled soft alpha that is the two boxes DILATED (a soft halo beyond the
// hard masks) plus a stray far blob with NO owner, then partitions and asserts:
//
//   (a) each person's decoded coverage ⊇ its hard mask, and the dilated halo is
//       assigned to the nearest person (external soft edge kept, at its soft value)
//   (b) the two persons stay pixel-exclusive (no pixel owned by both)
//   (c) the stray far blob (beyond max_dist from any seed) is dropped from all ids
//   (d) the output manifest / names round-trip the input
//
// No models / ONNX / OFX. Exit code 0 = all pass.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Cryptomatte.h"
#include "MattePartition.h"

using namespace hastur;

static int g_fails = 0;
static void check(bool ok, const char* msg) {
  if (!ok) {
    std::printf("FAIL: %s\n", msg);
    ++g_fails;
  }
}

// --- test geometry ---------------------------------------------------------
static const int W = 64, H = 48;
// Two disjoint hard boxes [x0,x1) x [y0,y1).
static const int aX0 = 8, aX1 = 20, aY0 = 10, aY1 = 30;   // person_00
static const int bX0 = 40, bX1 = 52, bY0 = 10, bY1 = 30;  // person_01
static const int kDilate = 3;                             // halo radius (chebyshev)
// Stray blob with no owner, far from either box.
static const int sX0 = 58, sX1 = 62, sY0 = 42, sY1 = 46;
static const float kMaxDist = 8.0f;

static const float kPoolHard = 1.0f;   // soft alpha over the hard interior
static const float kPoolHalo = 0.4f;   // soft alpha over the dilated halo
static const float kPoolBlob = 0.7f;   // soft alpha over the stray blob

static bool inBox(int x, int y, int x0, int x1, int y0, int y1) {
  return x >= x0 && x < x1 && y >= y0 && y < y1;
}
static bool inHardA(int x, int y) { return inBox(x, y, aX0, aX1, aY0, aY1); }
static bool inHardB(int x, int y) { return inBox(x, y, bX0, bX1, bY0, bY1); }
// Chebyshev dilation of each box by kDilate == an enlarged rectangle.
static bool inDilA(int x, int y) {
  return inBox(x, y, aX0 - kDilate, aX1 + kDilate, aY0 - kDilate, aY1 + kDilate);
}
static bool inDilB(int x, int y) {
  return inBox(x, y, bX0 - kDilate, bX1 + kDilate, bY0 - kDilate, bY1 + kDilate);
}
static bool inBlob(int x, int y) { return inBox(x, y, sX0, sX1, sY0, sY1); }

// Decode one id's coverage from a CryptoResult: for each pixel, scan every packed
// rank across all layers and take the coverage paired with the matching id (0 if
// absent). Mirrors how a compositor reads a Cryptomatte back.
static std::vector<float> decode(const CryptoResult& c, float id) {
  const size_t npix = static_cast<size_t>(c.width) * c.height;
  std::vector<float> out(npix, 0.0f);
  for (const std::vector<float>& layer : c.layers) {
    if (layer.size() < npix * 4) continue;
    for (size_t px = 0; px < npix; ++px) {
      for (int half = 0; half < 2; ++half) {
        const float lid = layer[px * 4 + half * 2 + 0];
        const float cov = layer[px * 4 + half * 2 + 1];
        if (std::fabs(lid - id) <= 1e-6f * (1.0f + std::fabs(id)) && cov > 0.0f)
          out[px] += cov;
      }
    }
  }
  return out;
}

int main() {
  const size_t npix = static_cast<size_t>(W) * H;

  // --- build the synthetic INPUT Cryptomatte from two hard boxes -----------
  std::vector<CryptoPerson> in_persons(2);
  in_persons[0].name = "person_00";
  in_persons[1].name = "person_01";
  in_persons[0].coverage.assign(npix, 0.0f);
  in_persons[1].coverage.assign(npix, 0.0f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t px = static_cast<size_t>(y) * W + x;
      if (inHardA(x, y)) in_persons[0].coverage[px] = 1.0f;
      if (inHardB(x, y)) in_persons[1].coverage[px] = 1.0f;
    }
  const int kLevels = 2;
  const CryptoResult input =
      BuildCryptomatte(W, H, "person", in_persons, kLevels);

  // ids manifest for the partitioner (name + Cryptomatte id float).
  std::vector<CryptoLabel> ids = {
      {"person_00", CryptoIdFloat("person_00")},
      {"person_01", CryptoIdFloat("person_01")},
  };

  // --- pooled soft alpha: dilated union (soft halo) + stray far blob -------
  std::vector<float> pool(npix, 0.0f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t px = static_cast<size_t>(y) * W + x;
      if (inHardA(x, y) || inHardB(x, y))
        pool[px] = kPoolHard;
      else if (inDilA(x, y) || inDilB(x, y))
        pool[px] = kPoolHalo;  // soft halo beyond the hard masks
      else if (inBlob(x, y))
        pool[px] = kPoolBlob;  // stray, no owner
    }

  // --- partition -----------------------------------------------------------
  const CryptoResult out =
      PartitionCryptomatte(input.layers, ids, pool, W, H, kMaxDist, kLevels,
                           "person");

  check(out.width == W && out.height == H, "output dimensions preserved");

  const std::vector<float> cov0 = decode(out, ids[0].id);
  const std::vector<float> cov1 = decode(out, ids[1].id);
  check(cov0.size() == npix && cov1.size() == npix, "decoded buffers sized W*H");

  // --- (a) hard mask ⊇ + soft halo assigned to nearest; ---------------------
  // --- (b) exclusivity; (c) stray blob dropped -----------------------------
  bool a_ok = true, halo_soft_ok = true, excl_ok = true, blob_dropped = true;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t px = static_cast<size_t>(y) * W + x;
      const float c0 = cov0[px], c1 = cov1[px];

      // (b) never owned by both people.
      if (c0 > 0.0f && c1 > 0.0f) excl_ok = false;

      if (inHardA(x, y)) {
        if (!(c0 > 0.0f)) a_ok = false;                 // ⊇ hard mask A
        if (std::fabs(c0 - kPoolHard) > 1e-6f) a_ok = false;
        if (c1 != 0.0f) excl_ok = false;
      } else if (inHardB(x, y)) {
        if (!(c1 > 0.0f)) a_ok = false;                 // ⊇ hard mask B
        if (std::fabs(c1 - kPoolHard) > 1e-6f) a_ok = false;
        if (c0 != 0.0f) excl_ok = false;
      } else if (inDilA(x, y)) {
        // Halo around A (A' and B' are far apart, so nearest owner is A).
        if (std::fabs(c0 - kPoolHalo) > 1e-6f) halo_soft_ok = false;
        if (c1 != 0.0f) excl_ok = false;
      } else if (inDilB(x, y)) {
        if (std::fabs(c1 - kPoolHalo) > 1e-6f) halo_soft_ok = false;
        if (c0 != 0.0f) excl_ok = false;
      } else if (inBlob(x, y)) {
        // (c) beyond max_dist from any seed -> dropped from every id.
        if (c0 != 0.0f || c1 != 0.0f) blob_dropped = false;
      } else {
        // Empty pool -> nothing to assign.
        if (c0 != 0.0f || c1 != 0.0f) blob_dropped = false;
      }
    }
  check(a_ok, "(a) each person's coverage ⊇ its hard mask (at the soft value)");
  check(halo_soft_ok,
        "(a) dilated halo assigned to nearest person at MatAnyone soft value");
  check(excl_ok, "(b) the two persons stay pixel-exclusive");
  check(blob_dropped, "(c) stray far blob (beyond max_dist) dropped from all ids");

  // --- (d) manifest / names round-trip the input ---------------------------
  check(out.manifest == input.manifest, "(d) output manifest == input manifest");
  check(out.type_name == input.type_name, "(d) output type_name == input");

  // Sanity: some coverage actually landed for each person.
  double sum0 = 0, sum1 = 0;
  for (size_t px = 0; px < npix; ++px) { sum0 += cov0[px]; sum1 += cov1[px]; }
  check(sum0 > 0.0 && sum1 > 0.0, "both persons received coverage");

  // --- clamp-drops-all: a pool with only the far blob yields no persons ----
  {
    std::vector<float> blob_only(npix, 0.0f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (inBlob(x, y)) blob_only[static_cast<size_t>(y) * W + x] = kPoolBlob;
    const CryptoResult r = PartitionCryptomatte(input.layers, ids, blob_only, W,
                                                H, kMaxDist, kLevels, "person");
    check(r.layers.empty(), "blob-only pool -> no owned coverage -> empty layers");
    check(r.manifest == "{}", "blob-only pool -> empty manifest");
  }

  std::printf(
      "matte_partition_validate: W=%d H=%d, manifest=%s, sum0=%.1f sum1=%.1f\n",
      W, H, out.manifest.c_str(), sum0, sum1);
  if (g_fails == 0) std::printf("matte_partition_validate: all checks passed\n");
  return g_fails == 0 ? 0 : 1;
}
