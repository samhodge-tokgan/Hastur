// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// exr_write_validate — self-contained checks for WriteCryptoExr (src/CryptoExr.*),
// the direct-to-disk Cryptomatte EXR writer the Matte Partition node uses instead
// of the host's (broken, on Natron 2.5) multi-plane Write.
//
// It builds a synthetic input Cryptomatte from two hard person boxes + a pooled
// soft alpha, partitions it with PartitionCryptomatte, writes an EXR to a temp path
// via WriteCryptoExr, reads it back with tinyexr, and asserts:
//
//   (a) the file round-trips: dimensions, and the pooled matte lands in Color RGBA
//   (b) the Cryptomatte channels hastur.CryptoObject00/01.{R,G,B,A} round-trip
//       BIT-EXACT (exact float32 ids survive the lossless ZIP codec)
//   (c) the cryptomatte/<key>/{name,manifest,hash,conversion} attributes are present,
//       the key == CryptoTypeKey("person"), and the manifest matches + parses (JSON)
//   (d) (stretch) a supplied hastur/skeleton JSON string attribute round-trips
//
// No OFX / ONNX / GPU. Exit code 0 = all pass.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "CryptoExr.h"
#include "Cryptomatte.h"
#include "MattePartition.h"
#include "tinyexr.h"

using namespace hastur;

static int g_fails = 0;
static void check(bool ok, const char* msg) {
  if (!ok) {
    std::printf("FAIL: %s\n", msg);
    ++g_fails;
  }
}

// --- geometry ---------------------------------------------------------------
static const int W = 40, H = 30;
static bool inBox(int x, int y, int x0, int x1, int y0, int y1) {
  return x >= x0 && x < x1 && y >= y0 && y < y1;
}
static bool inA(int x, int y) { return inBox(x, y, 4, 14, 6, 22); }   // person_00
static bool inB(int x, int y) { return inBox(x, y, 24, 34, 6, 22); }  // person_01

// Find a named channel index in a parsed EXRImage/EXRHeader (or -1).
static int channelIndex(const EXRHeader& hdr, const char* name) {
  for (int i = 0; i < hdr.num_channels; ++i)
    if (std::strcmp(hdr.channels[i].name, name) == 0) return i;
  return -1;
}

// Read a string custom attribute by name (empty if absent).
static std::string attr(const EXRHeader& hdr, const std::string& name) {
  for (int i = 0; i < hdr.num_custom_attributes; ++i) {
    const EXRAttribute& a = hdr.custom_attributes[i];
    if (name == a.name)
      return std::string(reinterpret_cast<const char*>(a.value),
                         static_cast<size_t>(a.size));
  }
  return std::string();
}

int main() {
  const size_t npix = static_cast<size_t>(W) * H;

  // --- synthetic input Cryptomatte + pooled soft alpha ----------------------
  std::vector<CryptoPerson> in_persons(2);
  in_persons[0].name = "person_00";
  in_persons[1].name = "person_01";
  in_persons[0].coverage.assign(npix, 0.f);
  in_persons[1].coverage.assign(npix, 0.f);
  std::vector<float> pool(npix, 0.f);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t px = static_cast<size_t>(y) * W + x;
      if (inA(x, y)) { in_persons[0].coverage[px] = 1.f; pool[px] = 1.f; }
      if (inB(x, y)) { in_persons[1].coverage[px] = 1.f; pool[px] = 0.5f; }
    }
  const int kLevels = 2;
  const CryptoResult input = BuildCryptomatte(W, H, "person", in_persons, kLevels);
  const std::vector<CryptoLabel> ids = {
      {"person_00", CryptoIdFloat("person_00")},
      {"person_01", CryptoIdFloat("person_01")},
  };
  const CryptoResult out =
      PartitionCryptomatte(input.layers, ids, pool, W, H, /*max_dist=*/64.f,
                           kLevels, "person");
  check(!out.layers.empty(), "partition produced crypto layers");

  // --- write to a temp EXR (lossless ZIP) + a skeleton string attribute -----
  const char* td = std::getenv("TMPDIR");
  std::string dir = td && *td ? td : "/tmp";
  if (dir.back() == '/') dir.pop_back();
  const std::string path = dir + "/hastur_exr_write_validate.exr";
  const std::string skel_json =
      "{\"frame\":7,\"people\":[{\"track_id\":0},{\"track_id\":1}]}";

  const bool wrote = WriteCryptoExr(path, pool, out, kLevels,
                                    ExrCompression::kZIP, skel_json);
  check(wrote, "WriteCryptoExr returned success");

  // --- read it back with tinyexr --------------------------------------------
  EXRVersion ver;
  check(ParseEXRVersionFromFile(&ver, path.c_str()) == TINYEXR_SUCCESS,
        "ParseEXRVersionFromFile");
  EXRHeader hdr;
  InitEXRHeader(&hdr);
  const char* err = nullptr;
  check(ParseEXRHeaderFromFile(&hdr, &ver, path.c_str(), &err) == TINYEXR_SUCCESS,
        "ParseEXRHeaderFromFile");
  if (err) { FreeEXRErrorMessage(err); err = nullptr; }
  // Force float32 read-back for every channel.
  for (int i = 0; i < hdr.num_channels; ++i)
    hdr.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  EXRImage img;
  InitEXRImage(&img);
  check(LoadEXRImageFromFile(&img, &hdr, path.c_str(), &err) == TINYEXR_SUCCESS,
        "LoadEXRImageFromFile");
  if (err) { FreeEXRErrorMessage(err); err = nullptr; }

  // (a) dimensions + Color RGBA carries the pooled matte.
  check(img.width == W && img.height == H, "(a) EXR dimensions round-trip");
  const int cR = channelIndex(hdr, "R"), cA = channelIndex(hdr, "A");
  check(cR >= 0 && cA >= 0, "(a) Color R/A channels present");
  bool color_ok = true;
  if (cR >= 0) {
    const float* R = reinterpret_cast<const float*>(img.images[cR]);
    for (size_t px = 0; px < npix; ++px)
      if (std::fabs(R[px] - pool[px]) > 0.f) color_ok = false;
  }
  check(color_ok, "(a) Color R == pooled matte (exact)");

  // (b) crypto channels round-trip BIT-EXACT (exact float32 ids, lossless codec).
  bool crypto_exact = true;
  bool ids_seen = false;
  for (int L = 0; L < kLevels; ++L) {
    char pfx[32];
    std::snprintf(pfx, sizeof(pfx), "hastur.CryptoObject%02d", L);
    const char* comps[4] = {"R", "G", "B", "A"};
    const std::vector<float>& layer = out.layers[L];
    for (int c = 0; c < 4; ++c) {
      const std::string nm = std::string(pfx) + "." + comps[c];
      const int ci = channelIndex(hdr, nm.c_str());
      if (ci < 0) { crypto_exact = false; continue; }
      const float* D = reinterpret_cast<const float*>(img.images[ci]);
      for (size_t px = 0; px < npix; ++px) {
        const float expect = layer[px * 4 + static_cast<size_t>(c)];
        // BIT-EXACT compare (memcmp of the raw float bits).
        if (std::memcmp(&D[px], &expect, sizeof(float)) != 0) crypto_exact = false;
        if (c == 0 && expect != 0.f) ids_seen = true;  // an id landed on rank 0
      }
    }
  }
  check(crypto_exact, "(b) crypto channels round-trip BIT-EXACT");
  check(ids_seen, "(b) at least one non-zero Cryptomatte id was written");

  // (c) Cryptomatte metadata attributes present + manifest matches/parses.
  const std::string key = CryptoTypeKey("person");
  const std::string base = "cryptomatte/" + key + "/";
  check(attr(hdr, base + "name") == "person", "(c) cryptomatte/<key>/name == person");
  const std::string man = attr(hdr, base + "manifest");
  check(!man.empty(), "(c) cryptomatte/<key>/manifest present");
  check(man == out.manifest, "(c) manifest attribute == crypto.manifest");
  // Minimal JSON sanity: object braces + both person names appear.
  check(man.front() == '{' && man.back() == '}', "(c) manifest parses as a JSON object");
  check(man.find("person_00") != std::string::npos &&
            man.find("person_01") != std::string::npos,
        "(c) manifest lists both persons");
  check(attr(hdr, base + "hash") == "MurmurHash3_32", "(c) hash attribute");
  check(attr(hdr, base + "conversion") == "uint32_to_float32",
        "(c) conversion attribute");

  // (d) stretch: the skeleton JSON string attribute round-trips verbatim.
  check(attr(hdr, "hastur/skeleton") == skel_json,
        "(d) hastur/skeleton string attribute round-trips");

  FreeEXRImage(&img);
  FreeEXRHeader(&hdr);
  std::remove(path.c_str());

  std::printf("exr_write_validate: %dx%d, %d channels, key=%s, manifest=%s\n", W, H,
              img.num_channels, key.c_str(), out.manifest.c_str());
  if (g_fails == 0) std::printf("exr_write_validate: all checks passed\n");
  return g_fails == 0 ? 0 : 1;
}
