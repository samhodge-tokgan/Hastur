// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// CryptoExr.cpp — see CryptoExr.h for the contract and channel/attribute layout.

#include "CryptoExr.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>

#include "Cryptomatte.h"  // CryptoTypeKey
#include "tinyexr.h"

namespace hastur {
namespace {

int ToTinyExrCompression(ExrCompression c) {
  switch (c) {
    case ExrCompression::kNone: return TINYEXR_COMPRESSIONTYPE_NONE;
    case ExrCompression::kRLE:  return TINYEXR_COMPRESSIONTYPE_RLE;
    case ExrCompression::kZIPS: return TINYEXR_COMPRESSIONTYPE_ZIPS;
    case ExrCompression::kZIP:  return TINYEXR_COMPRESSIONTYPE_ZIP;
    case ExrCompression::kPIZ:  return TINYEXR_COMPRESSIONTYPE_PIZ;
  }
  return TINYEXR_COMPRESSIONTYPE_ZIP;
}

// One planar float32 channel (tinyexr stores channels separately, not interleaved).
struct Chan {
  std::string name;
  std::vector<float> data;
};

}  // namespace

bool WriteCryptoExr(const std::string& path, const std::vector<float>& pool,
                    const CryptoResult& crypto, int levels,
                    ExrCompression compression, const std::string& skeleton_json) {
  const int W = crypto.width, H = crypto.height;
  if (W <= 0 || H <= 0 || path.empty()) return false;
  const size_t npix = static_cast<size_t>(W) * H;

  int nLevels = levels < 0 ? 0 : levels;
  if (nLevels > static_cast<int>(crypto.layers.size()))
    nLevels = static_cast<int>(crypto.layers.size());

  // The Cryptomatte typename. The spec REQUIRES the numbered crypto channel-layers
  // to be prefixed with this exact string ("<type>00", "<type>01"), and it must
  // equal the "cryptomatte/<key>/name" metadata below — Nuke/Fusion's Cryptomatte
  // node reads `name`, then looks for layers `<name>NN`. A mismatch (e.g. channels
  // "hastur.CryptoObject00" with name "person") makes the node find nothing = the
  // matte reads as useless in Nuke. So derive the prefix from `type`, not a literal.
  const std::string type = crypto.type_name.empty() ? std::string("person")
                                                    : crypto.type_name;

  // --- Build the channel list --------------------------------------------------
  // Color R,G,B,A = pooled matte (m,m,m,m) — the beauty/preview; then per level the
  // four packed crypto channels named "<type>NN.[RGBA]" (spec-compliant layers).
  std::vector<Chan> chans;
  chans.reserve(4 + static_cast<size_t>(nLevels) * 4);

  auto poolAt = [&](size_t px) { return px < pool.size() ? pool[px] : 0.f; };
  static const char* kRGBA[4] = {"R", "G", "B", "A"};
  for (int c = 0; c < 4; ++c) {
    Chan ch;
    ch.name = kRGBA[c];
    ch.data.resize(npix);
    for (size_t px = 0; px < npix; ++px) ch.data[px] = poolAt(px);
    chans.push_back(std::move(ch));
  }
  for (int L = 0; L < nLevels; ++L) {
    const std::vector<float>& layer = crypto.layers[L];
    char pfx[80];
    std::snprintf(pfx, sizeof(pfx), "%s%02d", type.c_str(), L);
    for (int c = 0; c < 4; ++c) {
      Chan ch;
      ch.name = std::string(pfx) + "." + kRGBA[c];
      ch.data.assign(npix, 0.f);
      if (layer.size() >= npix * 4)
        for (size_t px = 0; px < npix; ++px)
          ch.data[px] = layer[px * 4 + static_cast<size_t>(c)];
      chans.push_back(std::move(ch));
    }
  }

  // OpenEXR channels are conventionally stored alphabetically; strict readers rely
  // on it. Sort by name — readers (incl. our test) match by name, so ordering is
  // otherwise irrelevant, but this keeps Nuke/Fusion happy.
  std::sort(chans.begin(), chans.end(),
            [](const Chan& a, const Chan& b) { return a.name < b.name; });

  const int nch = static_cast<int>(chans.size());

  EXRHeader header;
  InitEXRHeader(&header);
  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = nch;
  image.width = W;
  image.height = H;
  std::vector<unsigned char*> imgPtrs(nch);
  for (int i = 0; i < nch; ++i)
    imgPtrs[i] = reinterpret_cast<unsigned char*>(chans[i].data.data());
  image.images = imgPtrs.data();

  header.num_channels = nch;
  std::vector<EXRChannelInfo> chinfo(nch);
  std::vector<int> pixTypes(nch, TINYEXR_PIXELTYPE_FLOAT);
  std::vector<int> reqTypes(nch, TINYEXR_PIXELTYPE_FLOAT);
  for (int i = 0; i < nch; ++i) {
    std::memset(&chinfo[i], 0, sizeof(EXRChannelInfo));
    std::strncpy(chinfo[i].name, chans[i].name.c_str(), 255);
    chinfo[i].name[255] = '\0';
    chinfo[i].pixel_type = TINYEXR_PIXELTYPE_FLOAT;
  }
  header.channels = chinfo.data();
  header.pixel_types = pixTypes.data();
  header.requested_pixel_types = reqTypes.data();
  header.compression_type = ToTinyExrCompression(compression);

  // --- Cryptomatte metadata attributes (+ optional skeleton) -------------------
  // EXRAttribute.value is NOT owned by tinyexr, so the backing strings must outlive
  // SaveEXRImageToFile. A deque keeps element addresses stable across push_backs.
  std::deque<std::string> storage;
  std::vector<EXRAttribute> attrs;
  auto addStr = [&](const std::string& name, const std::string& val) {
    storage.push_back(val);
    EXRAttribute a;
    std::memset(&a, 0, sizeof(a));
    std::strncpy(a.name, name.c_str(), 255);
    a.name[255] = '\0';
    std::strncpy(a.type, "string", 255);
    a.value = reinterpret_cast<unsigned char*>(&storage.back()[0]);
    a.size = static_cast<int>(storage.back().size());
    attrs.push_back(a);
  };

  const std::string key = CryptoTypeKey(type);
  const std::string base = "cryptomatte/" + key + "/";
  addStr(base + "name", type);
  addStr(base + "manifest", crypto.manifest.empty() ? std::string("{}")
                                                    : crypto.manifest);
  addStr(base + "hash", "MurmurHash3_32");
  addStr(base + "conversion", "uint32_to_float32");
  if (!skeleton_json.empty()) addStr("hastur/skeleton", skeleton_json);

  header.num_custom_attributes = static_cast<int>(attrs.size());
  header.custom_attributes = attrs.empty() ? nullptr : attrs.data();

  const char* err = nullptr;
  const int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
  if (err) FreeEXRErrorMessage(err);
  // header/image reference our stack vectors (not malloc'd by tinyexr), so do NOT
  // call FreeEXRHeader/FreeEXRImage here — that would free memory we own.
  return ret == TINYEXR_SUCCESS;
}

}  // namespace hastur
