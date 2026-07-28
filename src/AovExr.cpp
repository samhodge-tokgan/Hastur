// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// AovExr.cpp — see AovExr.h. Mirrors the tinyexr planar-float write in CryptoExr.cpp.
#include "AovExr.h"

#include <algorithm>
#include <cstring>
#include <vector>

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

struct Chan {
  std::string name;
  std::vector<float> data;
};

}  // namespace

bool WriteAovExr(const std::string& path, int W, int H,
                 const std::vector<std::string>& names,
                 const std::vector<PersonAov>& aovs, ExrCompression compression) {
  if (W <= 0 || H <= 0 || path.empty()) return false;
  const size_t npix = static_cast<size_t>(W) * H;

  std::vector<Chan> chans;
  std::vector<float> preview(npix, 0.f);  // Color RGBA = union of alphas (preview)
  const size_t np = std::min(aovs.size(), names.size());
  for (size_t i = 0; i < np; ++i) {
    const PersonAov& a = aovs[i];
    if (a.coverage.size() != npix) continue;  // no mesh / empty person -> skip
    const std::string& nm = names[i];
    { Chan c; c.name = nm + ".A"; c.data = a.coverage; chans.push_back(std::move(c)); }
    if (a.pref.size() == npix * 3) {
      static const char* kPref[3] = {"Pref.R", "Pref.G", "Pref.B"};
      for (int k = 0; k < 3; ++k) {
        Chan c; c.name = nm + "." + kPref[k]; c.data.resize(npix);
        for (size_t px = 0; px < npix; ++px) c.data[px] = a.pref[px * 3 + k];
        chans.push_back(std::move(c));
      }
    }
    if (a.depth.size() == npix) {
      Chan c; c.name = nm + ".Z"; c.data = a.depth; chans.push_back(std::move(c));
    }
    for (size_t px = 0; px < npix; ++px)
      if (a.coverage[px] > preview[px]) preview[px] = a.coverage[px];
  }
  if (chans.empty()) return false;
  static const char* kRGBA[4] = {"R", "G", "B", "A"};
  for (int c = 0; c < 4; ++c) { Chan ch; ch.name = kRGBA[c]; ch.data = preview; chans.push_back(std::move(ch)); }

  // OpenEXR readers conventionally expect alphabetical channel order.
  std::sort(chans.begin(), chans.end(),
            [](const Chan& a, const Chan& b) { return a.name < b.name; });
  const int nch = static_cast<int>(chans.size());

  EXRHeader header; InitEXRHeader(&header);
  EXRImage image; InitEXRImage(&image);
  image.num_channels = nch; image.width = W; image.height = H;
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

  const char* err = nullptr;
  const int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
  if (err) FreeEXRErrorMessage(err);
  // header/image reference our stack vectors — do NOT FreeEXRHeader/FreeEXRImage.
  return ret == TINYEXR_SUCCESS;
}

}  // namespace hastur
