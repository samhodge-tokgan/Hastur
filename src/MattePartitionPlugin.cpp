// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Matte Partition OpenFX plugin — the SECOND plugin in the Hastur bundle (Phase
// 6b-ii). It wraps the tested PartitionCryptomatte CORE (src/MattePartition.h) in
// a multi-plane OFX node so it runs in a Natron/Nuke graph alongside SAM 3D Body.
//
// Data flow (per frame, stateless):
//
//   Cryptomatte input  --(hastur.CryptoObject00/01 planes)-->  in_layers
//   Pool input         --(A channel / luminance)------------>  pool (soft alpha)
//                                   |
//                        PartitionCryptomatte(...)  (Voronoi nearest-owner fill)
//                                   |
//   Output  <--(fresh hastur.CryptoObject00/01 planes + baked manifest)
//
// It reuses HasturPlugin's multi-plane machinery verbatim via MultiPlane.h (the
// PlaneDef/EncodePlane/PlaneForEncoded token-match fallback / plane+property
// suites / render-plane stash), and the Cryptomatte + partition cores. NO ONNX,
// NO GPU: the node is pure C++ image processing.
//
// Like the Hastur AOV path, the multi-plane wire protocol here is source-accurate
// against the Foundry/Natron extension but has NOT yet been exercised in a real
// host from this repo — that smoke is Phase 6c. On ANY failure the node degrades
// gracefully (writes the pooled matte / a black frame) and never crashes the host.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsProcessing.H"

#include "CryptoExr.h"
#include "Cryptomatte.h"
#include "MattePartition.h"
#include "MultiPlane.h"
#include "Register.h"

#define kMpPluginName "Matte Partition"
#define kMpPluginGrouping "Tokgan"
#define kMpPluginDescription \
  "Partition a pooled soft alpha (e.g. MatAnyone2 / Kthanid) into a fresh " \
  "per-person Cryptomatte, using an upstream Hastur Cryptomatte's front-most " \
  "IDs as Voronoi seeds. Internal seams follow Cryptomatte ownership; external " \
  "edges keep the pooled matte's softness."
#define kMpPluginIdentifier "com.tokgan.MattePartition"
#define kMpPluginVersionMajor 0
#define kMpPluginVersionMinor 1

// Clip names.
#define kClipCryptomatte "Cryptomatte"
#define kClipPool "Pool"

// Param names.
#define kParamMaxDist "maxDist"
#define kParamSeedThresh "seedThresh"
#define kParamCryptoLevels "cryptoLevels"
#define kMpParamCryptoManifest "cryptoManifest"
#define kParamCryptoExrPath "cryptoExrPath"
#define kParamCryptoCompression "cryptoCompression"
#define kParamSkeletonSidecar "skeletonSidecar"
#define kMpParamPickName "pickName"

// How many person_NN slots the baked candidate manifest enumerates. The input
// scan maps IDs against this same table, so emitted names are always a subset.
static constexpr int kMpManifestPeople = 256;

namespace {

// The two Cryptomatte output/input planes this node reads and emits. `aov` is the
// Cryptomatte layer index (rank pair) that plane carries. IDs match Hastur's so a
// Hastur Cryptomatte feeds straight in and the output round-trips downstream.
const hplane::PlaneDef kMpPlanes[] = {
    {0, "hastur.CryptoObject00", "CryptoObject00", {"R", "G", "B", "A"}, 4},
    {1, "hastur.CryptoObject01", "CryptoObject01", {"R", "G", "B", "A"}, 4},
};
constexpr size_t kMpPlaneCount = sizeof(kMpPlanes) / sizeof(kMpPlanes[0]);

// Set by AppendMattePartition (before any action); the delegating main entry
// passes it to mainEntryStr so the Support library dispatches to THIS plugin.
std::string g_mpUid;
// Planes the host asked for this render call (stashed from the render inArgs).
std::vector<std::string> g_mpRenderPlanes;

// Cryptomatte ids are exact float32 bit patterns that survive the EXR round-trip.
// Compare exactly: a relative epsilon collapses distinct ids across the huge id
// dynamic range, causing the manifest-less scan below to misassign persons.
inline bool IdEquals(float a, float b) {
  return a == b;
}

template <class T>
inline T ClampToDepth(float v, float maxv) {
  float s = v * maxv;
  if (s < 0.f) s = 0.f;
  if (s > maxv) s = maxv;
  return static_cast<T>(s + 0.5f);
}

// The static candidate manifest: person_00..person_(N-1) -> 8-hex hash. Fully
// determined by the names (CryptoIdHex is deterministic), so it can be baked at
// describe time — no render-time param write (which OFX forbids).
std::string BuildCandidateManifest(int n) {
  std::string m = "{";
  for (int i = 0; i < n; ++i) {
    char nm[16];
    std::snprintf(nm, sizeof(nm), "person_%02d", i);
    if (i) m += ",";
    m += "\"" + std::string(nm) + "\":\"" + hastur::CryptoIdHex(nm) + "\"";
  }
  m += "}";
  return m;
}

// GetClipComponents: we EMIT colour + the two Cryptomatte planes on the output,
// we READ colour + the two Cryptomatte planes from the Cryptomatte input, and
// colour from the Pool input. Pass everything else through from the Cryptomatte
// clip.
OfxStatus MpGetClipComponents(OfxPropertySetHandle outArgs) {
  const OfxPropertySuiteV1* props = hplane::PropSuite();
  if (!props) return kOfxStatReplyDefault;
  const std::string base = kFnOfxImageEffectActionGetClipComponentsPropString;

  // Output clip: colour + both Cryptomatte planes.
  const std::string outName = base + kOfxImageEffectOutputClipName;
  int i = 0;
  props->propSetString(outArgs, outName.c_str(), i++, kFnOfxImagePlaneColour);
  for (const hplane::PlaneDef& p : kMpPlanes)
    props->propSetString(outArgs, outName.c_str(), i++,
                         hplane::EncodePlane(p).c_str());

  // Cryptomatte input clip: colour + both Cryptomatte planes (we need them).
  const std::string cryptoName = base + kClipCryptomatte;
  int j = 0;
  props->propSetString(outArgs, cryptoName.c_str(), j++, kFnOfxImagePlaneColour);
  for (const hplane::PlaneDef& p : kMpPlanes)
    props->propSetString(outArgs, cryptoName.c_str(), j++,
                         hplane::EncodePlane(p).c_str());

  // Pool input clip: colour only.
  const std::string poolName = base + kClipPool;
  props->propSetString(outArgs, poolName.c_str(), 0, kFnOfxImagePlaneColour);

  props->propSetString(outArgs, kFnOfxImageEffectPropPassThroughClip, 0,
                       kClipCryptomatte);
  return kOfxStatOK;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

class MattePartitionPlugin : public OFX::ImageEffect {
 public:
  explicit MattePartitionPlugin(OfxImageEffectHandle handle)
      : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _cryptoClip = fetchClip(kClipCryptomatte);
    _poolClip = fetchClip(kClipPool);
    _maxDist = fetchDoubleParam(kParamMaxDist);
    _seedThresh = fetchDoubleParam(kParamSeedThresh);
    _cryptoLevels = fetchIntParam(kParamCryptoLevels);
    _cryptoManifest = fetchStringParam(kMpParamCryptoManifest);
    _cryptoExrPath = fetchStringParam(kParamCryptoExrPath);
    _cryptoCompression = fetchChoiceParam(kParamCryptoCompression);
    _skeletonSidecar = fetchStringParam(kParamSkeletonSidecar);
    _pickName = fetchStringParam(kMpParamPickName);
  }

  void render(const OFX::RenderArguments& args) override;

 private:
  bool renderPartition(const OFX::RenderArguments& args);
  void renderBlack(const OFX::RenderArguments& args);

  OFX::Clip* _dstClip = nullptr;
  OFX::Clip* _cryptoClip = nullptr;
  OFX::Clip* _poolClip = nullptr;
  OFX::DoubleParam* _maxDist = nullptr;
  OFX::DoubleParam* _seedThresh = nullptr;
  OFX::IntParam* _cryptoLevels = nullptr;
  OFX::StringParam* _cryptoManifest = nullptr;
  OFX::StringParam* _cryptoExrPath = nullptr;
  OFX::ChoiceParam* _cryptoCompression = nullptr;
  OFX::StringParam* _skeletonSidecar = nullptr;
  OFX::StringParam* _pickName = nullptr;
};

namespace {

// Read one plane (fetched via the Foundry plane suite) into a top-down HWC float
// buffer of `nch` channels over the frame bounds `b`. Plane pixels are bottom-up
// (OFX), so top-down row r maps to OFX y = b.y2-1-r — the exact inverse of the
// write path. Returns false if the plane could not be fetched.
bool ReadPlaneTopDown(OFX::Clip* clip, double time, const std::string& plane,
                      const OfxRectI& b, int W, int H, int nch,
                      std::vector<float>& out) {
  const FnOfxImageEffectPlaneSuiteV1* ps = hplane::PlaneSuite();
  const OfxPropertySuiteV1* props = hplane::PropSuite();
  if (!ps || !props || !clip) return false;
  OfxPropertySetHandle imgProps = nullptr;
  if (ps->clipGetImagePlane(clip->getHandle(), time, plane.c_str(), nullptr,
                            &imgProps) != kOfxStatOK ||
      !imgProps)
    return false;
  void* base = nullptr;
  props->propGetPointer(imgProps, kOfxImagePropData, 0, &base);
  int bnd[4] = {0, 0, 0, 0};
  props->propGetIntN(imgProps, kOfxImagePropBounds, 4, bnd);
  int rowBytes = 0;
  props->propGetInt(imgProps, kOfxImagePropRowBytes, 0, &rowBytes);
  char* depthStr = nullptr;
  props->propGetString(imgProps, kOfxImageEffectPropPixelDepth, 0, &depthStr);
  if (!base) return false;
  OFX::BitDepthEnum tbd = OFX::eBitDepthFloat;
  float scale = 1.f;
  if (depthStr) {
    if (!std::strcmp(depthStr, kOfxBitDepthByte)) { tbd = OFX::eBitDepthUByte; scale = 1.f / 255.f; }
    else if (!std::strcmp(depthStr, kOfxBitDepthShort)) { tbd = OFX::eBitDepthUShort; scale = 1.f / 65535.f; }
  }
  const OfxRectI tb{bnd[0], bnd[1], bnd[2], bnd[3]};
  out.assign(static_cast<size_t>(W) * H * nch, 0.f);
  for (int y = tb.y1; y < tb.y2; ++y) {
    const int rr = b.y2 - 1 - y;  // top-down row
    if (rr < 0 || rr >= H) continue;
    const char* rowp =
        static_cast<const char*>(base) + static_cast<size_t>(y - tb.y1) * rowBytes;
    for (int x = tb.x1; x < tb.x2; ++x) {
      const int cc = x - b.x1;
      if (cc < 0 || cc >= W) continue;
      float* d = &out[(static_cast<size_t>(rr) * W + cc) * nch];
      const size_t off = static_cast<size_t>(x - tb.x1) * nch;
      if (tbd == OFX::eBitDepthFloat) {
        const float* sp = reinterpret_cast<const float*>(rowp) + off;
        for (int c = 0; c < nch; ++c) d[c] = sp[c];  // ids are exact float bits
      } else if (tbd == OFX::eBitDepthUShort) {
        const unsigned short* sp = reinterpret_cast<const unsigned short*>(rowp) + off;
        for (int c = 0; c < nch; ++c) d[c] = sp[c] * scale;
      } else {
        const unsigned char* sp = reinterpret_cast<const unsigned char*>(rowp) + off;
        for (int c = 0; c < nch; ++c) d[c] = sp[c] * scale;
      }
    }
  }
  return true;
}

// Expand a printf-style frame pattern (e.g. ".../crypto_%04d.exr") with the frame
// number. If the pattern has no conversion the frame arg is harmlessly ignored.
std::string ExpandFramePattern(const std::string& pat, long frame) {
  if (pat.empty()) return pat;
  const int n = std::snprintf(nullptr, 0, pat.c_str(), frame);
  if (n < 0) return pat;  // malformed pattern -> leave as-is
  std::vector<char> buf(static_cast<size_t>(n) + 1);
  std::snprintf(buf.data(), buf.size(), pat.c_str(), frame);
  return std::string(buf.data());
}

// Read a whole file into a string (empty on any failure / absence).
std::string ReadFileToString(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::string();
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// The cryptoCompression choice options, in declared order. ALL are lossless (the
// crypto ids must survive bit-exact); the writer enforces this by construction.
hastur::ExrCompression CompressionForChoice(int idx) {
  switch (idx) {
    case 0: return hastur::ExrCompression::kZIP;
    case 1: return hastur::ExrCompression::kZIPS;
    case 2: return hastur::ExrCompression::kPIZ;
    case 3: return hastur::ExrCompression::kRLE;
    case 4: return hastur::ExrCompression::kNone;
    default: return hastur::ExrCompression::kZIP;
  }
}

}  // namespace

// Returns true if it produced output; false to fall back to a black frame.
bool MattePartitionPlugin::renderPartition(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) return false;
  // NB: dst may be non-RGBA on multi-plane hosts (the default plane is used only
  // for BOUNDS here; the Colour matte is written into whichever buffer we fetch
  // below). Do not bail on non-RGBA -- that black-frames the whole partition.
  const bool dstIsRGBA = dst->getPixelComponents() == OFX::ePixelComponentRGBA;
  const OfxRectI b = dst->getBounds();
  const int W = b.x2 - b.x1, H = b.y2 - b.y1;
  if (W <= 0 || H <= 0) return false;
  const size_t npix = static_cast<size_t>(W) * H;

  double dMaxDist = 64, dSeed = 0.5;
  int levels = 2;
  _maxDist->getValue(dMaxDist);
  _seedThresh->getValue(dSeed);
  _cryptoLevels->getValue(levels);
  const float maxDist = static_cast<float>(dMaxDist);
  const float seedThresh = static_cast<float>(dSeed);
  if (levels < 1) levels = 1;

  // (1) Read the input Cryptomatte planes -> in_layers (each W*H*4 top-down), and
  // the Pool clip -> pool (soft alpha). Both go through the shared plane machinery.
  std::vector<std::vector<float>> in_layers;
  if (_cryptoClip && _cryptoClip->isConnected()) {
    for (const hplane::PlaneDef& p : kMpPlanes) {
      std::vector<float> layer;
      if (ReadPlaneTopDown(_cryptoClip, args.time, hplane::EncodePlane(p), b, W, H,
                           4, layer))
        in_layers.push_back(std::move(layer));
      else
        break;  // rank 0 is what seeds; stop at the first missing layer
    }
  }

  // Pool soft alpha: prefer the A channel; if A is uniformly opaque (a matte piped
  // as grey RGB with alpha=1), fall back to luminance of RGB.
  std::vector<float> pool(npix, 0.f);
  if (_poolClip && _poolClip->isConnected()) {
    std::unique_ptr<OFX::Image> pimg(_poolClip->fetchImage(args.time));
    if (pimg.get()) {
      const OFX::BitDepthEnum pbd = pimg->getPixelDepth();
      const int pnc = pimg->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
      const OfxRectI pb = pimg->getBounds();
      float scale = 1.f;
      if (pbd == OFX::eBitDepthUShort) scale = 1.f / 65535.f;
      else if (pbd == OFX::eBitDepthUByte) scale = 1.f / 255.f;
      std::vector<float> alpha(npix, 0.f), luma(npix, 0.f);
      bool alphaVaries = false;
      auto readpx = [&](const void* praw, size_t px) {
        float r = 0, g = 0, bl = 0, a = 1.f;
        if (pbd == OFX::eBitDepthFloat) {
          const float* p = static_cast<const float*>(praw);
          r = p[0]; g = pnc > 1 ? p[1] : r; bl = pnc > 2 ? p[2] : r;
          a = pnc > 3 ? p[3] : 1.f;
        } else if (pbd == OFX::eBitDepthUShort) {
          const unsigned short* p = static_cast<const unsigned short*>(praw);
          r = p[0] * scale; g = pnc > 1 ? p[1] * scale : r;
          bl = pnc > 2 ? p[2] * scale : r; a = pnc > 3 ? p[3] * scale : 1.f;
        } else {
          const unsigned char* p = static_cast<const unsigned char*>(praw);
          r = p[0] * scale; g = pnc > 1 ? p[1] * scale : r;
          bl = pnc > 2 ? p[2] * scale : r; a = pnc > 3 ? p[3] * scale : 1.f;
        }
        luma[px] = 0.2126f * r + 0.7152f * g + 0.0722f * bl;
        alpha[px] = a;
        if (a < 0.999f) alphaVaries = true;
      };
      for (int r = 0; r < H; ++r) {
        const int oy = b.y2 - 1 - r;
        if (oy < pb.y1 || oy >= pb.y2) continue;
        for (int c = 0; c < W; ++c) {
          const int ox = b.x1 + c;
          if (ox < pb.x1 || ox >= pb.x2) continue;
          const void* praw = pimg->getPixelAddress(ox, oy);
          if (praw) readpx(praw, static_cast<size_t>(r) * W + c);
        }
      }
      pool = alphaVaries ? std::move(alpha) : std::move(luma);
    }
  }

  // ---- PICK MODE: the pickName param is a 3-way seed selector that decodes coverage
  // out of the input Cryptomatte into `pool`, then lets the SAME tested partition/emit
  // path below run. The Colour output is the decoded coverage matte (the seed a
  // downstream Kthanid needs); on Natron 2.5 only Colour flows node-to-node (its
  // Color.RGBA-only plane limit) which is exactly that seed. Modes:
  //   ""          -> normal Voronoi partition (unchanged; Pool = a refined union).
  //   "person_NN" -> isolate ONE person's coverage  (max-fidelity per-person seed;
  //                  one Kthanid pass per id -> highest quality, highest cost).
  //   "*"         -> UNION of every id's coverage    (one big matte -> ONE Kthanid
  //                  pass for the whole frame; a second MattePartition (mode "") then
  //                  Voronoi-splits the refined union back per person -> cheapest, and
  //                  "enough detail" when subjects don't overlap).
  std::string pickName;
  if (_pickName) _pickName->getValue(pickName);
  const bool pickMode = !pickName.empty();
  const bool unionAll = (pickName == "*");
  if (pickMode) {
    std::vector<float> cov(npix, 0.f);
    const float tgt = unionAll ? 0.f : hastur::CryptoIdFloat(pickName);
    for (const auto& L : in_layers) {
      if (L.size() < npix * 4) continue;
      for (size_t px = 0; px < npix; ++px) {  // two ranks/layer: (R,G) and (B,A)
        if (unionAll) {                        // union = total matte alpha
          cov[px] += L[px * 4 + 1] + L[px * 4 + 3];
        } else {
          if (IdEquals(L[px * 4 + 0], tgt)) cov[px] += L[px * 4 + 1];
          if (IdEquals(L[px * 4 + 2], tgt)) cov[px] += L[px * 4 + 3];
        }
      }
    }
    for (float& v : cov) { if (v < 0.f) v = 0.f; else if (v > 1.f) v = 1.f; }
    pool = std::move(cov);  // Colour output + partition seed = the picked coverage
  }

  // (2) IDs without an external manifest: scan rank 0 of the front-most input
  // layer for distinct id floats (coverage > seedThresh) and map each to its
  // person_NN via the static person_00..255 table (relative-epsilon match). This
  // is robust to sparse/large track ids and needs no host metadata.
  std::vector<hastur::CryptoLabel> table;
  table.reserve(kMpManifestPeople);
  for (int i = 0; i < kMpManifestPeople; ++i) {
    char nm[16];
    std::snprintf(nm, sizeof(nm), "person_%02d", i);
    table.push_back({nm, hastur::CryptoIdFloat(nm)});
  }
  std::vector<hastur::CryptoLabel> ids;
  if (pickMode && !unionAll) {
    // Single-id partition: every pooled pixel belongs to the picked person.
    ids.push_back({pickName, hastur::CryptoIdFloat(pickName)});
  } else if (pickMode && unionAll) {
    // Union seed: no per-id partition here (a downstream MattePartition splits the
    // refined union). Leave ids empty -> emit is the union matte on Colour.
  } else if (!in_layers.empty() && in_layers[0].size() >= npix * 4) {
    const std::vector<float>& l0 = in_layers[0];
    std::vector<char> seen(table.size(), 0);
    for (size_t px = 0; px < npix; ++px) {
      const float id0 = l0[px * 4 + 0];
      const float cov0 = l0[px * 4 + 1];
      if (cov0 <= seedThresh) continue;
      for (size_t k = 0; k < table.size(); ++k) {
        if (!seen[k] && IdEquals(id0, table[k].id)) {
          seen[k] = 1;
          ids.push_back(table[k]);
          break;
        }
      }
    }
  }

  // (3) Partition. NOTE: the core bakes its own seed-coverage threshold at 0.5
  // today; `seedThresh` here gates the id SCAN above but is not yet threaded into
  // PartitionCryptomatte (would require changing the tested core signature). The
  // param is wired and functional for the scan; core threading is a follow-up.
  hastur::CryptoResult out =
      hastur::PartitionCryptomatte(in_layers, ids, pool, W, H, maxDist, levels);

  // (3b) Write the Cryptomatte EXR straight to disk when a path is set. This is the
  // node's own multi-plane EXR emitter — Natron 2.5 HANGS assembling a plugin-
  // produced multi-plane Write, so the node bypasses the host Write entirely and
  // writes a decodable Cryptomatte EXR itself (color = pooled matte; hastur.
  // CryptoObjectNN planes = out.layers; Psyop metadata attributes). Empty path =
  // disabled (zero cost). Best-effort: any failure is swallowed (never fails the
  // render). Idempotent per frame (overwrite).
  std::string exrPat;
  if (_cryptoExrPath) _cryptoExrPath->getValue(exrPat);
  if (!exrPat.empty()) {
    const long frame = std::lround(args.time);
    const std::string exrPath = ExpandFramePattern(exrPat, frame);

    // STRETCH: fold the per-frame skeleton JSON sidecar into the same EXR as the
    // string attribute hastur/skeleton, so {matte + identity + pose} ride together.
    std::string skelJson;
    if (_skeletonSidecar) {
      std::string skelDir;
      _skeletonSidecar->getValue(skelDir);
      if (!skelDir.empty()) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "skeleton_%04ld.json", frame);
        std::string sep =
            (skelDir.back() == '/' || skelDir.back() == '\\') ? "" : "/";
        skelJson = ReadFileToString(skelDir + sep + nm);  // empty if absent -> skip
      }
    }

    int comp = 0;
    if (_cryptoCompression) _cryptoCompression->getValue(comp);
    hastur::WriteCryptoExr(exrPath, pool, out, levels,
                           CompressionForChoice(comp), skelJson);
  }

  // (4) Emit. Multi-plane hosts request planes via g_mpRenderPlanes; satisfy each.
  // Colour -> the pooled matte (grey + alpha) so the node is viewable; the crypto
  // planes -> their packed layers via the plane suite. Non-multi-plane hosts get
  // just the colour matte (they cannot carry the Cryptomatte planes anyway).
  const OfxRectI& win = args.renderWindow;
  auto composeCrypto = [&](int lvl, int rr, int cc, float o[4]) {
    o[0] = o[1] = o[2] = o[3] = 0.f;
    const size_t idx = (static_cast<size_t>(rr) * W + cc) * 4;
    if (lvl >= 0 && lvl < static_cast<int>(out.layers.size()) &&
        idx + 3 < out.layers[lvl].size()) {
      const float* L = &out.layers[lvl][idx];
      o[0] = L[0]; o[1] = L[1]; o[2] = L[2]; o[3] = L[3];
    }
  };
  auto poolAt = [&](int rr, int cc) {
    const size_t px = static_cast<size_t>(rr) * W + cc;
    return px < pool.size() ? pool[px] : 0.f;
  };

  // Write the pooled matte to the colour (RGBA) output image.
  auto writeColour = [&](OFX::Image* img) {
    const OFX::BitDepthEnum tbd = img->getPixelDepth();
    for (int y = win.y1; y < win.y2; ++y) {
      if (abort()) break;
      const int rr = b.y2 - 1 - y;
      for (int x = win.x1; x < win.x2; ++x) {
        const int cc = x - b.x1;
        float m = 0.f;
        if (rr >= 0 && rr < H && cc >= 0 && cc < W) m = poolAt(rr, cc);
        void* praw = img->getPixelAddress(x, y);
        if (!praw) continue;
        const float o[4] = {m, m, m, m};
        if (tbd == OFX::eBitDepthFloat) {
          float* dp = static_cast<float*>(praw);
          dp[0] = o[0]; dp[1] = o[1]; dp[2] = o[2]; dp[3] = o[3];
        } else if (tbd == OFX::eBitDepthUShort) {
          unsigned short* dp = static_cast<unsigned short*>(praw);
          for (int c = 0; c < 4; ++c) dp[c] = ClampToDepth<unsigned short>(o[c], 65535.f);
        } else {
          unsigned char* dp = static_cast<unsigned char*>(praw);
          for (int c = 0; c < 4; ++c) dp[c] = ClampToDepth<unsigned char>(o[c], 255.f);
        }
      }
    }
  };

  // Write a crypto layer into a raw plane buffer (4 channels) fetched via the
  // plane suite. `tb` is the plane's bounds (bottom-up, like the frame).
  auto writeCryptoRaw = [&](void* base, const OfxRectI& tb, int rowBytes,
                            OFX::BitDepthEnum tbd, int lvl) {
    for (int y = win.y1; y < win.y2; ++y) {
      if (abort()) break;
      if (y < tb.y1 || y >= tb.y2) continue;
      const int rr = b.y2 - 1 - y;
      char* rowp = static_cast<char*>(base) + static_cast<size_t>(y - tb.y1) * rowBytes;
      for (int x = win.x1; x < win.x2; ++x) {
        if (x < tb.x1 || x >= tb.x2) continue;
        const int cc = x - b.x1;
        float o[4] = {0, 0, 0, 0};
        if (rr >= 0 && rr < H && cc >= 0 && cc < W) composeCrypto(lvl, rr, cc, o);
        const size_t off = static_cast<size_t>(x - tb.x1) * 4;
        if (tbd == OFX::eBitDepthFloat) {
          float* dp = reinterpret_cast<float*>(rowp) + off;
          for (int c = 0; c < 4; ++c) dp[c] = o[c];
        } else if (tbd == OFX::eBitDepthUShort) {
          unsigned short* dp = reinterpret_cast<unsigned short*>(rowp) + off;
          for (int c = 0; c < 4; ++c) dp[c] = ClampToDepth<unsigned short>(o[c], 65535.f);
        } else {
          unsigned char* dp = reinterpret_cast<unsigned char*>(rowp) + off;
          for (int c = 0; c < 4; ++c) dp[c] = ClampToDepth<unsigned char>(o[c], 255.f);
        }
      }
    }
  };

  // Write the pooled matte (grey + alpha) into a raw Colour-plane buffer fetched
  // via the plane suite, for multi-plane hosts whose default dst plane is not
  // RGBA. Mirrors writeCryptoRaw but emits {m,m,m,m}.
  auto writePoolRaw = [&](void* base, const OfxRectI& tb, int rowBytes,
                          OFX::BitDepthEnum tbd) {
    for (int y = win.y1; y < win.y2; ++y) {
      if (abort()) break;
      if (y < tb.y1 || y >= tb.y2) continue;
      const int rr = b.y2 - 1 - y;
      char* rowp = static_cast<char*>(base) + static_cast<size_t>(y - tb.y1) * rowBytes;
      for (int x = win.x1; x < win.x2; ++x) {
        if (x < tb.x1 || x >= tb.x2) continue;
        const int cc = x - b.x1;
        float m = 0.f;
        if (rr >= 0 && rr < H && cc >= 0 && cc < W) m = poolAt(rr, cc);
        const size_t off = static_cast<size_t>(x - tb.x1) * 4;
        const float o[4] = {m, m, m, m};
        if (tbd == OFX::eBitDepthFloat) {
          float* dp = reinterpret_cast<float*>(rowp) + off;
          for (int c = 0; c < 4; ++c) dp[c] = o[c];
        } else if (tbd == OFX::eBitDepthUShort) {
          unsigned short* dp = reinterpret_cast<unsigned short*>(rowp) + off;
          for (int c = 0; c < 4; ++c) dp[c] = ClampToDepth<unsigned short>(o[c], 65535.f);
        } else {
          unsigned char* dp = reinterpret_cast<unsigned char*>(rowp) + off;
          for (int c = 0; c < 4; ++c) dp[c] = ClampToDepth<unsigned char>(o[c], 255.f);
        }
      }
    }
  };

  const bool multiPlane = !g_mpRenderPlanes.empty();
  if (!multiPlane) {
    writeColour(dst.get());
    return true;
  }
  const FnOfxImageEffectPlaneSuiteV1* ps = hplane::PlaneSuite();
  const OfxPropertySuiteV1* props = hplane::PropSuite();
  for (const std::string& pl : g_mpRenderPlanes) {
    if (pl == kFnOfxImagePlaneColour) {
      // Default dst is RGBA: write straight into it. Otherwise fetch the Colour
      // plane explicitly and write the pooled matte into that raw buffer.
      if (dstIsRGBA) { writeColour(dst.get()); continue; }
      if (!ps || !props) continue;
      OfxPropertySetHandle cImgProps = nullptr;
      if (ps->clipGetImagePlane(_dstClip->getHandle(), args.time,
                                kFnOfxImagePlaneColour, nullptr, &cImgProps) != kOfxStatOK ||
          !cImgProps)
        continue;
      void* cbase = nullptr;
      props->propGetPointer(cImgProps, kOfxImagePropData, 0, &cbase);
      int cbnd[4] = {0, 0, 0, 0};
      props->propGetIntN(cImgProps, kOfxImagePropBounds, 4, cbnd);
      int cRowBytes = 0;
      props->propGetInt(cImgProps, kOfxImagePropRowBytes, 0, &cRowBytes);
      char* cDepthStr = nullptr;
      props->propGetString(cImgProps, kOfxImageEffectPropPixelDepth, 0, &cDepthStr);
      OFX::BitDepthEnum cbd = OFX::eBitDepthFloat;
      if (cDepthStr) {
        if (!std::strcmp(cDepthStr, kOfxBitDepthByte)) cbd = OFX::eBitDepthUByte;
        else if (!std::strcmp(cDepthStr, kOfxBitDepthShort)) cbd = OFX::eBitDepthUShort;
      }
      if (!cbase) continue;
      const OfxRectI ctb{cbnd[0], cbnd[1], cbnd[2], cbnd[3]};
      writePoolRaw(cbase, ctb, cRowBytes, cbd);
      continue;
    }
    const hplane::PlaneDef* def = hplane::PlaneForEncoded(pl, kMpPlanes, kMpPlaneCount);
    if (!def || !ps || !props) continue;
    OfxPropertySetHandle imgProps = nullptr;
    if (ps->clipGetImagePlane(_dstClip->getHandle(), args.time, pl.c_str(), nullptr,
                              &imgProps) != kOfxStatOK ||
        !imgProps)
      continue;
    void* base = nullptr;
    props->propGetPointer(imgProps, kOfxImagePropData, 0, &base);
    int bnd[4] = {0, 0, 0, 0};
    props->propGetIntN(imgProps, kOfxImagePropBounds, 4, bnd);
    int rowBytes = 0;
    props->propGetInt(imgProps, kOfxImagePropRowBytes, 0, &rowBytes);
    char* depthStr = nullptr;
    props->propGetString(imgProps, kOfxImageEffectPropPixelDepth, 0, &depthStr);
    OFX::BitDepthEnum tbd = OFX::eBitDepthFloat;
    if (depthStr) {
      if (!std::strcmp(depthStr, kOfxBitDepthByte)) tbd = OFX::eBitDepthUByte;
      else if (!std::strcmp(depthStr, kOfxBitDepthShort)) tbd = OFX::eBitDepthUShort;
    }
    if (!base) continue;
    const OfxRectI tb{bnd[0], bnd[1], bnd[2], bnd[3]};
    writeCryptoRaw(base, tb, rowBytes, tbd, def->aov);
  }
  return true;
}

// Fill the output colour image with black (safety fallback on any failure).
void MattePartitionPlugin::renderBlack(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) return;
  const OFX::BitDepthEnum tbd = dst->getPixelDepth();
  const OfxRectI& win = args.renderWindow;
  for (int y = win.y1; y < win.y2; ++y) {
    for (int x = win.x1; x < win.x2; ++x) {
      void* praw = dst->getPixelAddress(x, y);
      if (!praw) continue;
      if (tbd == OFX::eBitDepthFloat) {
        float* dp = static_cast<float*>(praw);
        dp[0] = dp[1] = dp[2] = dp[3] = 0.f;
      } else if (tbd == OFX::eBitDepthUShort) {
        unsigned short* dp = static_cast<unsigned short*>(praw);
        dp[0] = dp[1] = dp[2] = dp[3] = 0;
      } else {
        unsigned char* dp = static_cast<unsigned char*>(praw);
        dp[0] = dp[1] = dp[2] = dp[3] = 0;
      }
    }
  }
}

void MattePartitionPlugin::render(const OFX::RenderArguments& args) {
  try {
    if (renderPartition(args)) return;
  } catch (...) {
    // Fall through to the black fallback; never crash the host.
  }
  renderBlack(args);
}

////////////////////////////////////////////////////////////////////////////////

namespace {

// Delegating main entry: intercept the Foundry multi-plane actions, then hand off
// to the Support library's dispatch with THIS plugin's UID.
OfxStatus MpMainEntry(const char* action, const void* handle,
                      OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  if (std::strcmp(action, kFnOfxImageEffectActionGetClipComponents) == 0)
    return MpGetClipComponents(outArgs);
  if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
    hplane::StashRenderPlanes(inArgs, g_mpRenderPlanes);
  return OFX::Private::mainEntryStr(action, handle, inArgs, outArgs, g_mpUid.c_str());
}

}  // namespace

// Explicit factory (instead of mDeclarePluginFactory) so we can override
// getMainEntry() to install the multi-plane action interceptor, exactly as the
// SAM 3D Body plugin does.
class MattePartitionFactory
    : public OFX::PluginFactoryHelper<MattePartitionFactory> {
 public:
  MattePartitionFactory(const std::string& id, unsigned int verMaj,
                        unsigned int verMin)
      : OFX::PluginFactoryHelper<MattePartitionFactory>(id, verMaj, verMin) {}
  void load() override {}
  void unload() override {}
  void describe(OFX::ImageEffectDescriptor& desc) override;
  void describeInContext(OFX::ImageEffectDescriptor& desc,
                         OFX::ContextEnum context) override;
  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle,
                                   OFX::ContextEnum context) override {
    return new MattePartitionPlugin(handle);
  }
  OfxPluginEntryPoint* getMainEntry() override { return MpMainEntry; }
};

using namespace OFX;

void MattePartitionFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kMpPluginName, kMpPluginName, kMpPluginName);
  desc.setPluginGrouping(kMpPluginGrouping);
  desc.setPluginDescription(kMpPluginDescription);

  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  desc.setSupportsTiles(false);           // Voronoi needs the whole frame
  desc.setTemporalClipAccess(false);      // stateless per-frame
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);

  // Opt into the Foundry/Nuke + Natron multi-plane extension so hosts call
  // GetClipComponents and route the Cryptomatte planes to/from this node.
  desc.getPropertySet().propSetInt(kFnOfxImageEffectPropMultiPlanar, 1, false);
}

void MattePartitionFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                              OFX::ContextEnum /*context*/) {
  // Cryptomatte input (multi-planar source: the upstream Hastur Cryptomatte).
  ClipDescriptor* crypto = desc.defineClip(kClipCryptomatte);
  crypto->addSupportedComponent(ePixelComponentRGBA);
  crypto->setTemporalClipAccess(false);
  crypto->setSupportsTiles(false);
  crypto->setIsMask(false);

  // Pool input (the pooled soft alpha; we read its A channel or luminance).
  ClipDescriptor* pool = desc.defineClip(kClipPool);
  pool->addSupportedComponent(ePixelComponentRGBA);
  pool->setTemporalClipAccess(false);
  pool->setSupportsTiles(false);
  pool->setIsMask(false);
  pool->setOptional(true);

  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");

  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamMaxDist);
    p->setLabels("Max distance", "Max dist", "Voronoi clamp (px)");
    p->setHint("Voronoi clamp radius in pixels. Pooled-alpha pixels farther than "
               "this from any owned Cryptomatte seed are dropped (kills the "
               "ex-player phantom halo left when a tracked person has exited). "
               "<= 0 disables the clamp.");
    p->setRange(0.0, 100000.0);
    p->setDisplayRange(0.0, 512.0);
    p->setDefault(64.0);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamSeedThresh);
    p->setLabels("Seed threshold", "Seed thresh", "Seed coverage threshold");
    p->setHint("Minimum front-most Cryptomatte coverage for a pixel to count as an "
               "owned seed when scanning which person IDs are present. (The "
               "partition core currently bakes its own internal seed threshold at "
               "0.5.)");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.5);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamCryptoLevels);
    p->setLabels("Crypto levels", "Levels", "Cryptomatte layers");
    p->setHint("Number of Cryptomatte layers (rank pairs) to pack in the output. "
               "2 layers = ranks 0..3 (CryptoObject00/01), the Psyop default.");
    p->setRange(1, 3);
    p->setDisplayRange(1, 3);
    p->setDefault(2);
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kMpParamCryptoManifest);
    p->setLabels("Cryptomatte manifest", "Manifest", "Cryptomatte manifest");
    p->setHint("Read-only. The candidate JSON name->hash map for the person_NN IDs "
               "(person_00..person_255). The output only ever emits a subset of "
               "these; a downstream Cryptomatte decoder / EXR Write can attach it.");
    p->setStringType(eStringTypeMultiLine);
    p->setEnabled(false);
    p->setEvaluateOnChange(false);
    p->setDefault(BuildCandidateManifest(kMpManifestPeople));
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kParamCryptoExrPath);
    p->setLabels("Cryptomatte EXR out", "Crypto EXR", "Cryptomatte EXR out");
    p->setHint("Optional printf-style output path for the node's OWN Cryptomatte "
               "EXR, e.g. '/renders/crypto_%04d.exr' (%0Nd is expanded with the "
               "frame number). When set, the node writes a decodable, self-contained "
               "Cryptomatte EXR itself — color = the pooled matte, hastur.CryptoObject "
               "planes = the partitioned layers, plus the Psyop cryptomatte/<key>/* "
               "metadata — bypassing the host's multi-plane Write (which HANGS on "
               "Natron 2.5). Empty = disabled (zero cost).");
    p->setStringType(eStringTypeFilePath);
    p->setDefault("");
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamCryptoCompression);
    p->setLabels("EXR compression", "Compression", "Cryptomatte EXR compression");
    p->setHint("Compression for the node-written Cryptomatte EXR. All options are "
               "LOSSLESS: Cryptomatte ids are exact float32 bit patterns and ANY "
               "lossy codec (e.g. DWAB) corrupts them. Writing the display RGB at a "
               "lossy DWAB while keeping the crypto lossless needs a MULTI-PART EXR "
               "(the OpenEXR/Imf follow-up); this single-part writer keeps the whole "
               "file lossless.");
    p->appendOption("ZIP (default)");   // 0
    p->appendOption("ZIPS");            // 1
    p->appendOption("PIZ");             // 2
    p->appendOption("RLE");             // 3
    p->appendOption("None");            // 4
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kParamSkeletonSidecar);
    p->setLabels("Skeleton sidecar dir", "Skeleton dir", "Skeleton sidecar dir");
    p->setHint("Optional directory holding Hastur's per-frame skeleton JSON "
               "sidecars ('skeleton_<frame:04d>.json', written by the SAM 3D Body "
               "node). When set alongside a Cryptomatte EXR out path, the frame's "
               "skeleton JSON is embedded in the SAME EXR as the string attribute "
               "'hastur/skeleton', so matte + identity + pose ride in one file. "
               "Absent sidecar for a frame = skipped silently.");
    p->setStringType(eStringTypeDirectoryPath);
    p->setDefault("");
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kMpParamPickName);
    p->setLabels("Pick person", "Pick", "Pick Cryptomatte name");
    p->setHint("PICK MODE -- a per-person seed selector that decodes coverage out of "
               "the input Cryptomatte into the Colour (RGBA) output, for a minimal "
               "downstream matting node (e.g. Kthanid) to seed from. Three values: "
               "empty = normal Voronoi partition (Pool = a refined union); a name "
               "(e.g. person_00) = isolate ONLY that person's coverage (one matting "
               "pass per id; highest fidelity, highest cost); '*' = the UNION of all "
               "ids' coverage (one matting pass for the whole frame, then a second "
               "Matte Partition in empty mode Voronoi-splits the refined union -- "
               "cheapest). On Natron 2.5 only the Colour plane flows node-to-node "
               "(its Color.RGBA-only limit), which is exactly the seed; on Nuke the "
               "output also carries a valid Cryptomatte.");
    p->setDefault("");
    page->addChild(*p);
  }
}

// Appended to the bundle's plugin list from OFX::Plugin::getPluginIDs so the
// bundle exports TWO plugins.
namespace hasturreg {
void AppendMattePartition(OFX::PluginFactoryArray& ids) {
  static MattePartitionFactory f(kMpPluginIdentifier, kMpPluginVersionMajor,
                                 kMpPluginVersionMinor);
  g_mpUid = f.getUID();  // the delegating main entry passes this to mainEntryStr
  ids.push_back(&f);
}
}  // namespace hasturreg
