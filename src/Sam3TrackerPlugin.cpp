// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// SAM 3 Tracker OpenFX plugin — the THIRD plugin in the Hastur bundle (Phase
// 5e-ii). It wraps the integrated Sam3MultiTracker engine (src/Sam3MultiTracker.h)
// in an OFX node whose PRODUCT is the external-tracks sidecar Hastur's SAM 3D Body
// node consumes downstream (src/ExternalTracks.h) — replacing the Python pre-pass.
//
// Data flow:
//
//   Source (whole image sequence, TEMPORAL clip access)
//        |
//        |  render() FIRST call -> whole-clip pre-pass (blocking, progress-visible):
//        |    per frame: BeginFrame -> Detect -> associate -> Seed/Propagate
//        |    forward pass (seed -> end), then backward pass (seed -> start)
//        |    (mirrors skylab's validate_bidir.cpp bidirectional driver)
//        v
//   tracksOutputDir/tracks.txt + masks/<frame:04d>_<id:02d>.msk   <- the product
//        |
//   Output  <- Source passed through UNCHANGED (the node emits no new image)
//
// This node is NOT multi-plane. It is stateless per-frame apart from a done-flag
// guard: it caches its whole-clip result (writes the sidecar once) and recomputes
// only when a param / clip / frame-range change invalidates the signature.
//
// Compile-only in this phase (Phase 5e-ii): a real host smoke against the ONNX
// graphs is the separate Phase 5e-ii-b (on the CUDA render host). On ANY failure
// the node degrades gracefully — it always passes Source through and never crashes
// the host.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsProcessing.H"

#include "Register.h"

#if HASTUR_WITH_ONNX
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "OrtSessionManager.h"  // hastur::Ep, hastur::ComputeUnits
#include "Sam3MultiTracker.h"
#endif

#define kT3PluginName "SAM 3 Tracker"
#define kT3PluginGrouping "Tokgan"
#define kT3PluginDescription                                                   \
  "Run the SAM 3 video tracker over the WHOLE clip and write the external-"     \
  "tracks sidecar (tracks.txt + per-frame .msk masks) that Hastur's SAM 3D "    \
  "Body node consumes downstream. Detect + associate + bidirectional propagate " \
  "from a mid-clip max-visibility seed frame; Output passes Source through "     \
  "unchanged (the node's product is the sidecar file, not a new image)."
#define kT3PluginIdentifier "com.tokgan.Sam3Tracker"
#define kT3PluginVersionMajor 0
#define kT3PluginVersionMinor 1

// Param names.
#define kT3ParamModelDir "modelDir"
#define kT3ParamTracksOutputDir "tracksOutputDir"
#define kT3ParamSeedFrame "seedFrame"
#define kT3ParamBidirectional "bidirectional"
#define kT3ParamScoreThresh "scoreThresh"
#define kT3ParamNmsIou "nmsIou"
#define kT3ParamComputeUnits "computeUnits"

////////////////////////////////////////////////////////////////////////////////
// Passthrough copy processor (Source -> Output, verbatim). Mirrors HasturPlugin's
// CopyProcessor; the node emits no new image, so render always copies Source.
namespace {

class T3CopyBase : public OFX::ImageProcessor {
 protected:
  OFX::Image* _srcImg;
 public:
  explicit T3CopyBase(OFX::ImageEffect& e) : OFX::ImageProcessor(e), _srcImg(nullptr) {}
  void setSrcImg(OFX::Image* v) { _srcImg = v; }
};

template <class PIX, int nComponents>
class T3CopyProcessor : public T3CopyBase {
 public:
  explicit T3CopyProcessor(OFX::ImageEffect& e) : T3CopyBase(e) {}
  void multiThreadProcessImages(OfxRectI w) override {
    for (int y = w.y1; y < w.y2; ++y) {
      if (_effect.abort()) break;
      PIX* dst = static_cast<PIX*>(_dstImg->getPixelAddress(w.x1, y));
      if (!dst) continue;
      for (int x = w.x1; x < w.x2; ++x) {
        const PIX* src =
            static_cast<const PIX*>(_srcImg ? _srcImg->getPixelAddress(x, y) : nullptr);
        if (src)
          for (int c = 0; c < nComponents; ++c) dst[c] = src[c];
        else
          for (int c = 0; c < nComponents; ++c) dst[c] = PIX(0);
        dst += nComponents;
      }
    }
  }
};

#if HASTUR_WITH_ONNX

// Directory holding this .ofx binary (Contents/<arch>), resolved from a symbol
// inside the loaded module — same approach as HasturPlugin's PluginBinaryDir.
std::string T3PluginBinaryDir() {
#ifdef _WIN32
  HMODULE h = nullptr;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&T3PluginBinaryDir), &h);
  char buf[MAX_PATH] = {0};
  GetModuleFileNameA(h, buf, MAX_PATH);
  std::string p(buf);
  size_t slash = p.find_last_of("\\/");
  return slash == std::string::npos ? std::string() : p.substr(0, slash);
#else
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&T3PluginBinaryDir), &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
  }
  return {};
#endif
}

// The bundle's sibling Contents/Resources dir (where runtime model assets ship).
std::string T3BundleResourceDir() {
  std::string bin = T3PluginBinaryDir();
  if (bin.empty()) return {};
  size_t slash = bin.find_last_of("\\/");
  if (slash == std::string::npos) return {};
  return (std::filesystem::path(bin.substr(0, slash)) / "Resources").string();
}

// Resolve the directory that holds the 5 ONNX graphs (G1..G5) + the .npy
// constants. Candidates, in order: the param dir, each $HASTUR_MODEL_DIR entry,
// then the bundle Resources dir. The first that actually contains G1.onnx wins;
// otherwise the param dir (or the first non-empty candidate) is returned so the
// engine constructor throws a clear "cannot open" and we fall back to passthrough.
std::string T3ResolveModelDir(const std::string& param_dir) {
  namespace fs = std::filesystem;
#ifdef _WIN32
  const char kSep = ';';
#else
  const char kSep = ':';
#endif
  std::vector<std::string> cands;
  if (!param_dir.empty()) cands.push_back(param_dir);
  if (const char* env = std::getenv("HASTUR_MODEL_DIR")) {
    std::string s(env);
    size_t i = 0;
    while (i <= s.size()) {
      size_t j = s.find(kSep, i);
      if (j == std::string::npos) j = s.size();
      if (j > i) cands.push_back(s.substr(i, j - i));
      i = j + 1;
    }
  }
  const std::string res = T3BundleResourceDir();
  if (!res.empty()) cands.push_back(res);

  std::error_code ec;
  for (const std::string& d : cands)
    if (fs::exists(fs::path(d) / "G1.onnx", ec)) return d;
  if (!cands.empty()) return cands.front();
  return param_dir;
}

// Build the engine's MultiPaths (absolute file paths) from a resolved model dir.
// File names match the exported SAM 3 asset set the engine + validate driver use.
hastur::MultiPaths T3BuildPaths(const std::string& dir) {
  namespace fs = std::filesystem;
  const fs::path d(dir);
  hastur::MultiPaths p;
  p.g1 = (d / "G1.onnx").string();
  p.g2 = (d / "G2.onnx").string();
  p.g3 = (d / "G3.onnx").string();
  p.g4 = (d / "G4.onnx").string();
  p.g5 = (d / "G5.onnx").string();
  p.tpos = (d / "const_maskmem_tpos_enc.npy").string();
  p.no_mem = (d / "const_no_mem_embed.npy").string();
  p.proj_w = (d / "const_objptr_tpos_proj_w.npy").string();
  p.proj_b = (d / "const_objptr_tpos_proj_b.npy").string();
  p.lang_feats = (d / "const_lang_feats_person.npy").string();
  p.lang_mask = (d / "const_lang_mask_person.npy").string();
  return p;
}

hastur::ComputeUnits T3ChoiceToUnits(int choice) {
  switch (choice) {
    case 0: return hastur::ComputeUnits::All;
    case 1: return hastur::ComputeUnits::CpuAndGpu;
    case 2: return hastur::ComputeUnits::CpuAndAne;
    case 3: return hastur::ComputeUnits::CpuOnly;
    default: return hastur::ComputeUnits::All;
  }
}

// Read one interleaved pixel to normalized float ([0,1] for int depths).
template <class PIX>
inline void T3ReadPix(const PIX* p, int nc, float scale, float& r, float& g, float& b) {
  r = p[0] * scale;
  g = nc > 1 ? p[1] * scale : r;
  b = nc > 2 ? p[2] * scale : r;
}

// Fill a top-down (row 0 = top) HWC RGB float buffer from an OFX image over its
// bounds. OFX pixels are bottom-up, so top row r maps to OFX y = y2-1-r. (Same
// orientation convention as HasturPlugin::ReadFrameRGBA and the sidecar format:
// the produced masks/boxes are therefore top-down, as ExternalTracks requires.)
template <class PIX>
void T3ReadFrameRGB(OFX::Image* img, const OfxRectI& b, int W, int H, float scale,
                    std::vector<float>& rgb) {
  const int nc = img->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
  for (int r = 0; r < H; ++r) {
    const int oy = b.y2 - 1 - r;
    for (int c = 0; c < W; ++c) {
      const int ox = b.x1 + c;
      const PIX* p = static_cast<const PIX*>(img->getPixelAddress(ox, oy));
      float* d = &rgb[(static_cast<size_t>(r) * W + c) * 3];
      if (p) T3ReadPix<PIX>(p, nc, scale, d[0], d[1], d[2]);
      else { d[0] = d[1] = d[2] = 0.f; }
    }
  }
}

// The engine's input geometry + normalization.
constexpr int kIn = hastur::kMImgSize;   // 1008
constexpr int kLR = hastur::kMLowRes;    // 288

// Bilinear-resample a top-down W*H*3 RGB frame to a planar CHW 3*1008*1008 buffer,
// normalized (mean/std 0.5, i.e. 2*v-1) as the SAM 3 engine expects.
void T3ToPlanarCHW(const std::vector<float>& rgb, int W, int H,
                   std::vector<float>& chw) {
  chw.assign(static_cast<size_t>(3) * kIn * kIn, 0.f);
  const size_t plane = static_cast<size_t>(kIn) * kIn;
  for (int ty = 0; ty < kIn; ++ty) {
    float gy = (ty + 0.5f) * (static_cast<float>(H) / kIn) - 0.5f;
    if (gy < 0.f) gy = 0.f;
    if (gy > H - 1.f) gy = H - 1.f;
    const int y0 = static_cast<int>(gy);
    const int y1 = std::min(y0 + 1, H - 1);
    const float wy = gy - y0;
    for (int tx = 0; tx < kIn; ++tx) {
      float gx = (tx + 0.5f) * (static_cast<float>(W) / kIn) - 0.5f;
      if (gx < 0.f) gx = 0.f;
      if (gx > W - 1.f) gx = W - 1.f;
      const int x0 = static_cast<int>(gx);
      const int x1 = std::min(x0 + 1, W - 1);
      const float wx = gx - x0;
      const size_t o = static_cast<size_t>(ty) * kIn + tx;
      for (int c = 0; c < 3; ++c) {
        const float v00 = rgb[(static_cast<size_t>(y0) * W + x0) * 3 + c];
        const float v01 = rgb[(static_cast<size_t>(y0) * W + x1) * 3 + c];
        const float v10 = rgb[(static_cast<size_t>(y1) * W + x0) * 3 + c];
        const float v11 = rgb[(static_cast<size_t>(y1) * W + x1) * 3 + c];
        const float top = v00 + (v01 - v00) * wx;
        const float bot = v10 + (v11 - v10) * wx;
        const float v = top + (bot - top) * wy;
        chw[c * plane + o] = 2.f * v - 1.f;  // mean/std 0.5 normalization
      }
    }
  }
}

// IoU of a binary detection mask (288) against a track's low-res logits (>0).
// Ported verbatim from validate_bidir.cpp.
double T3IouBinLogit(const std::vector<uint8_t>& det, const std::vector<float>& trk) {
  long inter = 0, uni = 0;
  for (int i = 0; i < kLR * kLR; ++i) {
    int a = det[i], bb = (i < static_cast<int>(trk.size()) && trk[i] > 0.f) ? 1 : 0;
    inter += (a & bb);
    uni += (a | bb);
  }
  return uni ? static_cast<double>(inter) / uni : 0.0;
}

#endif  // HASTUR_WITH_ONNX

}  // namespace

////////////////////////////////////////////////////////////////////////////////

class Sam3TrackerPlugin : public OFX::ImageEffect {
 public:
  explicit Sam3TrackerPlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _modelDir = fetchStringParam(kT3ParamModelDir);
    _tracksOutputDir = fetchStringParam(kT3ParamTracksOutputDir);
    _seedFrame = fetchIntParam(kT3ParamSeedFrame);
    _bidirectional = fetchBooleanParam(kT3ParamBidirectional);
    _scoreThresh = fetchDoubleParam(kT3ParamScoreThresh);
    _nmsIou = fetchDoubleParam(kT3ParamNmsIou);
    _computeUnits = fetchChoiceParam(kT3ParamComputeUnits);
  }

  void render(const OFX::RenderArguments& args) override;

 private:
  void renderPassthrough(const OFX::RenderArguments& args);
#if HASTUR_WITH_ONNX
  // The whole-clip pre-pass; writes the sidecar. Returns true on success (or when
  // there was genuinely nothing to do). Guarded by _prepassMx + _done.
  bool runPrepass(const OFX::RenderArguments& args);
  std::string PrepassSignature() const;
#endif

  OFX::Clip* _dstClip = nullptr;
  OFX::Clip* _srcClip = nullptr;
  OFX::StringParam* _modelDir = nullptr;
  OFX::StringParam* _tracksOutputDir = nullptr;
  OFX::IntParam* _seedFrame = nullptr;
  OFX::BooleanParam* _bidirectional = nullptr;
  OFX::DoubleParam* _scoreThresh = nullptr;
  OFX::DoubleParam* _nmsIou = nullptr;
  OFX::ChoiceParam* _computeUnits = nullptr;

  std::mutex _prepassMx;
  bool _done = false;
  std::string _sig;
};

////////////////////////////////////////////////////////////////////////////////

void Sam3TrackerPlugin::renderPassthrough(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);
  const OFX::BitDepthEnum bd = dst->getPixelDepth();
  const OFX::PixelComponentEnum comp = dst->getPixelComponents();
  std::unique_ptr<OFX::Image> src(
      _srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);

  auto run = [&](T3CopyBase& p) {
    p.setSrcImg(src.get());
    p.setDstImg(dst.get());
    p.setRenderWindow(args.renderWindow);
    p.process();
  };
  if (comp == OFX::ePixelComponentRGBA) {
    if (bd == OFX::eBitDepthFloat) { T3CopyProcessor<float, 4> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { T3CopyProcessor<unsigned short, 4> p(*this); run(p); }
    else { T3CopyProcessor<unsigned char, 4> p(*this); run(p); }
  } else {
    if (bd == OFX::eBitDepthFloat) { T3CopyProcessor<float, 1> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { T3CopyProcessor<unsigned short, 1> p(*this); run(p); }
    else { T3CopyProcessor<unsigned char, 1> p(*this); run(p); }
  }
}

#if HASTUR_WITH_ONNX

std::string Sam3TrackerPlugin::PrepassSignature() const {
  std::string modelDir, tracksOut;
  _modelDir->getValue(modelDir);
  _tracksOutputDir->getValue(tracksOut);
  int seed = -1; _seedFrame->getValue(seed);
  bool bidir = true; _bidirectional->getValue(bidir);
  double score = 0.5, nms = 0.1; _scoreThresh->getValue(score); _nmsIou->getValue(nms);
  int cu = 0; _computeUnits->getValue(cu);
  OfxRangeD fr = _srcClip ? _srcClip->getFrameRange() : OfxRangeD{0, 0};
  char buf[256];
  std::snprintf(buf, sizeof(buf), "|%d|%d|%.4f|%.4f|%d|%.3f..%.3f", seed,
                static_cast<int>(bidir), score, nms, cu, fr.min, fr.max);
  return modelDir + "|" + tracksOut + buf;
}

bool Sam3TrackerPlugin::runPrepass(const OFX::RenderArguments& args) {
  namespace fs = std::filesystem;
  using hastur::Sam3MultiTracker;

  // ---- guard: run once per (params/clip/frame-range) signature ----
  const std::string sig = PrepassSignature();
  std::lock_guard<std::mutex> lk(_prepassMx);
  if (_done && sig == _sig) return true;

  if (!_srcClip || !_srcClip->isConnected()) return false;

  // ---- params ----
  std::string modelDirParam, tracksOut;
  _modelDir->getValue(modelDirParam);
  _tracksOutputDir->getValue(tracksOut);
  int seedParam = -1; _seedFrame->getValue(seedParam);
  bool bidir = true; _bidirectional->getValue(bidir);
  double dScore = 0.5, dNms = 0.1;
  _scoreThresh->getValue(dScore); _nmsIou->getValue(dNms);
  const float scoreThresh = static_cast<float>(dScore);
  const float nmsIou = static_cast<float>(dNms);
  int cu = 0; _computeUnits->getValue(cu);
  if (tracksOut.empty()) {
    std::fprintf(stderr, "[Sam3Tracker] tracksOutputDir is empty; nothing to write\n");
    return false;
  }

  // ---- clip frame range (kOfxImageEffectPropFrameRange via the Source clip) ----
  OfxRangeD fr = _srcClip->getFrameRange();
  long t0 = std::lround(fr.min), t1 = std::lround(fr.max);
  if (t1 < t0) { double a, b; timeLineGetBounds(a, b); t0 = std::lround(a); t1 = std::lround(b); }
  if (t1 < t0) return false;
  const int N = static_cast<int>(t1 - t0 + 1);

  // ---- lazily fetch + convert a frame (idx 0..N-1 -> time t0+idx) to planar CHW.
  // Cached for the whole pre-pass so each frame is fetched at most once even though
  // the scan / forward / backward passes each revisit frames. plateW/plateH are
  // captured from the first successful fetch (all frames assumed same size). ----
  std::unordered_map<int, std::vector<float>> cache;
  int plateW = 0, plateH = 0;
  auto getFrame = [&](int idx) -> const std::vector<float>* {
    auto it = cache.find(idx);
    if (it != cache.end()) return it->second.empty() ? nullptr : &it->second;
    std::vector<float>& slot = cache[idx];  // inserts empty
    std::unique_ptr<OFX::Image> img(_srcClip->fetchImage(static_cast<double>(t0 + idx)));
    if (!img.get()) return nullptr;
    const OfxRectI b = img->getBounds();
    const int W = b.x2 - b.x1, H = b.y2 - b.y1;
    if (W <= 0 || H <= 0) return nullptr;
    if (plateW == 0) { plateW = W; plateH = H; }
    std::vector<float> rgb(static_cast<size_t>(W) * H * 3, 0.f);
    const OFX::BitDepthEnum bd = img->getPixelDepth();
    if (bd == OFX::eBitDepthFloat) T3ReadFrameRGB<float>(img.get(), b, W, H, 1.f, rgb);
    else if (bd == OFX::eBitDepthUShort) T3ReadFrameRGB<unsigned short>(img.get(), b, W, H, 1.f / 65535.f, rgb);
    else T3ReadFrameRGB<unsigned char>(img.get(), b, W, H, 1.f / 255.f, rgb);
    T3ToPlanarCHW(rgb, W, H, slot);
    return &slot;
  };

  // ---- build the engine (loads G1..G5 + constants). Throws on missing assets;
  // the caller catches and falls back to passthrough. ----
  hastur::Ep ep = hastur::Ep::Auto;
  hastur::MultiPaths paths = T3BuildPaths(T3ResolveModelDir(modelDirParam));
  Sam3MultiTracker eng(paths, ep, T3ChoiceToUnits(cu));
  std::fprintf(stderr,
               "[Sam3Tracker] pre-pass: frames %ld..%ld (N=%d) accel=%d "
               "score=%.3f nms=%.3f bidir=%d\n",
               t0, t1, N, static_cast<int>(eng.accelerator_active()), scoreThresh,
               nmsIou, static_cast<int>(bidir));

  // ---- determine the seed frame (idx space). -1 => auto max-visibility: the
  // frame whose detections cover the most area (a scan of BeginFrame+Detect). ----
  const int autoScan = (seedParam < 0) ? N : 0;
  int SEED = 0;
  if (seedParam >= 0) {
    SEED = static_cast<int>(std::lround(seedParam - t0));
    if (SEED < 0) SEED = 0;
    if (SEED > N - 1) SEED = N - 1;
  }

  // ---- progress: scan (optional) + seed + forward + backward ----
  const int forwardCount = (N - 1) - SEED;  // provisional; recomputed after auto seed
  int total = autoScan + 1 + std::max(0, N - 1) + (bidir ? N : 0);
  if (total < 1) total = 1;
  int stepDone = 0;
  progressStart("SAM 3 Tracker: tracking clip (whole-clip pre-pass)");
  auto tick = [&]() -> bool {
    ++stepDone;
    return progressUpdate(static_cast<double>(stepDone) / total);
  };

  if (autoScan) {
    double best = -1.0;
    for (int f = 0; f < N; ++f) {
      if (abort()) { progressEnd(); return false; }
      const std::vector<float>* chw = getFrame(f);
      if (chw) {
        eng.BeginFrame(chw->data(), f, N);
        auto dets = eng.Detect(scoreThresh, nmsIou);
        double vis = 0.0;
        for (auto& d : dets) vis += static_cast<double>(d.area);
        std::fprintf(stderr, "  scan %3d/%d: %zu dets, vis=%.0f\n", f + 1, N,
                     dets.size(), vis);
        if (vis > best) { best = vis; SEED = f; }
      }
      if (!tick()) { progressEnd(); return false; }
    }
    std::fprintf(stderr, "[Sam3Tracker] auto seed frame = %d (idx), plate=%ld\n",
                 SEED, t0 + SEED);
  }
  (void)forwardCount;

  // ---- driver state (mirrors validate_bidir.cpp) ----
  std::vector<Sam3MultiTracker::ObjState> tracks;
  int next_id = 0;
  const float NEW_DET = hastur::kNewDetThresh;   // 0.7
  const float ASSOC = hastur::kAssocIou;         // 0.1
  const float TRK_ASSOC = hastur::kTrkAssocIou;  // 0.5
  const int PRUNE_UNMATCHED = 8;

  // STREAM each frame's 1008² mask to disk the instant the frame is finalized (every
  // frame is recorded exactly once — seed, then forward, then backward passes). This
  // keeps the tracker's peak memory O(active tracks) instead of O(clip × tracks): the
  // old code buffered every frame's mask (~4 MB each) in RAM until the end — ~21 GB on
  // a 352-frame, 15-track clip — which is what capped resolution and clip length. We
  // now retain only a tiny per-mask bbox row (kIn space; scaled to plate px when
  // tracks.txt is written at the end). Directory must exist before the first record.
  const fs::path sdir(tracksOut);
  const fs::path mdir = sdir / "masks";
  { std::error_code ec; fs::create_directories(mdir, ec); }
  struct MaskRow { long frame; int id; int x0, y0, x1, y1; std::string rel; };
  std::vector<MaskRow> track_rows;
  auto write_mask = [&](long frame, int id, const std::vector<float>& hi) {
    if (static_cast<int>(hi.size()) < kIn * kIn) return;
    int x0 = kIn, y0 = kIn, x1 = -1, y1 = -1;
    std::vector<uint8_t> cov(static_cast<size_t>(kIn) * kIn, 0);
    for (int y = 0; y < kIn; ++y)
      for (int x = 0; x < kIn; ++x) {
        const size_t idx = static_cast<size_t>(y) * kIn + x;
        // last_high = G4's pred_masks_high_res, sigmoid-ed to [0,1]; write soft 0..255
        // coverage at full 1008² (Hastur maps uint8/255 -> coverage). bbox = 0.5 iso.
        float p = hi[idx];
        if (p < 0.f) p = 0.f; else if (p > 1.f) p = 1.f;
        cov[idx] = static_cast<uint8_t>(p * 255.f + 0.5f);
        if (p > 0.5f) { x0 = std::min(x0, x); y0 = std::min(y0, y);
                        x1 = std::max(x1, x); y1 = std::max(y1, y); }
      }
    if (x1 < 0) return;  // empty mask -> no file/row
    char rel[64];
    std::snprintf(rel, sizeof(rel), "masks/%04ld_%02d.msk", frame, id);
    std::ofstream mk(sdir / rel, std::ios::binary | std::ios::trunc);
    if (mk) {
      mk.write("MSK1", 4);
      int32_t mw = kIn, mh = kIn;
      mk.write(reinterpret_cast<const char*>(&mw), 4);
      mk.write(reinterpret_cast<const char*>(&mh), 4);
      mk.write(reinterpret_cast<const char*>(cov.data()),
               static_cast<std::streamsize>(cov.size()));
    }
    track_rows.push_back({frame, id, x0, y0, x1, y1, std::string(rel)});
  };
  auto record_frame = [&](int f) {
    const long plate = t0 + f;
    for (auto& t : tracks) {
      if (!t.active || t.last_area <= 0) continue;
      write_mask(plate, t.id, t.last_high);  // streams to disk; only a bbox row is kept
    }
  };

  // one detect->associate->spawn/prune step on the current (already-BeginFrame'd)
  // frame — ported verbatim from validate_bidir.cpp's process_frame.
  auto process_frame = [&](int f, bool reverse) {
    for (auto& t : tracks) if (t.active) eng.PropagateObj(t, reverse);
    auto dets = eng.Detect(scoreThresh, nmsIou);
    const int M = static_cast<int>(tracks.size());
    const int D = static_cast<int>(dets.size());
    std::vector<double> trk_best(M, 0.0), det_best(D, 0.0);
    for (int di = 0; di < D; ++di)
      for (int ti = 0; ti < M; ++ti) {
        if (!tracks[ti].active) continue;
        double j = T3IouBinLogit(dets[di].bin288, tracks[ti].last_low);
        if (j > det_best[di]) det_best[di] = j;
        if (j > trk_best[ti]) trk_best[ti] = j;
      }
    int pruned = 0;
    for (int ti = 0; ti < M; ++ti) {
      if (!tracks[ti].active) continue;
      if (trk_best[ti] >= TRK_ASSOC) tracks[ti].last_match = f;
      else if (std::abs(f - tracks[ti].last_match) >= PRUNE_UNMATCHED) {
        tracks[ti].active = false; ++pruned;
      }
    }
    int spawned = 0;
    for (int di = 0; di < D; ++di) {
      if (dets[di].score >= NEW_DET && det_best[di] < ASSOC) {
        Sam3MultiTracker::ObjState t; t.id = next_id++;
        eng.SeedObj(t, dets[di].soft1008.data());
        tracks.push_back(std::move(t));
        ++spawned;
      }
    }
    int nactive = 0; for (auto& t : tracks) if (t.active) ++nactive;
    std::fprintf(stderr, "  frame %3d (%s): %d dets, %d active (spawn=%d prune=%d)\n",
                 f, reverse ? "bwd" : "fwd", D, nactive, spawned, pruned);
  };

  // ---- seed all detections at the seed frame ----
  {
    const std::vector<float>* chw = getFrame(SEED);
    if (!chw) { progressEnd(); return false; }
    eng.BeginFrame(chw->data(), SEED, N);
    auto dets0 = eng.Detect(scoreThresh, nmsIou);
    std::fprintf(stderr, "[Sam3Tracker] seed frame %d: %zu dets -> seed\n", SEED,
                 dets0.size());
    for (auto& d : dets0) {
      Sam3MultiTracker::ObjState t; t.id = next_id++;
      eng.SeedObj(t, d.soft1008.data());
      tracks.push_back(std::move(t));
    }
    record_frame(SEED);
  }
  if (!tick()) { progressEnd(); return false; }

  // snapshot the seed-frame states for the backward pass (forward mutates tracks).
  std::vector<Sam3MultiTracker::ObjState> seed_snapshot = tracks;
  const int seed_next_id = next_id;

  // ---- FORWARD pass: seed+1 .. N-1 ----
  std::fprintf(stderr, "[Sam3Tracker] forward pass %d -> %d\n", SEED + 1, N - 1);
  for (int f = SEED + 1; f < N; ++f) {
    if (abort()) { progressEnd(); return false; }
    const std::vector<float>* chw = getFrame(f);
    if (chw) { eng.BeginFrame(chw->data(), f, N); process_frame(f, /*reverse=*/false); record_frame(f); }
    std::fprintf(stderr, "[Sam3Tracker] forward %d/%d\n", f - SEED, std::max(1, N - 1 - SEED));
    if (!tick()) { progressEnd(); return false; }
  }

  // ---- BACKWARD pass: restore snapshot, seed-1 .. 0 (only when bidirectional) ----
  if (bidir) {
    tracks = seed_snapshot;
    next_id = seed_next_id;  // backward-spawned objects continue the id counter
    std::fprintf(stderr, "[Sam3Tracker] backward pass %d -> %d\n", SEED - 1, 0);
    for (int f = SEED - 1; f >= 0; --f) {
      if (abort()) { progressEnd(); return false; }
      const std::vector<float>* chw = getFrame(f);
      if (chw) { eng.BeginFrame(chw->data(), f, N); process_frame(f, /*reverse=*/true); record_frame(f); }
      std::fprintf(stderr, "[Sam3Tracker] backward %d/%d\n", SEED - f, std::max(1, SEED));
      if (!tick()) { progressEnd(); return false; }
    }
  }

  progressUpdate(1.0);
  progressEnd();

  if (plateW == 0 || plateH == 0) {
    std::fprintf(stderr, "[Sam3Tracker] no frames fetched; sidecar not written\n");
    return false;
  }

  // ---- masks already streamed to disk during the passes; write tracks.txt ----
  // (sdir/mdir were created before the passes, above.) Distinct ids actually recorded,
  // sorted — the "# ids" header enumerates every track id so a full Cryptomatte manifest
  // can be baked downstream.
  std::vector<int> ids;
  for (auto& r : track_rows) ids.push_back(r.id);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

  std::ofstream tt(sdir / "tracks.txt", std::ios::binary | std::ios::trunc);
  if (!tt) {
    std::fprintf(stderr, "[Sam3Tracker] cannot open %s/tracks.txt for write\n",
                 sdir.string().c_str());
    return false;
  }
  tt << "# hastur-tracks v1\n";
  tt << "# ids"; for (int id : ids) tt << " " << id; tt << "\n";
  tt << "# frame track_id x0 y0 x1 y1 mask_relpath\n";

  const float sx = static_cast<float>(plateW) / kIn;  // bbox mask-space (kIn) -> plate px
  const float sy = static_cast<float>(plateH) / kIn;
  for (auto& r : track_rows)
    tt << r.frame << " " << r.id << " " << (r.x0 * sx) << " " << (r.y0 * sy) << " "
       << ((r.x1 + 1) * sx) << " " << ((r.y1 + 1) * sy) << " " << r.rel << "\n";
  tt.flush();
  std::fprintf(stderr,
               "[Sam3Tracker] wrote %s/tracks.txt (%zu rows, %zu ids) + masks/\n",
               sdir.string().c_str(), track_rows.size(), ids.size());

  _done = true;
  _sig = sig;
  return true;
}

#endif  // HASTUR_WITH_ONNX

void Sam3TrackerPlugin::render(const OFX::RenderArguments& args) {
#if HASTUR_WITH_ONNX
  // FIRST render call (per signature) runs the whole-clip pre-pass that writes the
  // sidecar. Any failure degrades to a plain passthrough — never crash the host.
  try {
    runPrepass(args);
  } catch (const std::exception& e) {
    hasturreg::SafeSetMessage(*this, OFX::Message::eMessageError, "sam3TrackerPrepass",
                              std::string("SAM 3 Tracker pre-pass error: ") + e.what());
  } catch (...) {
    hasturreg::SafeSetMessage(*this, OFX::Message::eMessageError, "sam3TrackerPrepass",
                              "SAM 3 Tracker pre-pass error (unknown)");
  }
#endif
  // On EVERY render (including the first, after the pre-pass) pass Source through
  // to Output unchanged — the node's product is the sidecar, not a new image.
  renderPassthrough(args);
}

////////////////////////////////////////////////////////////////////////////////

// Explicit factory. This node is NOT multi-plane, so it uses the Support library's
// default per-factory main entry (no getMainEntry override / no action intercept).
class Sam3TrackerFactory : public OFX::PluginFactoryHelper<Sam3TrackerFactory> {
 public:
  Sam3TrackerFactory(const std::string& id, unsigned int verMaj, unsigned int verMin)
      : OFX::PluginFactoryHelper<Sam3TrackerFactory>(id, verMaj, verMin) {}
  void load() override {}
  void unload() override {}
  void describe(OFX::ImageEffectDescriptor& desc) override;
  void describeInContext(OFX::ImageEffectDescriptor& desc,
                         OFX::ContextEnum context) override;
  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle,
                                   OFX::ContextEnum /*context*/) override {
    return new Sam3TrackerPlugin(handle);
  }
};

using namespace OFX;

void Sam3TrackerFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kT3PluginName, kT3PluginName, kT3PluginName);
  desc.setPluginGrouping(kT3PluginGrouping);
  desc.setPluginDescription(kT3PluginDescription);

  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  // STATEFUL whole-clip pre-pass: forbid the host from cloning/parallelizing this
  // effect. The default (fully-safe) lets Natron 2.5 spawn per-frame render clones —
  // and since each clone re-runs the ENTIRE first-render pre-pass (SAM 3 whole-clip
  // tracking) that is a huge duplicated GPU cost + OOM risk. Unsafe = one render call
  // at a time on the ORIGINAL instance only, no clones (one pre-pass, one engine).
  desc.setRenderThreadSafety(OFX::eRenderUnsafe);
  desc.setSupportsMultiResolution(true);
  desc.setSupportsTiles(false);            // the pre-pass needs whole frames
  // The node reads the WHOLE clip in its render pre-pass: temporal clip access at
  // the effect (and, in describeInContext, the Source clip).
  desc.setTemporalClipAccess(true);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);
}

void Sam3TrackerFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                           OFX::ContextEnum /*context*/) {
  ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(ePixelComponentRGBA);
  src->setTemporalClipAccess(true);        // the node reads the whole clip
  src->setSupportsTiles(false);
  src->setIsMask(false);

  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");

  {
    StringParamDescriptor* p = desc.defineStringParam(kT3ParamModelDir);
    p->setLabels("Model dir", "Model dir", "Model directory");
    p->setHint("Directory holding the 5 SAM 3 tracker ONNX graphs (G1.onnx..G5.onnx) "
               "and the .npy constants (const_maskmem_tpos_enc, const_no_mem_embed, "
               "const_objptr_tpos_proj_w/b, const_lang_feats_person, "
               "const_lang_mask_person). Empty resolves from $HASTUR_MODEL_DIR then "
               "the bundle Contents/Resources.");
    p->setStringType(eStringTypeDirectoryPath);
    p->setDefault("");
    page->addChild(*p);
  }
  {
    StringParamDescriptor* p = desc.defineStringParam(kT3ParamTracksOutputDir);
    p->setLabels("Tracks output dir", "Tracks out", "Tracks output directory");
    p->setHint("Directory to WRITE the external-tracks sidecar into: tracks.txt "
               "(the Hastur external-tracks format) plus masks/<frame>_<id>.msk "
               "coverage. Point Hastur's SAM 3D Body 'Person tracks' param at this "
               "same directory to drive its pose from these boxes+ids.");
    p->setStringType(eStringTypeDirectoryPath);
    p->setDefault("");
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kT3ParamSeedFrame);
    p->setLabels("Seed frame", "Seed frame", "Seed frame");
    p->setHint("Frame to seed all tracks from before propagating both ways. -1 = "
               "auto: pick the max-visibility frame (largest total detection area). "
               "Given as a plate frame number.");
    p->setRange(-1, 1000000);
    p->setDisplayRange(-1, 1000);
    p->setDefault(-1);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kT3ParamBidirectional);
    p->setLabels("Bidirectional", "Bidirectional", "Bidirectional");
    p->setHint("Propagate from the seed frame BOTH forward (seed->end) and backward "
               "(seed->start). Off = forward only (frames before the seed get no "
               "tracks).");
    p->setDefault(true);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kT3ParamScoreThresh);
    p->setLabels("Score threshold", "Score thresh", "Detection score threshold");
    p->setHint("Per-frame detection score threshold (sigmoid*presence) fed to the "
               "SAM 3 detector head before mask-IoU NMS.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.5);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kT3ParamNmsIou);
    p->setLabels("NMS IoU", "NMS IoU", "Detection NMS IoU");
    p->setHint("Mask-IoU threshold for non-max suppression of per-frame detections.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.1);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kT3ParamComputeUnits);
    p->setLabels("Compute units", "Compute units", "Compute units");
    p->setHint("ONNX Runtime accelerator selection (CoreML MLComputeUnits on macOS; "
               "CUDA elsewhere).");
    p->appendOption("All");
    p->appendOption("CPU + GPU");
    p->appendOption("CPU + Neural Engine");
    p->appendOption("CPU only");
    p->setDefault(0);
    page->addChild(*p);
  }
}

// Appended to the bundle's plugin list from OFX::Plugin::getPluginIDs so the
// bundle exports a THIRD plugin (Sam3dBody + MattePartition + Sam3Tracker),
// i.e. OfxGetNumberOfPlugins() == 3. Defined out-of-line (declared in Register.h).
namespace hasturreg {
void AppendSam3Tracker(OFX::PluginFactoryArray& ids) {
  static Sam3TrackerFactory f(kT3PluginIdentifier, kT3PluginVersionMajor,
                              kT3PluginVersionMinor);
  ids.push_back(&f);
}
}  // namespace hasturreg
