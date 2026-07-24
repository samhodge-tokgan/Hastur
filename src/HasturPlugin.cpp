// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// SAM 3D Body OpenFX plugin (project "Hastur").
//
// Recovers a posed 3D human body mesh (MHR) from a single image and renders it
// back into the OFX stream. M4 wires the full BODY-ONLY pipeline:
//
//   person_detector.onnx -> per-person crop + camera conditioning ->
//   sam3dbody_body.onnx  -> pose_corrective.onnx -> C++ MHR mesh (mhr_assets.bin)
//   -> perspective camera solve -> software rasterizer -> depth-ordered composite
//
// (Hands are M7; the hand-presence gate is computed but the refiner is deferred.)
//
// The pipeline object is cached per (model dir + compute units) so it is not
// rebuilt every frame. On ANY failure the effect logs a user message and passes
// the source through unchanged — it must never crash the host.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.H"

#include "Register.h"

#if HASTUR_WITH_ONNX
#include "CameraMatrix.h"
#include "CameraSolver.h"
#include "Cryptomatte.h"
#include "OrtAccel.h"
#include "Sam3dBodyPipeline.h"
#include "nuke/fnOfxExtensions.h"
#endif

#define kPluginName "SAM 3D Body"
#define kPluginGrouping "Tokgan"
#define kPluginDescription \
  "Recover a posed 3D human body mesh (MHR) from a single image using SAM 3D " \
  "Body via ONNX Runtime with hardware acceleration."
#define kPluginIdentifier "com.tokgan.Sam3dBody"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 8

// Param names.
#define kParamModelDir "modelDir"
#define kParamComputeUnits "computeUnits"
#define kParamScoreThresh "scoreThresh"
#define kParamMaxPeople "maxPeople"
#define kParamGrey "greyLevel"
#define kParamGarment "garment"
#define kParamLeotardColor "leotardColor"
#define kParamSkinColor "skinColor"
#define kParamOverrideCam "overrideCamera"
#define kParamFocal "focalPx"
#define kParamFov "fovDeg"
#define kParamSupersample "supersample"
#define kParamPremult "premult"
#define kParamCompOverSrc "compositeOverSource"
// AOV output selection (portable single-plane fallback) + camera/manifest data.
#define kParamOutputAov "outputAov"
#define kParamBakeCamera "bakeCameraData"
#define kParamCamIntrinsics "camIntrinsics"
#define kParamCamWorldToNdc "camWorldToNDC"
#define kParamCamNdcToWorld "camNDCToWorld"
#define kParamCryptoManifest "cryptoManifest"
#define kParamCryptoCoverage "cryptoCoverage"

// cryptoCoverage choice order — MUST match hastur::CryptoCoverage enum.
enum { kCovMesh = 0, kCovSam3Mask = 1, kCovBoth = 2 };

// outputAov choice order (index -> pass). Beauty (0) is the classic RGBA render.
enum HasturAov {
  kAovBeauty = 0,
  kAovDepth,
  kAovPosition,
  kAovNormal,
  kAovPref,
  kAovST,
  kAovCrypto00,
  kAovCrypto01,
  // Appended after Crypto01 to preserve existing outputAov choice indices.
  kAovNref,
};

////////////////////////////////////////////////////////////////////////////////
// Passthrough copy processor (used on any pipeline failure).

class CopyBase : public OFX::ImageProcessor {
 protected:
  OFX::Image* _srcImg;
 public:
  explicit CopyBase(OFX::ImageEffect& e) : OFX::ImageProcessor(e), _srcImg(nullptr) {}
  void setSrcImg(OFX::Image* v) { _srcImg = v; }
};

template <class PIX, int nComponents>
class CopyProcessor : public CopyBase {
 public:
  explicit CopyProcessor(OFX::ImageEffect& e) : CopyBase(e) {}
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
namespace {

// Directory holding this .ofx binary (Contents/<arch>), resolved from the
// address of a symbol inside the loaded module.
std::string PluginBinaryDir() {
#ifdef _WIN32
  HMODULE h = nullptr;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&PluginBinaryDir), &h);
  char buf[MAX_PATH] = {0};
  GetModuleFileNameA(h, buf, MAX_PATH);
  std::string p(buf);
  size_t slash = p.find_last_of("\\/");
  return slash == std::string::npos ? std::string() : p.substr(0, slash);
#else
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&PluginBinaryDir), &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
  }
  return {};
#endif
}

// Bundle model search path: the .ofx lives in Contents/<arch>; models resolve
// from the sibling Contents/Resources.
std::string BundleResourceDir() {
  std::string bin = PluginBinaryDir();
  if (bin.empty()) return {};
#ifdef _WIN32
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  size_t slash = bin.find_last_of("\\/");
  if (slash == std::string::npos) return {};
  std::string contents = bin.substr(0, slash);  // .../Contents
  return contents + sep + "Resources";
}

// Build the colon-separated model search path: explicit param dir, then the
// bundle Resources dir. ($HASTUR_MODEL_DIR is appended by the pipeline itself.)
std::string ModelSearchPath(const std::string& param_dir) {
  std::string path = param_dir;
  std::string res = BundleResourceDir();
  if (!res.empty()) {
    if (!path.empty()) path += ':';
    path += res;
  }
  return path;
}

hastur::ComputeUnits ChoiceToUnits(int choice) {
  switch (choice) {
    case 0: return hastur::ComputeUnits::All;
    case 1: return hastur::ComputeUnits::CpuAndGpu;
    case 2: return hastur::ComputeUnits::CpuAndAne;
    case 3: return hastur::ComputeUnits::CpuOnly;
    default: return hastur::ComputeUnits::All;
  }
}

// ---------------------------------------------------------------------------
// Multi-plane (Foundry/Nuke + Natron) AOV output. Implemented directly against
// the vendored fnOfxExtensions.h constants + the raw plane/property suites, so
// the SAM-licensed plugin needs no GPL support code. See docs/AOVS.md.
//
// UNVERIFIED AGAINST A HOST: the wire protocol is source-accurate but has not yet
// been exercised in Nuke/Natron here (needs the gated models + a host). The
// portable outputAov path (single RGBA plane) is the tested fallback.
// ---------------------------------------------------------------------------
struct PlaneDef {
  int aov;                 // HasturAov
  const char* id;          // stable unique plane id
  const char* label;       // display label
  const char* chans[4];    // channel names
  int nch;                 // channel count
};
const PlaneDef kPlanes[] = {
    {kAovDepth, "hastur.depth", "Depth", {"Z", nullptr, nullptr, nullptr}, 1},
    {kAovPosition, "hastur.position", "Position", {"X", "Y", "Z", nullptr}, 3},
    {kAovNormal, "hastur.normal", "Normal", {"X", "Y", "Z", nullptr}, 3},
    {kAovPref, "hastur.pref", "Pref", {"X", "Y", "Z", nullptr}, 3},
    {kAovNref, "hastur.nref", "Nref", {"X", "Y", "Z", nullptr}, 3},
    {kAovST, "hastur.st", "ST", {"U", "V", nullptr, nullptr}, 2},
    {kAovCrypto00, "hastur.CryptoObject00", "CryptoObject00", {"R", "G", "B", "A"}, 4},
    {kAovCrypto01, "hastur.CryptoObject01", "CryptoObject01", {"R", "G", "B", "A"}, 4},
};

// Natron/Nuke multi-plane component-string encoding for one plane.
std::string EncodePlane(const PlaneDef& p) {
  std::string s = std::string(kNatronOfxImageComponentsPlaneName) + p.id +
                  kNatronOfxImageComponentsPlaneLabel + p.label;
  for (int i = 0; i < p.nch; ++i)
    s += std::string(kNatronOfxImageComponentsPlaneChannel) + p.chans[i];
  return s;
}

const PlaneDef* PlaneForEncoded(const std::string& enc) {
  for (const PlaneDef& p : kPlanes)
    if (EncodePlane(p) == enc) return &p;
  return nullptr;
}

const OfxPropertySuiteV1* PropSuite() {
  static const OfxPropertySuiteV1* s =
      static_cast<const OfxPropertySuiteV1*>(OFX::fetchSuite(kOfxPropertySuite, 1, true));
  return s;
}
const FnOfxImageEffectPlaneSuiteV1* PlaneSuite() {
  static const FnOfxImageEffectPlaneSuiteV1* s =
      static_cast<const FnOfxImageEffectPlaneSuiteV1*>(
          OFX::fetchSuite(kFnOfxImageEffectPlaneSuite, 1, true));
  return s;
}

// Set by getPluginIDs (before any action); the delegate main entry needs it.
std::string g_uid;
// Planes the host asked for this render call (stashed from the render inArgs).
std::vector<std::string> g_renderPlanes;

// GetClipComponents: declare colour + every AOV plane on the output clip, colour
// on the source clip, and pass everything else through from Source.
OfxStatus HasturGetClipComponents(OfxPropertySetHandle outArgs) {
  const OfxPropertySuiteV1* props = PropSuite();
  if (!props) return kOfxStatReplyDefault;
  std::string outName =
      std::string(kFnOfxImageEffectActionGetClipComponentsPropString) +
      kOfxImageEffectOutputClipName;
  int i = 0;
  props->propSetString(outArgs, outName.c_str(), i++, kFnOfxImagePlaneColour);
  for (const PlaneDef& p : kPlanes) {
    std::string enc = EncodePlane(p);
    props->propSetString(outArgs, outName.c_str(), i++, enc.c_str());
  }
  std::string inName =
      std::string(kFnOfxImageEffectActionGetClipComponentsPropString) +
      kOfxImageEffectSimpleSourceClipName;
  props->propSetString(outArgs, inName.c_str(), 0, kFnOfxImagePlaneColour);
  props->propSetString(outArgs, kFnOfxImageEffectPropPassThroughClip, 0,
                       kOfxImageEffectSimpleSourceClipName);
  return kOfxStatOK;
}

void StashRenderPlanes(OfxPropertySetHandle inArgs) {
  g_renderPlanes.clear();
  const OfxPropertySuiteV1* props = PropSuite();
  if (!props) return;
  int n = 0;
  if (props->propGetDimension(inArgs, kOfxImageEffectPropRenderPlanes, &n) !=
      kOfxStatOK)
    return;
  for (int i = 0; i < n; ++i) {
    char* v = nullptr;
    if (props->propGetString(inArgs, kOfxImageEffectPropRenderPlanes, i, &v) ==
            kOfxStatOK &&
        v)
      g_renderPlanes.emplace_back(v);
  }
}

// The plugin's OFX main entry: intercept the Foundry multi-plane actions, then
// delegate everything to the C++ Support library's dispatch (mainEntryStr).
OfxStatus HasturMainEntry(const char* action, const void* handle,
                          OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  if (std::strcmp(action, kFnOfxImageEffectActionGetClipComponents) == 0)
    return HasturGetClipComponents(outArgs);
  if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
    StashRenderPlanes(inArgs);
  return OFX::Private::mainEntryStr(action, handle, inArgs, outArgs, g_uid.c_str());
}

}  // namespace
#endif  // HASTUR_WITH_ONNX

////////////////////////////////////////////////////////////////////////////////

class Sam3dBodyPlugin : public OFX::ImageEffect {
 protected:
  OFX::Clip* _dstClip;
  OFX::Clip* _srcClip;

  OFX::StringParam* _modelDir = nullptr;
  OFX::ChoiceParam* _computeUnits = nullptr;
  OFX::DoubleParam* _scoreThresh = nullptr;
  OFX::IntParam* _maxPeople = nullptr;
  OFX::DoubleParam* _grey = nullptr;
  OFX::BooleanParam* _garment = nullptr;
  OFX::RGBParam* _leotardColor = nullptr;
  OFX::RGBParam* _skinColor = nullptr;
  OFX::BooleanParam* _overrideCam = nullptr;
  OFX::DoubleParam* _focal = nullptr;
  OFX::DoubleParam* _fov = nullptr;
  OFX::IntParam* _supersample = nullptr;
  OFX::BooleanParam* _premult = nullptr;
  OFX::BooleanParam* _compOverSrc = nullptr;
  OFX::ChoiceParam* _outputAov = nullptr;
  OFX::ChoiceParam* _cryptoCoverage = nullptr;
  OFX::PushButtonParam* _bakeCamera = nullptr;
  OFX::StringParam* _camIntrinsics = nullptr;
  OFX::StringParam* _camWorldToNdc = nullptr;
  OFX::StringParam* _camNdcToWorld = nullptr;
  OFX::StringParam* _cryptoManifest = nullptr;

#if HASTUR_WITH_ONNX
  std::unique_ptr<hastur::Sam3dBodyPipeline> _pipeline;
  std::string _pipelineKey;  // model dir + compute units (humbaba _engineKey)
  // NOTE: the FrameResult cache is process-global (see FrameCacheGet/Put in the
  // .cpp) so it is shared across node instances — e.g. the Nuke AOV gizmo.
#endif

 public:
  explicit Sam3dBodyPlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _modelDir = fetchStringParam(kParamModelDir);
    _computeUnits = fetchChoiceParam(kParamComputeUnits);
    _scoreThresh = fetchDoubleParam(kParamScoreThresh);
    _maxPeople = fetchIntParam(kParamMaxPeople);
    _grey = fetchDoubleParam(kParamGrey);
    _garment = fetchBooleanParam(kParamGarment);
    _leotardColor = fetchRGBParam(kParamLeotardColor);
    _skinColor = fetchRGBParam(kParamSkinColor);
    _overrideCam = fetchBooleanParam(kParamOverrideCam);
    _focal = fetchDoubleParam(kParamFocal);
    _fov = fetchDoubleParam(kParamFov);
    _supersample = fetchIntParam(kParamSupersample);
    _premult = fetchBooleanParam(kParamPremult);
    _compOverSrc = fetchBooleanParam(kParamCompOverSrc);
    _outputAov = fetchChoiceParam(kParamOutputAov);
    _cryptoCoverage = fetchChoiceParam(kParamCryptoCoverage);
    _bakeCamera = fetchPushButtonParam(kParamBakeCamera);
    _camIntrinsics = fetchStringParam(kParamCamIntrinsics);
    _camWorldToNdc = fetchStringParam(kParamCamWorldToNdc);
    _camNdcToWorld = fetchStringParam(kParamCamNdcToWorld);
    _cryptoManifest = fetchStringParam(kParamCryptoManifest);
  }

  void render(const OFX::RenderArguments& args) override;
  void changedParam(const OFX::InstanceChangedArgs& args,
                    const std::string& name) override;

 private:
  void renderPassthrough(const OFX::RenderArguments& args);
#if HASTUR_WITH_ONNX
  bool renderPipeline(const OFX::RenderArguments& args);
  void bakeCameraData(double time);
#endif
};

////////////////////////////////////////////////////////////////////////////////

void Sam3dBodyPlugin::renderPassthrough(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) OFX::throwSuiteStatusException(kOfxStatFailed);
  const OFX::BitDepthEnum bd = dst->getPixelDepth();
  const OFX::PixelComponentEnum comp = dst->getPixelComponents();
  std::unique_ptr<OFX::Image> src(_srcClip->fetchImage(args.time));

  auto run = [&](CopyBase& p) {
    p.setSrcImg(src.get());
    p.setDstImg(dst.get());
    p.setRenderWindow(args.renderWindow);
    p.process();
  };
  if (comp == OFX::ePixelComponentRGBA) {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 4> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 4> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 4> p(*this); run(p); }
  } else {
    if (bd == OFX::eBitDepthFloat) { CopyProcessor<float, 1> p(*this); run(p); }
    else if (bd == OFX::eBitDepthUShort) { CopyProcessor<unsigned short, 1> p(*this); run(p); }
    else { CopyProcessor<unsigned char, 1> p(*this); run(p); }
  }
}

#if HASTUR_WITH_ONNX
namespace {

// --- Process-global FrameResult cache --------------------------------------
// A whole FrameResult (beauty + all AOV buffers) is expensive to compute
// (inference). Multiple node instances that share the same input + geometry
// params — e.g. the 8 instances inside the Nuke AOV gizmo, one per plane —
// otherwise each re-run inference. This small LRU, keyed on a hash of the input
// pixels + the geometry params, lets the FIRST instance compute and the rest
// reuse the result (~1x inference instead of Nx). A per-key mutex makes it
// compute-once even under concurrent (multi-threaded) host rendering, while
// different keys (different frames/inputs) still compute in parallel.
std::mutex g_cacheMx;
std::deque<std::pair<std::string, std::shared_ptr<hastur::FrameResult>>> g_cache;
constexpr size_t kFrameCacheMax = 4;

std::mutex g_keyLocksMx;
std::unordered_map<std::string, std::shared_ptr<std::mutex>> g_keyLocks;

std::shared_ptr<hastur::FrameResult> FrameCacheGet(const std::string& key) {
  std::lock_guard<std::mutex> lk(g_cacheMx);
  for (auto it = g_cache.begin(); it != g_cache.end(); ++it) {
    if (it->first == key) {
      auto fr = it->second;
      auto e = *it;
      g_cache.erase(it);
      g_cache.push_front(e);  // most-recently-used to front
      return fr;
    }
  }
  return nullptr;
}
void FrameCachePut(const std::string& key,
                   const std::shared_ptr<hastur::FrameResult>& fr) {
  std::lock_guard<std::mutex> lk(g_cacheMx);
  for (auto& e : g_cache)
    if (e.first == key) { e.second = fr; return; }
  g_cache.emplace_front(key, fr);
  while (g_cache.size() > kFrameCacheMax) g_cache.pop_back();
}
std::shared_ptr<std::mutex> FrameCacheKeyLock(const std::string& key) {
  std::lock_guard<std::mutex> lk(g_keyLocksMx);
  auto& m = g_keyLocks[key];
  if (!m) m = std::make_shared<std::mutex>();
  return m;
}

// FNV-1a over the input RGB bytes — a cheap fingerprint (µs–ms) so two different
// plates at the same frame/time don't collide in the global cache.
uint64_t HashRgb(const std::vector<float>& rgb) {
  uint64_t h = 1469598103934665603ull;
  const auto* p = reinterpret_cast<const unsigned char*>(rgb.data());
  const size_t n = rgb.size() * sizeof(float);
  for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
  return h;
}

// Read one interleaved pixel to normalized float (0..1 for int depths).
template <class PIX>
inline void ReadPix(const PIX* p, int nc, float scale, float& r, float& g, float& b,
                    float& a) {
  r = p[0] * scale;
  g = nc > 1 ? p[1] * scale : r;
  b = nc > 2 ? p[2] * scale : r;
  a = nc > 3 ? p[3] * scale : 1.f;
}

// Fill a top-down (row 0 = top) HWC RGBA float buffer from an OFX image over its
// bounds. OFX pixels are bottom-up, so top row r maps to OFX y = y2-1-r.
template <class PIX>
void ReadFrameRGBA(OFX::Image* img, const OfxRectI& b, int W, int H, float scale,
                   std::vector<float>& out4) {
  const int nc = img->getPixelComponents() == OFX::ePixelComponentRGBA ? 4 : 1;
  for (int r = 0; r < H; ++r) {
    const int oy = b.y2 - 1 - r;
    for (int c = 0; c < W; ++c) {
      const int ox = b.x1 + c;
      const PIX* p = static_cast<const PIX*>(img->getPixelAddress(ox, oy));
      float* d = &out4[(static_cast<size_t>(r) * W + c) * 4];
      if (p) ReadPix<PIX>(p, nc, scale, d[0], d[1], d[2], d[3]);
      else { d[0] = d[1] = d[2] = 0.f; d[3] = 0.f; }
    }
  }
}

template <class T>
inline T ClampToDepth(float v, float maxv) {
  float s = v * maxv;
  if (s < 0.f) s = 0.f;
  if (s > maxv) s = maxv;
  return static_cast<T>(s + 0.5f);
}

}  // namespace

// Returns true if it wrote the destination; false to fall back to passthrough.
bool Sam3dBodyPlugin::renderPipeline(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) return false;
  std::unique_ptr<OFX::Image> src(
      _srcClip && _srcClip->isConnected() ? _srcClip->fetchImage(args.time) : nullptr);
  if (!src.get()) return false;

  if (dst->getPixelComponents() != OFX::ePixelComponentRGBA) return false;

  const OfxRectI b = src->getBounds();
  const int W = b.x2 - b.x1, H = b.y2 - b.y1;
  if (W <= 0 || H <= 0) return false;

  // 1. Read source into a top-down RGBA float frame + a contiguous RGB frame.
  std::vector<float> srcRGBA(static_cast<size_t>(W) * H * 4, 0.f);
  const OFX::BitDepthEnum sbd = src->getPixelDepth();
  if (sbd == OFX::eBitDepthFloat)
    ReadFrameRGBA<float>(src.get(), b, W, H, 1.f, srcRGBA);
  else if (sbd == OFX::eBitDepthUShort)
    ReadFrameRGBA<unsigned short>(src.get(), b, W, H, 1.f / 65535.f, srcRGBA);
  else
    ReadFrameRGBA<unsigned char>(src.get(), b, W, H, 1.f / 255.f, srcRGBA);

  std::vector<float> rgb(static_cast<size_t>(W) * H * 3);
  for (size_t i = 0, n = static_cast<size_t>(W) * H; i < n; ++i) {
    rgb[i * 3 + 0] = srcRGBA[i * 4 + 0];
    rgb[i * 3 + 1] = srcRGBA[i * 4 + 1];
    rgb[i * 3 + 2] = srcRGBA[i * 4 + 2];
  }

  // 2. Assemble params + (re)build the cached pipeline on key change.
  hastur::PipelineParams p;
  std::string paramDir;
  _modelDir->getValue(paramDir);
  p.model_dir = ModelSearchPath(paramDir);
  int cu = 0; _computeUnits->getValue(cu);
  p.units = ChoiceToUnits(cu);
  double d = 0;
  _scoreThresh->getValue(d); p.detector_score_thresh = static_cast<float>(d);
  int mp = 1; _maxPeople->getValue(mp); p.max_people = mp;
  _grey->getValue(d); p.grey = static_cast<float>(d);
  bool ov = false; _overrideCam->getValue(ov); p.override_camera = ov;
  _focal->getValue(d); p.focal_override = static_cast<float>(d);
  _fov->getValue(d); p.fov_override_deg = static_cast<float>(d);
  int ss = 2; _supersample->getValue(ss); p.ssaa = ss;
  bool premult = false; _premult->getValue(premult);
  bool compOverSrc = false; _compOverSrc->getValue(compOverSrc);
  p.premultiply = false;  // pipeline emits straight; premult applied at write
  bool garment = true; _garment->getValue(garment); p.garment = garment;
  double lr = 0, lg = 0, lb = 0; _leotardColor->getValue(lr, lg, lb);
  p.leotard_rgb[0] = static_cast<float>(lr);
  p.leotard_rgb[1] = static_cast<float>(lg);
  p.leotard_rgb[2] = static_cast<float>(lb);
  double kr = 0, kg = 0, kb = 0; _skinColor->getValue(kr, kg, kb);
  p.skin_rgb[0] = static_cast<float>(kr);
  p.skin_rgb[1] = static_cast<float>(kg);
  p.skin_rgb[2] = static_cast<float>(kb);
  int aov = kAovBeauty; _outputAov->getValue(aov);
  // Multi-plane hosts stash the planes they want in g_renderPlanes (via the
  // intercepted GetClipComponents/render actions). If any non-colour plane is
  // requested we must emit AOVs regardless of the single-plane outputAov param.
  bool wantAovPlane = false;
  for (const std::string& pl : g_renderPlanes)
    if (pl != kFnOfxImagePlaneColour) wantAovPlane = true;
  const bool multiPlane = !g_renderPlanes.empty();
  p.emit_aovs = (aov != kAovBeauty) || wantAovPlane;
  int cc = 0; _cryptoCoverage->getValue(cc);
  p.crypto_coverage = static_cast<hastur::CryptoCoverage>(cc);  // 0=Mesh 1=SAM3 2=Both
  // Host frame time drives the stable-person-id track table (gate/reset). Not
  // part of the pixel-keyed compute cache below, so a held still still computes
  // once; genuine moving frames differ in pixels and each advance the tracks.
  p.time = args.time;

  const std::string key = p.model_dir + "|" + std::to_string(cu);

  // 3. Run — result cached in a PROCESS-GLOBAL LRU keyed on the input pixels +
  // geometry params (NOT the write-time outputAov/premult/comp options). This
  // makes the Nuke AOV gizmo (8 instances, one per plane, same input) run
  // inference ONCE, and lets a single node switch outputAov for free. The
  // per-key lock guarantees compute-once even under concurrent host rendering.
  char sig[340];
  std::snprintf(sig, sizeof(sig),
                "%d|%.4f|%d|%.4f|%d|%.4f|%.4f|%d|%d|%dx%d|%016llx"
                "|%d|%.3f,%.3f,%.3f|%.3f,%.3f,%.3f|%d",
                cu, p.detector_score_thresh, p.max_people, p.grey,
                static_cast<int>(p.override_camera), p.focal_override,
                p.fov_override_deg, p.ssaa, static_cast<int>(p.emit_aovs), W, H,
                static_cast<unsigned long long>(HashRgb(rgb)),
                static_cast<int>(p.garment), p.leotard_rgb[0], p.leotard_rgb[1],
                p.leotard_rgb[2], p.skin_rgb[0], p.skin_rgb[1], p.skin_rgb[2],
                static_cast<int>(p.crypto_coverage));
  const std::string fkey = key + "|" + sig;

  std::shared_ptr<hastur::FrameResult> frp = FrameCacheGet(fkey);
  if (!frp) {
    std::shared_ptr<std::mutex> klock = FrameCacheKeyLock(fkey);
    std::lock_guard<std::mutex> computing(*klock);
    frp = FrameCacheGet(fkey);  // another thread may have filled it while we waited
    if (!frp) {
      if (!_pipeline || key != _pipelineKey) {
        _pipeline = std::make_unique<hastur::Sam3dBodyPipeline>();
        _pipelineKey = key;
      }
      auto computed = std::make_shared<hastur::FrameResult>(
          _pipeline->Run(rgb.data(), W, H, p));
      if (static_cast<int>(computed->render.data.size()) != W * H * 4) {
        if (!_pipeline->ok())
          hasturreg::SafeSetMessage(*this, OFX::Message::eMessageError,
                                    "hasturPipeline",
                                    "SAM 3D Body: " + _pipeline->last_error());
        return false;
      }
      if (computed->people.empty() && !_pipeline->last_error().empty()) {
        // No detections / no mesh — not an error worth failing the render; pass
        // the plate through so the host still gets a frame.
        return false;
      }
      FrameCachePut(fkey, computed);
      frp = computed;
    }
  }
  const hastur::FrameResult& fr = *frp;
  const std::vector<float>& R = fr.render.data;  // straight RGBA, top-down
  const hastur::AovBuffers& av = fr.aovs;
  const hastur::CryptoResult& cr = fr.crypto;

  // 4. Compose one output pixel for a given AOV. Beauty composites as before;
  // data AOVs write point-sampled values (coverage in alpha, no premult / over);
  // Cryptomatte layers write their raw packed RGBA. `o[4]` = R,G,B,A.
  auto compose = [&](int aovSel, int rr, int cc, float o[4]) {
    o[0] = o[1] = o[2] = o[3] = 0.f;
    const size_t px = static_cast<size_t>(rr) * W + cc;
    const size_t idx = px * 4;
    if (aovSel == kAovBeauty) {
      const float mr = R[idx + 0], mg = R[idx + 1], mb = R[idx + 2], ma = R[idx + 3];
      if (compOverSrc) {
        const float* sp = &srcRGBA[idx];
        const float inv = 1.f - ma;
        o[0] = mr * ma + sp[0] * inv; o[1] = mg * ma + sp[1] * inv;
        o[2] = mb * ma + sp[2] * inv; o[3] = ma + sp[3] * inv;
      } else if (premult) {
        o[0] = mr * ma; o[1] = mg * ma; o[2] = mb * ma; o[3] = ma;
      } else {
        o[0] = mr; o[1] = mg; o[2] = mb; o[3] = ma;
      }
      return;
    }
    const float cov = px < av.coverage.size() ? av.coverage[px] : 0.f;
    switch (aovSel) {
      case kAovDepth:
        o[0] = o[1] = o[2] = px < av.depth.size() ? av.depth[px] : 0.f;
        o[3] = cov;
        break;
      case kAovPosition:
        if (px * 3 + 2 < av.position.size()) {
          o[0] = av.position[px * 3]; o[1] = av.position[px * 3 + 1];
          o[2] = av.position[px * 3 + 2];
        }
        o[3] = cov;
        break;
      case kAovNormal:
        if (px * 3 + 2 < av.normal.size()) {
          o[0] = av.normal[px * 3]; o[1] = av.normal[px * 3 + 1];
          o[2] = av.normal[px * 3 + 2];
        }
        o[3] = cov;
        break;
      case kAovPref:
        if (px * 3 + 2 < av.pref.size()) {
          o[0] = av.pref[px * 3]; o[1] = av.pref[px * 3 + 1];
          o[2] = av.pref[px * 3 + 2];
        }
        o[3] = cov;
        break;
      case kAovNref:
        if (px * 3 + 2 < av.nref.size()) {
          o[0] = av.nref[px * 3]; o[1] = av.nref[px * 3 + 1];
          o[2] = av.nref[px * 3 + 2];
        }
        o[3] = cov;
        break;
      case kAovST:
        if (av.has_st && px * 2 + 1 < av.st.size()) {
          o[0] = av.st[px * 2]; o[1] = av.st[px * 2 + 1];
        }
        o[3] = cov;
        break;
      case kAovCrypto00:
      case kAovCrypto01: {
        const int lvl = aovSel - kAovCrypto00;
        if (lvl < static_cast<int>(cr.layers.size()) &&
            idx + 3 < cr.layers[lvl].size()) {
          const float* L = &cr.layers[lvl][idx];
          o[0] = L[0]; o[1] = L[1]; o[2] = L[2]; o[3] = L[3];
        }
        break;
      }
      default:
        break;
    }
  };

  const OfxRectI& win = args.renderWindow;
  // Write `aovSel` into an OFX::Image (the RGBA colour plane).
  auto writeImage = [&](OFX::Image* img, int aovSel) {
    const OFX::BitDepthEnum tbd = img->getPixelDepth();
    for (int y = win.y1; y < win.y2; ++y) {
      if (abort()) break;
      const int rr = b.y2 - 1 - y;  // render row (top-down)
      for (int x = win.x1; x < win.x2; ++x) {
        const int cc = x - b.x1;
        float o[4] = {0, 0, 0, 0};
        if (rr >= 0 && rr < H && cc >= 0 && cc < W) compose(aovSel, rr, cc, o);
        void* praw = img->getPixelAddress(x, y);
        if (!praw) continue;
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
  // Write `aovSel` into a raw plane buffer (nComps channels) fetched via the
  // Foundry plane suite. `tb` is the plane's bounds (bottom-up like the frame).
  auto writeRaw = [&](void* base, const OfxRectI& tb, int rowBytes,
                      OFX::BitDepthEnum tbd, int nComps, int aovSel) {
    for (int y = win.y1; y < win.y2; ++y) {
      if (abort()) break;
      if (y < tb.y1 || y >= tb.y2) continue;
      const int rr = b.y2 - 1 - y;
      char* rowp = static_cast<char*>(base) +
                   static_cast<size_t>(y - tb.y1) * rowBytes;
      for (int x = win.x1; x < win.x2; ++x) {
        if (x < tb.x1 || x >= tb.x2) continue;
        const int cc = x - b.x1;
        float o[4] = {0, 0, 0, 0};
        if (rr >= 0 && rr < H && cc >= 0 && cc < W) compose(aovSel, rr, cc, o);
        const size_t off = static_cast<size_t>(x - tb.x1) * nComps;
        if (tbd == OFX::eBitDepthFloat) {
          float* dp = reinterpret_cast<float*>(rowp) + off;
          for (int c = 0; c < nComps; ++c) dp[c] = o[c];
        } else if (tbd == OFX::eBitDepthUShort) {
          unsigned short* dp = reinterpret_cast<unsigned short*>(rowp) + off;
          for (int c = 0; c < nComps; ++c) dp[c] = ClampToDepth<unsigned short>(o[c], 65535.f);
        } else {
          unsigned char* dp = reinterpret_cast<unsigned char*>(rowp) + off;
          for (int c = 0; c < nComps; ++c) dp[c] = ClampToDepth<unsigned char>(o[c], 255.f);
        }
      }
    }
  };

  if (!multiPlane) {
    // Portable single-plane path: the selected outputAov -> the RGBA output.
    writeImage(dst.get(), aov);
  } else {
    // Multi-plane host: satisfy each requested plane. Colour -> the standard
    // output image; data/crypto planes -> their own buffers via the plane suite.
    const FnOfxImageEffectPlaneSuiteV1* ps = PlaneSuite();
    const OfxPropertySuiteV1* props = PropSuite();
    for (const std::string& pl : g_renderPlanes) {
      if (pl == kFnOfxImagePlaneColour) { writeImage(dst.get(), kAovBeauty); continue; }
      const PlaneDef* def = PlaneForEncoded(pl);
      if (!def || !ps || !props) continue;
      OfxPropertySetHandle imgProps = nullptr;
      if (ps->clipGetImagePlane(_dstClip->getHandle(), args.time, pl.c_str(),
                                nullptr, &imgProps) != kOfxStatOK ||
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
      writeRaw(base, tb, rowBytes, tbd, def->nch, def->aov);
    }
  }
  hasturreg::SafeClearMessage(*this);
  return true;
}

// Computes the frame-level camera intrinsics + m44f matrices and the Cryptomatte
// manifest, and stores them on the read-only String params. Runs in changedParam
// (a "bake" push button) rather than render(), because OFX forbids setting params
// during render. None of this needs inference: the matrices are fully determined
// by the frame size + focal/fov params, and the manifest by max_people (the
// person_NN names -> hashes are static).
void Sam3dBodyPlugin::bakeCameraData(double time) {
  OfxRectD rod = _srcClip && _srcClip->isConnected()
                     ? _srcClip->getRegionOfDefinition(time)
                     : _dstClip->getRegionOfDefinition(time);
  const int W = static_cast<int>(rod.x2 - rod.x1);
  const int H = static_cast<int>(rod.y2 - rod.y1);
  if (W <= 0 || H <= 0) return;

  bool ov = false; _overrideCam->getValue(ov);
  double focalOv = 0, fovOv = 0;
  _focal->getValue(focalOv);
  _fov->getValue(fovOv);
  const hastur::CamInt K = hastur::DefaultCamInt(
      W, H, ov ? static_cast<float>(fovOv) : 0.f,
      ov ? static_cast<float>(focalOv) : 0.f);
  const hastur::CameraMatrices cm = hastur::BuildCameraMatrices(
      hastur::CamFocalX(K), hastur::CamCx(K), hastur::CamCy(K), W, H);

  auto mat16 = [](const std::array<float, 16>& m) {
    char buf[16 * 16];
    int n = 0;
    for (int i = 0; i < 16; ++i)
      n += std::snprintf(buf + n, sizeof(buf) - n, "%s%.9g", i ? " " : "", m[i]);
    return std::string(buf);
  };
  char intr[128];
  std::snprintf(intr, sizeof(intr), "focal=%.6g cx=%.6g cy=%.6g W=%d H=%d",
                cm.focal, cm.principal[0], cm.principal[1], W, H);
  _camIntrinsics->setValue(intr);
  _camWorldToNdc->setValue(mat16(cm.world_to_ndc));
  _camNdcToWorld->setValue(mat16(cm.ndc_to_world));

  int mp = 1; _maxPeople->getValue(mp);
  std::string manifest = "{";
  for (int i = 0; i < std::max(1, mp); ++i) {
    char nm[16];
    std::snprintf(nm, sizeof(nm), "person_%02d", i);
    if (i) manifest += ",";
    manifest += "\"" + std::string(nm) + "\":\"" + hastur::CryptoIdHex(nm) + "\"";
  }
  manifest += "}";
  _cryptoManifest->setValue(manifest);
}
#endif  // HASTUR_WITH_ONNX

void Sam3dBodyPlugin::changedParam(const OFX::InstanceChangedArgs& args,
                                   const std::string& name) {
#if HASTUR_WITH_ONNX
  if (name == kParamBakeCamera) {
    try {
      bakeCameraData(args.time);
    } catch (...) {
      // Non-fatal: leave the read-only params as they were.
    }
  }
#else
  (void)args; (void)name;
#endif
}

void Sam3dBodyPlugin::render(const OFX::RenderArguments& args) {
#if HASTUR_WITH_ONNX
  try {
    if (renderPipeline(args)) return;
  } catch (const std::exception& e) {
    hasturreg::SafeSetMessage(*this, OFX::Message::eMessageError, "hasturPipeline",
                              std::string("SAM 3D Body pipeline error: ") + e.what());
  } catch (...) {
    hasturreg::SafeSetMessage(*this, OFX::Message::eMessageError, "hasturPipeline",
                              "SAM 3D Body pipeline error (unknown)");
  }
#endif
  // Fallback: never crash the host — pass the source through unchanged.
  renderPassthrough(args);
}

////////////////////////////////////////////////////////////////////////////////

// Explicit factory (instead of mDeclarePluginFactory) so we can override
// getMainEntry() to install the multi-plane action interceptor.
class Sam3dBodyFactory : public OFX::PluginFactoryHelper<Sam3dBodyFactory> {
 public:
  Sam3dBodyFactory(const std::string& id, unsigned int verMaj, unsigned int verMin)
      : OFX::PluginFactoryHelper<Sam3dBodyFactory>(id, verMaj, verMin) {}
  void load() override {}
  void unload() override {}
  void describe(OFX::ImageEffectDescriptor& desc) override;
  void describeInContext(OFX::ImageEffectDescriptor& desc,
                         OFX::ContextEnum context) override;
  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle,
                                   OFX::ContextEnum context) override;
#if HASTUR_WITH_ONNX
  OfxPluginEntryPoint* getMainEntry() override { return HasturMainEntry; }
#endif
};

using namespace OFX;

void Sam3dBodyFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kPluginName, kPluginName, kPluginName);
  desc.setPluginGrouping(kPluginGrouping);
  desc.setPluginDescription(kPluginDescription);

  desc.addSupportedContext(eContextFilter);
  desc.addSupportedContext(eContextGeneral);
  desc.addSupportedBitDepth(eBitDepthFloat);
  desc.addSupportedBitDepth(eBitDepthUByte);
  desc.addSupportedBitDepth(eBitDepthUShort);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(true);
  // Mesh recovery needs the whole image: no tiling.
  desc.setSupportsTiles(false);
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipPARs(false);

#if HASTUR_WITH_ONNX
  // Opt into the Foundry/Nuke + Natron multi-plane extension so hosts that
  // support it call GetClipComponents and can pull the AOV planes from one node.
  // Non-throwing: hosts without the extension simply ignore it and use the
  // portable outputAov single-plane fallback.
  desc.getPropertySet().propSetInt(kFnOfxImageEffectPropMultiPlanar, 1, false);
#endif
}

void Sam3dBodyFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                         OFX::ContextEnum /*context*/) {
  ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(ePixelComponentRGBA);
  src->setTemporalClipAccess(false);
  src->setSupportsTiles(false);
  src->setIsMask(false);

  ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  PageParamDescriptor* page = desc.definePageParam("Controls");

  {
    StringParamDescriptor* p = desc.defineStringParam(kParamModelDir);
    p->setLabels("Model dir", "Model dir", "Model directory");
    p->setHint("Directory holding person_detector.onnx, sam3dbody_body.onnx, "
               "mhr_assets.bin, pose_corrective.onnx. Empty resolves from "
               "$HASTUR_MODEL_DIR then the bundle Contents/Resources.");
    p->setStringType(eStringTypeDirectoryPath);
    p->setDefault("");
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamComputeUnits);
    p->setLabels("Compute units", "Compute units", "Compute units");
    p->setHint("ONNX Runtime accelerator selection (CoreML MLComputeUnits).");
    p->appendOption("All");
    p->appendOption("CPU + GPU");
    p->appendOption("CPU + Neural Engine");
    p->appendOption("CPU only");
    p->setDefault(0);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamScoreThresh);
    p->setLabels("Detector threshold", "Det threshold", "Detector score threshold");
    p->setHint("Minimum person-detection score. The detector ONNX exports boxes "
               "down to ~0.05, so this gate does the real filtering. Close "
               "subjects pin ~1.0, but distant / backlit / partially-occluded "
               "people score lower and a high gate drops them frame-to-frame "
               "(flicker / bare-shadow dropouts). Default 0.3 keeps distant "
               "subjects covered; raise toward 0.85 if false positives appear "
               "(e.g. animals ~0.7).");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.3);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamMaxPeople);
    p->setLabels("Max people", "Max people", "Max people");
    p->setHint("Cap the per-person loop. Excess low-confidence boxes are "
               "rejected by the detector threshold, so the default is safe "
               "for single-subject shots too.");
    p->setRange(1, 32);
    p->setDisplayRange(1, 8);
    p->setDefault(3);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamGrey);
    p->setLabels("Grey level", "Grey", "Neutral grey level");
    p->setHint("Linear neutral-grey albedo used for the plain-clay look when the "
               "Garment toggle is OFF.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.6);
    page->addChild(*p);
  }
  // --- Garment (leotard) ---------------------------------------------------
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamGarment);
    p->setLabels("Garment", "Garment", "Garment (leotard)");
    p->setHint("Render the figure clothed in a modest leotard (torso + upper "
               "arms + upper legs) instead of a bare mesh. On by default. Turn "
               "OFF for the plain neutral-grey clay (matte / relight base).");
    p->setDefault(true);
    page->addChild(*p);
  }
  {
    RGBParamDescriptor* p = desc.defineRGBParam(kParamLeotardColor);
    p->setLabels("Leotard colour", "Leotard", "Leotard garment colour");
    p->setHint("Linear RGB albedo of the leotard (torso / upper limbs). Default "
               "a fairly saturated blue for clear contrast against skin.");
    p->setDefault(0.08, 0.15, 0.6);  // fairly saturated blue
    page->addChild(*p);
  }
  {
    RGBParamDescriptor* p = desc.defineRGBParam(kParamSkinColor);
    p->setLabels("Skin colour", "Skin", "Skin / body colour");
    p->setHint("Linear RGB albedo of the uncovered skin (head, forearms, hands, "
               "lower legs, feet). Default neutral grey.");
    p->setDefault(0.6, 0.6, 0.6);  // neutral grey
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamOverrideCam);
    p->setLabels("Override camera", "Override cam", "Override camera intrinsics");
    p->setHint("Use the focal / FOV override below instead of the default "
               "diagonal focal.");
    p->setDefault(false);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamFocal);
    p->setLabels("Focal (px)", "Focal", "Focal length (pixels)");
    p->setHint("fx=fy in pixels when Override camera is on (>0 wins over FOV).");
    p->setRange(0.0, 100000.0);
    p->setDisplayRange(0.0, 8000.0);
    p->setDefault(0.0);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamFov);
    p->setLabels("FOV (deg)", "FOV", "Diagonal field of view (degrees)");
    p->setHint("Diagonal FOV used when Override camera is on and Focal is 0.");
    p->setRange(1.0, 179.0);
    p->setDisplayRange(10.0, 120.0);
    p->setDefault(55.0);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamSupersample);
    p->setLabels("Supersample", "SSAA", "Supersample (per axis)");
    p->setHint("Rasterizer edge anti-aliasing factor per axis (1 = off, 2 = 4x).");
    p->setRange(1, 4);
    p->setDisplayRange(1, 4);
    p->setDefault(2);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamPremult);
    p->setLabels("Premultiply", "Premult", "Premultiply output");
    p->setHint("Emit premultiplied (associated) alpha. Off = straight.");
    p->setDefault(false);
    page->addChild(*p);
  }
  {
    BooleanParamDescriptor* p = desc.defineBooleanParam(kParamCompOverSrc);
    p->setLabels("Composite over source", "Over source", "Composite over source");
    p->setHint("Composite the rendered mesh over the input plate. Off = output "
               "the render's grey-on-alpha straight through (default).");
    p->setDefault(false);
    page->addChild(*p);
  }
  // --- AOV output (portable single-plane fallback) + camera data ------------
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamOutputAov);
    p->setLabels("Output AOV", "Output AOV", "Output AOV");
    p->setHint("Which pass this node outputs on its RGBA plane. Beauty is the "
               "grey-mesh render; the rest are point-sampled data passes (no "
               "anti-aliasing) for downstream comp / 3D reconstruction. To pull "
               "several at once, clone the node with different selections — the "
               "inference result is cached per frame, so extra passes are cheap.");
    p->appendOption("Beauty (RGBA)");
    p->appendOption("Depth (Z)");
    p->appendOption("Position (XYZ, camera=world)");
    p->appendOption("Normal (XYZ, camera)");
    p->appendOption("Pref (reference position)");
    p->appendOption("ST (texture UV)");
    p->appendOption("CryptoObject00 (ranks 0,1)");
    p->appendOption("CryptoObject01 (ranks 2,3)");
    p->appendOption("Nref (bind-pose normal)");
    p->setDefault(kAovBeauty);
    page->addChild(*p);
  }
  {
    ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamCryptoCoverage);
    p->setLabels("Crypto coverage", "Crypto cov", "Cryptomatte coverage source");
    p->setHint("Where each person's Cryptomatte matte comes from. Mesh = the MHR "
               "clay silhouette (always available). SAM 3 mask = the detector's "
               "instance mask (matte-quality edges: hair, fingers, motion blur) "
               "when the detector model emits masks, else falls back to the mesh. "
               "Both = per-pixel union of the two. Only affects the Cryptomatte "
               "passes.");
    p->appendOption("Mesh silhouette");        // kCovMesh
    p->appendOption("SAM 3 mask");             // kCovSam3Mask
    p->appendOption("Both (union)");           // kCovBoth
    p->setDefault(kCovMesh);
    page->addChild(*p);
  }
  {
    PushButtonParamDescriptor* p = desc.definePushButtonParam(kParamBakeCamera);
    p->setLabels("Bake camera data", "Bake camera", "Bake camera data");
    p->setHint("Compute the camera intrinsics, world<->NDC m44f matrices and the "
               "Cryptomatte manifest for the current frame and store them in the "
               "read-only fields below (OFX forbids setting these during render).");
    page->addChild(*p);
  }
  auto roString = [&](const char* nm, const char* label, const char* hint) {
    StringParamDescriptor* p = desc.defineStringParam(nm);
    p->setLabels(label, label, label);
    p->setHint(hint);
    p->setStringType(eStringTypeMultiLine);
    p->setEnabled(false);  // read-only display
    p->setEvaluateOnChange(false);
    p->setDefault("");
    page->addChild(*p);
  };
  roString(kParamCamIntrinsics, "Camera intrinsics",
           "focal / principal point / frame size (baked).");
  roString(kParamCamWorldToNdc, "world->NDC (m44f)",
           "16 row-major floats mapping camera/world space to NDC (baked).");
  roString(kParamCamNdcToWorld, "NDC->world (m44f)",
           "16 row-major floats; inverse of world->NDC, for unprojection (baked).");
  roString(kParamCryptoManifest, "Cryptomatte manifest",
           "JSON name->hash map for the person_NN IDs (baked).");
}

OFX::ImageEffect* Sam3dBodyFactory::createInstance(OfxImageEffectHandle handle,
                                                   OFX::ContextEnum /*context*/) {
  return new Sam3dBodyPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static Sam3dBodyFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
#if HASTUR_WITH_ONNX
  g_uid = p.getUID();  // the delegating main entry passes this to mainEntryStr
#endif
  ids.push_back(&p);
}
}  // namespace Plugin
}  // namespace OFX
