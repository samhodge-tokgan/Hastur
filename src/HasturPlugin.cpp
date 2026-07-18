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
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.H"

#include "Register.h"

#if HASTUR_WITH_ONNX
#include "OrtAccel.h"
#include "Sam3dBodyPipeline.h"
#endif

#define kPluginName "SAM 3D Body"
#define kPluginGrouping "Tokgan"
#define kPluginDescription \
  "Recover a posed 3D human body mesh (MHR) from a single image using SAM 3D " \
  "Body via ONNX Runtime with hardware acceleration."
#define kPluginIdentifier "com.tokgan.Sam3dBody"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

// Param names.
#define kParamModelDir "modelDir"
#define kParamComputeUnits "computeUnits"
#define kParamScoreThresh "scoreThresh"
#define kParamMaxPeople "maxPeople"
#define kParamGrey "greyLevel"
#define kParamOverrideCam "overrideCamera"
#define kParamFocal "focalPx"
#define kParamFov "fovDeg"
#define kParamSupersample "supersample"
#define kParamPremult "premult"
#define kParamCompOverSrc "compositeOverSource"

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
  OFX::BooleanParam* _overrideCam = nullptr;
  OFX::DoubleParam* _focal = nullptr;
  OFX::DoubleParam* _fov = nullptr;
  OFX::IntParam* _supersample = nullptr;
  OFX::BooleanParam* _premult = nullptr;
  OFX::BooleanParam* _compOverSrc = nullptr;

#if HASTUR_WITH_ONNX
  std::unique_ptr<hastur::Sam3dBodyPipeline> _pipeline;
  std::string _pipelineKey;  // model dir + compute units (humbaba _engineKey)
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
    _overrideCam = fetchBooleanParam(kParamOverrideCam);
    _focal = fetchDoubleParam(kParamFocal);
    _fov = fetchDoubleParam(kParamFov);
    _supersample = fetchIntParam(kParamSupersample);
    _premult = fetchBooleanParam(kParamPremult);
    _compOverSrc = fetchBooleanParam(kParamCompOverSrc);
  }

  void render(const OFX::RenderArguments& args) override;

 private:
  void renderPassthrough(const OFX::RenderArguments& args);
#if HASTUR_WITH_ONNX
  bool renderPipeline(const OFX::RenderArguments& args);
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

  const OFX::BitDepthEnum bd = dst->getPixelDepth();
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

  const std::string key = p.model_dir + "|" + std::to_string(cu);
  if (!_pipeline || key != _pipelineKey) {
    _pipeline = std::make_unique<hastur::Sam3dBodyPipeline>();
    _pipelineKey = key;
  }

  // 3. Run.
  hastur::FrameResult fr = _pipeline->Run(rgb.data(), W, H, p);
  const std::vector<float>& R = fr.render.data;  // straight RGBA, top-down
  if (static_cast<int>(R.size()) != W * H * 4) {
    if (!_pipeline->ok())
      hasturreg::SafeSetMessage(*this, OFX::Message::eMessageError, "hasturPipeline",
                                "SAM 3D Body: " + _pipeline->last_error());
    return false;
  }
  if (fr.people.empty() && !_pipeline->last_error().empty()) {
    // No detections / no mesh — not an error worth failing the render; pass
    // the plate through so the host still gets a frame.
    return false;
  }

  // 4. Write dst over the render window, mapping OFX bottom-up -> render top-down.
  auto compose = [&](int rr, int cc, float& oR, float& oG, float& oB, float& oA) {
    const size_t idx = (static_cast<size_t>(rr) * W + cc) * 4;
    const float mr = R[idx + 0], mg = R[idx + 1], mb = R[idx + 2], ma = R[idx + 3];
    if (compOverSrc) {
      const float* sp = &srcRGBA[idx];
      const float inv = 1.f - ma;
      oR = mr * ma + sp[0] * inv;
      oG = mg * ma + sp[1] * inv;
      oB = mb * ma + sp[2] * inv;
      oA = ma + sp[3] * inv;
    } else if (premult) {
      oR = mr * ma; oG = mg * ma; oB = mb * ma; oA = ma;
    } else {
      oR = mr; oG = mg; oB = mb; oA = ma;
    }
  };

  const OfxRectI& win = args.renderWindow;
  for (int y = win.y1; y < win.y2; ++y) {
    if (abort()) break;
    const int rr = b.y2 - 1 - y;  // render row (top-down)
    for (int x = win.x1; x < win.x2; ++x) {
      const int cc = x - b.x1;
      float oR = 0, oG = 0, oB = 0, oA = 0;
      if (rr >= 0 && rr < H && cc >= 0 && cc < W) compose(rr, cc, oR, oG, oB, oA);
      if (bd == OFX::eBitDepthFloat) {
        float* dp = static_cast<float*>(dst->getPixelAddress(x, y));
        if (dp) { dp[0] = oR; dp[1] = oG; dp[2] = oB; dp[3] = oA; }
      } else if (bd == OFX::eBitDepthUShort) {
        unsigned short* dp = static_cast<unsigned short*>(dst->getPixelAddress(x, y));
        if (dp) {
          dp[0] = ClampToDepth<unsigned short>(oR, 65535.f);
          dp[1] = ClampToDepth<unsigned short>(oG, 65535.f);
          dp[2] = ClampToDepth<unsigned short>(oB, 65535.f);
          dp[3] = ClampToDepth<unsigned short>(oA, 65535.f);
        }
      } else {
        unsigned char* dp = static_cast<unsigned char*>(dst->getPixelAddress(x, y));
        if (dp) {
          dp[0] = ClampToDepth<unsigned char>(oR, 255.f);
          dp[1] = ClampToDepth<unsigned char>(oG, 255.f);
          dp[2] = ClampToDepth<unsigned char>(oB, 255.f);
          dp[3] = ClampToDepth<unsigned char>(oA, 255.f);
        }
      }
    }
  }
  hasturreg::SafeClearMessage(*this);
  return true;
}
#endif  // HASTUR_WITH_ONNX

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

mDeclarePluginFactory(Sam3dBodyFactory, {}, {});

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
    p->setHint("Minimum person-detection score.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.5);
    page->addChild(*p);
  }
  {
    IntParamDescriptor* p = desc.defineIntParam(kParamMaxPeople);
    p->setLabels("Max people", "Max people", "Max people");
    p->setHint("Cap the per-person loop (M4 default 1).");
    p->setRange(1, 32);
    p->setDisplayRange(1, 8);
    p->setDefault(1);
    page->addChild(*p);
  }
  {
    DoubleParamDescriptor* p = desc.defineDoubleParam(kParamGrey);
    p->setLabels("Grey level", "Grey", "Neutral grey level");
    p->setHint("Linear neutral-grey clay albedo for the rendered mesh.");
    p->setRange(0.0, 1.0);
    p->setDisplayRange(0.0, 1.0);
    p->setDefault(0.6);
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
}

OFX::ImageEffect* Sam3dBodyFactory::createInstance(OfxImageEffectHandle handle,
                                                   OFX::ContextEnum /*context*/) {
  return new Sam3dBodyPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static Sam3dBodyFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
  ids.push_back(&p);
}
}  // namespace Plugin
}  // namespace OFX
