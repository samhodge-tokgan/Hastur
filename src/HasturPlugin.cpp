// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// SAM 3D Body OpenFX plugin (project "Hastur").
//
// GOAL: recover a posed 3D human body/hand mesh (MHR) from a single image and render
// it back into the OFX stream. The full pipeline this effect will orchestrate:
//
//   TODO(detector)   person_detector.onnx  -> person / bbox detection (first stage)
//   TODO(body)       sam3dbody_body.onnx   -> SAM 3D body ViT (per-person body tokens)
//   TODO(hand)       sam3dbody_hand.onnx   -> hand ViT (per-hand refinement)
//   TODO(mhr)        mhr_assets.bin        -> MHR mesh recovery (params -> vertices)
//   TODO(pose)       pose_corrective.onnx  -> pose-corrective refinement
//   TODO(raster)     software rasterizer   -> composite the mesh over the plate
//
// M0: this is a PURE PASSTHROUGH. It advertises the OFX interface and copies the
// source image straight to the destination so the bundle loads and renders in an OFX
// host (Natron/Nuke/Resolve/Flame) before any inference is wired in. The reusable ONNX
// Runtime accelerator + isolation scaffold (src/OrtAccel.h, WinLoader.cpp, CMake) is in
// place; the engine stages above are added in later milestones.

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <memory>
#include <string>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.H"

#include "Register.h"

#if HASTUR_WITH_ONNX
#include "OrtAccel.h"  // reusable CoreML/CUDA EP selection (used by the engine stages)
#endif

#define kPluginName "SAM 3D Body"
#define kPluginGrouping "Tokgan"
#define kPluginDescription \
  "Recover a posed 3D human body/hand mesh (MHR) from a single image using SAM 3D " \
  "Body via ONNX Runtime with hardware acceleration. (M0: passthrough scaffold.)"
#define kPluginIdentifier "com.tokgan.Sam3dBody"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 1

////////////////////////////////////////////////////////////////////////////////
// Passthrough copy processor (M0: copies src RGBA/A straight to dst).

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
      for (int x = w.x1; x < w.x2; ++x) {
        const PIX* src = static_cast<const PIX*>(_srcImg ? _srcImg->getPixelAddress(x, y) : nullptr);
        if (src)
          for (int c = 0; c < nComponents; ++c) dst[c] = src[c];
        else
          for (int c = 0; c < nComponents; ++c) dst[c] = PIX(0);
        dst += nComponents;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

class Sam3dBodyPlugin : public OFX::ImageEffect {
 protected:
  OFX::Clip* _dstClip;
  OFX::Clip* _srcClip;

 public:
  explicit Sam3dBodyPlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
  }

  void render(const OFX::RenderArguments& args) override;

 private:
  void renderPassthrough(const OFX::RenderArguments& args);
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

void Sam3dBodyPlugin::render(const OFX::RenderArguments& args) {
  // TODO(detector/body/hand/mhr/pose/raster): run the SAM 3D Body pipeline and
  // composite the recovered mesh. M0 simply passes the source through unchanged.
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

  // TODO(params): parameters for model dir ($HASTUR_MODEL_DIR), compute units,
  // detection threshold, mesh overlay controls, etc. land with the engine stages.
  desc.definePageParam("Controls");
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
  // TODO(stages): register additional companion factories here as the pipeline grows
  // (see src/Register.h).
#if HASTUR_WITH_ONNX
  // Scaffold liveness anchor: touch ONNX Runtime so the bundled, privately-renamed
  // libonnxruntime_hastur is a REAL load-/delay-load dependency of the .ofx — which is
  // exactly what the CI isolation checks (NEEDED/otool/exports) assert. Harmless (queries
  // available providers once); replaced by the real engine sessions in later milestones.
  static const bool kAccel = hastur::AcceleratorAvailable();
  (void)kAccel;
#endif
}
}  // namespace Plugin
}  // namespace OFX
