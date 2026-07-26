// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// MultiPlane.h — the shared Foundry/Nuke + Natron multi-plane surface, extracted
// verbatim from HasturPlugin's original AOV machinery so BOTH bundle plugins (SAM
// 3D Body and Matte Partition) drive the SAME plane wire-protocol code instead of
// each reinventing it.
//
// Implemented directly against the vendored fnOfxExtensions.h constants + the raw
// plane/property suites, so the SAM-licensed plugins need no GPL support code.
//
// A plugin describes its plane set as a table of `PlaneDef` and:
//   * EncodePlane()      -> the Natron/Nuke component string for a plane,
//   * PlaneForEncoded()  -> the reverse lookup (with the v0.10.1 Natron token-match
//                           fallback that is CRITICAL for Natron round-tripping),
//   * PropSuite()/PlaneSuite() -> the raw OFX property + Foundry plane suites,
//   * StashRenderPlanes()-> capture the planes the host asked for this render.
//
// None of this is inference/ONNX specific; the header is dependency-light (OFX
// support headers + the vendored Foundry/Natron extension constants).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"

#include "nuke/fnOfxExtensions.h"

namespace hplane {

// One named output/input plane: a stable id, a display label, and its channels.
struct PlaneDef {
  int aov;                 // plugin-specific plane selector
  const char* id;          // stable unique plane id (e.g. "hastur.CryptoObject00")
  const char* label;       // display label
  const char* chans[4];    // channel names (nch valid entries)
  int nch;                 // channel count
};

// Natron/Nuke multi-plane component-string encoding for one plane.
inline std::string EncodePlane(const PlaneDef& p) {
  std::string s = std::string(kNatronOfxImageComponentsPlaneName) + p.id +
                  kNatronOfxImageComponentsPlaneLabel + p.label;
  for (int i = 0; i < p.nch; ++i)
    s += std::string(kNatronOfxImageComponentsPlaneChannel) + p.chans[i];
  return s;
}

// Reverse lookup: match an encoded component string to one of `planes`.
inline const PlaneDef* PlaneForEncoded(const std::string& enc,
                                       const PlaneDef* planes, size_t n) {
  // Exact match first (Nuke round-trips our EncodePlane string verbatim).
  for (size_t i = 0; i < n; ++i)
    if (EncodePlane(planes[i]) == enc) return &planes[i];
  // Natron re-encodes the requested plane with an extra _ChannelsLabel_<..>
  // segment our EncodePlane omits, so match on the stable PlaneName id token:
  //   NatronOfxImageComponentsPlaneName_<id>_PlaneLabel_
  for (size_t i = 0; i < n; ++i) {
    const std::string tok = std::string(kNatronOfxImageComponentsPlaneName) +
                            planes[i].id + kNatronOfxImageComponentsPlaneLabel;
    if (enc.find(tok) != std::string::npos) return &planes[i];
  }
  return nullptr;
}

inline const OfxPropertySuiteV1* PropSuite() {
  static const OfxPropertySuiteV1* s =
      static_cast<const OfxPropertySuiteV1*>(OFX::fetchSuite(kOfxPropertySuite, 1, true));
  return s;
}
inline const FnOfxImageEffectPlaneSuiteV1* PlaneSuite() {
  static const FnOfxImageEffectPlaneSuiteV1* s =
      static_cast<const FnOfxImageEffectPlaneSuiteV1*>(
          OFX::fetchSuite(kFnOfxImageEffectPlaneSuite, 1, true));
  return s;
}

// Capture the planes the host asked for this render call (from the render inArgs)
// into `out`. Cleared first; a missing property list yields an empty `out`.
inline void StashRenderPlanes(OfxPropertySetHandle inArgs,
                              std::vector<std::string>& out) {
  out.clear();
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
      out.emplace_back(v);
  }
  if (std::getenv("HASTUR_PLANE_DEBUG")) {
    std::fprintf(stderr, "[plane-dbg] StashRenderPlanes n=%d\n", n);
    for (const std::string& pl : out)
      std::fprintf(stderr, "[plane-dbg]   requested plane=[%s]\n", pl.c_str());
  }
}

}  // namespace hplane
