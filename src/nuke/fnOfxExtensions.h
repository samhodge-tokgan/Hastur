// Minimal vendored subset of the Foundry/Nuke OpenFX multi-plane extension.
//
// These are public OpenFX extension constants and suite structs (the "Fn" =
// Foundry namespace) used by Nuke and Natron for arbitrary image planes / AOVs.
// Reproduced from the AcademySoftwareFoundation / NatronGitHub `openfx`
// distribution's include/nuke/fnOfxExtensions.h and include/ofxNatron.h. Only the
// symbols Hastur uses are included, so the SAM-licensed plugin needs no GPL
// support code (Natron's ofxsMultiPlane helper is GPL and is deliberately NOT
// used). Constant string VALUES are the wire protocol and must match verbatim.
//
// See docs/AOVS.md for how Hastur drives these.

#pragma once

#include "ofxImageEffect.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Opt-in + actions ------------------------------------------------------
// Set to int 1 on the effect descriptor to declare the plugin multi-planar; the
// host then calls the GetClipComponents action to learn each clip's planes.
#define kFnOfxImageEffectPropMultiPlanar "uk.co.thefoundry.OfxImageEffectPropMultiPlanar"

// Custom action: the plugin lists, per clip, the planes it produces / needs.
#define kFnOfxImageEffectActionGetClipComponents \
  "uk.co.thefoundry.OfxImageEffectActionGetClipComponents"

// In the GetClipComponents outArgs, the per-clip plane list is a multi-value
// char* property whose NAME is this prefix concatenated with the clip name
// (e.g. "uk.co.thefoundry.OfxNeededComp_Output").
#define kFnOfxImageEffectActionGetClipComponentsPropString \
  "uk.co.thefoundry.OfxNeededComp_"

// Pass-through (for planes the plugin does not itself produce): which clip/time
// the host should pull the missing planes from.
#define kFnOfxImageEffectPropPassThroughComponents \
  "uk.co.thefoundry.OfxImageEffectPropPassThroughComponents"
#define kFnOfxImageEffectPropPassThroughClip \
  "uk.co.thefoundry.ImageEffectPropPassThroughClip"
#define kFnOfxImageEffectPropPassThroughTime \
  "uk.co.thefoundry.ImageEffectPropPassThroughTime"

// Render inArgs: the list of planes the host wants produced in this render call.
#define kOfxImageEffectPropRenderPlanes "OfxImageEffectPropRenderPlanes"

// The standard colour (RGBA) plane string.
#define kFnOfxImagePlaneColour "uk.co.thefoundry.OfxImagePlaneColour"

// --- Plane suite -----------------------------------------------------------
#define kFnOfxImageEffectPlaneSuite "uk.co.thefoundry.FnOfxImageEffectPlaneSuite"

typedef struct FnOfxImageEffectPlaneSuiteV1 {
  // Fetch a named plane's image from a clip. `region` may be NULL (full RoD).
  // The returned property set describes the image exactly like a normal OFX
  // image (kOfxImagePropData / Bounds / RowBytes / PixelDepth / Components).
  OfxStatus (*clipGetImagePlane)(OfxImageClipHandle clip, OfxTime time,
                                 const char* plane, const OfxRectD* region,
                                 OfxPropertySetHandle* imageHandle);
} FnOfxImageEffectPlaneSuiteV1;

// --- Natron plane/component string encoding --------------------------------
// A custom plane with N named channels is encoded into a single component string:
//   PlaneName + <id> [ + PlaneLabel + <label> ] [ + ChannelsLabel + <label> ]
//   + Channel + <ch0> + Channel + <ch1> + ...
#define kNatronOfxImageComponentsPlaneName "NatronOfxImageComponentsPlaneName_"
#define kNatronOfxImageComponentsPlaneLabel "_PlaneLabel_"
#define kNatronOfxImageComponentsPlaneChannelsLabel "_ChannelsLabel_"
#define kNatronOfxImageComponentsPlaneChannel "_Channel_"

#ifdef __cplusplus
}
#endif
