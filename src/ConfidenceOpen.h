// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// ConfidenceOpen -- open the inter-limb NEGATIVE SPACE that the SAM 3 detector/tracker mask
// bridges. The detector's SOFT mask already scores an inter-limb gap as LOW CONFIDENCE (prob well
// below the body); a plain `logit>0` (prob>0.5) binarise fills it. This keeps the prob>0.5
// silhouette but RE-OPENS large low-confidence INTERIOR blobs (the negative space), while leaving
// the confident body and the soft edge intact. It is the C++ counterpart of the validated Python
// prototype `Kthanid/tools/crop_pipeline/seed_repair.py::confidence_open`.
//
// Trade-off (inherent, see Kthanid/tools/crop_pipeline/VOID_FILL_BACKLOG.md): "low confidence" also
// covers low-contrast body (dark clothing) and motion-blurred extremities, so opening too hard
// punches body holes / erases extremities. `tau`/`minFrac`/`edgePx` are the dial; the downstream
// eroded-MHR union (Sam3dBody, crypto_coverage=Both) fills back any body holes this leaves.
#pragma once

#include <vector>

namespace hastur {

// Input: per-pixel mask LOGITS at inW*inH (row-major). Output: soft coverage [0,1] at outW*outH.
//  - keep prob = sigmoid(logit) > 0.5 as the silhouette,
//  - zero every INTERIOR connected component (eroded off the silhouette edge by `edgePx`) whose
//    pixels are all below `tau` confidence and whose area is >= `minFrac` of the silhouette area.
// Env overrides (parsed once): HASTUR_CONFOPEN_TAU, HASTUR_CONFOPEN_MINFRAC, HASTUR_CONFOPEN_EDGE.
// HASTUR_CONFOPEN_OFF=1 disables the re-open (returns the plain resized prob) for A/B.
std::vector<float> ConfidenceOpen(const float* logits, int inW, int inH, int outW, int outH,
                                  float tau = 0.85f, float minFrac = 0.004f, int edgePx = 12);

// Erode a soft coverage [0,1] by `radius` px (square SE): keep the soft coverage only where the
// >0.5 foreground lies deeper than `radius` from its own boundary; zero the outer band. Used to
// pull the MHR mesh body off the limb surfaces before it fills back the negative space
// (Sam3dBody crypto_coverage=Both). radius<=0 returns a copy unchanged.
std::vector<float> ErodeSoftByRadius(const std::vector<float>& cov, int W, int H, int radius);

}  // namespace hastur
