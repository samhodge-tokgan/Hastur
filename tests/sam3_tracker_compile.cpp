// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// sam3_tracker_compile — a compile/link smoke for the SAM 3 video tracker engine
// (Phase 5e-i). It only #includes the two engine headers and asserts the mirrored
// model config constants; it instantiates nothing (running the tracker needs the 5
// ONNX graphs, which are runtime assets, not in the repo). Its purpose is to prove
// the engine headers parse against Hastur's OrtSessionManager/OrtAccel/ComputeUnits
// and that the sources link into an ONNX-enabled build. The plugin target already
// compiles the .cpp sources; this is a small, explicit, named guard on top of that.

#include "Sam3MultiTracker.h"
#include "Sam3TrackerEngine.h"

// Config constants shared with the built SAM 3 model must agree between the
// single-object engine and the N-object driver (same probed graph geometry).
static_assert(hastur::kNumMaskMem == hastur::kMNumMaskMem, "num_maskmem mismatch");
static_assert(hastur::kMemDim == hastur::kMMemDim, "mem_dim mismatch");
static_assert(hastur::kHidden == hastur::kMHidden, "hidden mismatch");
static_assert(hastur::kMaxObjPtrs == hastur::kMMaxObjPtrs, "max_obj_ptrs mismatch");
static_assert(hastur::kGrid == hastur::kMGrid, "grid mismatch");
static_assert(hastur::kImgSize == hastur::kMImgSize, "img_size mismatch");
static_assert(hastur::kLowRes == hastur::kMLowRes, "low_res mismatch");
static_assert(hastur::kPtrTokens == hastur::kMPtrTokens, "ptr_tokens mismatch");

int main() {
  // Reference the tracker types without constructing them (no graphs available).
  (void)sizeof(hastur::Sam3TrackerEngine);
  (void)sizeof(hastur::Sam3MultiTracker);
  (void)sizeof(hastur::Sam3Paths);
  (void)sizeof(hastur::MultiPaths);
  return 0;
}
