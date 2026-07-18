// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Sam3dBodyEngine — ONNX Runtime wrapper around the body-only SAM-3D-Body core
// regressor exported by tools/export_sam3dbody.py (Track B / M3).
//
// The fused graph has a FIXED batch=1 / 512² shape and three inputs and four
// outputs, exactly mirroring the frozen contract in src/MeshTypes.h:
//
//   inputs : image[1,3,512,512]  (ImageNet-normalized CHW)
//            ray_cond[1,2,32,32] (token-grid ray directions)
//            condition_info[1,3] (CLIFF cx/f, cy/f, b/f)
//   outputs: pred[1,519]  pred_cam[1,3]  hand_logits[1,2,2]  hand_box[1,2,4]
//
// The graph has fp16 inputs/outputs; this engine takes/returns fp32 (the
// CropInputs / BodyPrediction contract) and does the fp16 cast at the boundary.
//
// It mirrors the humbaba MoGeEngine / Hastur DetectorEngine structure: create a
// session on the platform accelerator EP (CoreML on macOS, CUDA elsewhere) with a
// CPU fallback, then Run() one CropInputs -> one BodyPrediction. Some graph ops
// (GridSample, Atan2, ScatterND from the MHR refinement) fall back to CPU under
// CoreML; the session still builds and runs. See tools/BODY_EXPORT_REPORT.md.
#pragma once

#include <memory>
#include <string>

#include "MeshTypes.h"  // hastur::CropInputs, BodyPrediction, ComputeUnits

namespace hastur {

class Sam3dBodyEngine {
 public:
  // Loads `model_path` (the body-regressor ONNX) and creates a session.
  // `intra_threads` <= 0 leaves the ORT default.
  Sam3dBodyEngine(const std::string& model_path, ComputeUnits units,
                  int intra_threads);
  ~Sam3dBodyEngine();

  Sam3dBodyEngine(const Sam3dBodyEngine&) = delete;
  Sam3dBodyEngine& operator=(const Sam3dBodyEngine&) = delete;

  // Runs the body regressor on one prepared crop. Returns the 519-d MHR
  // parameter vector, the 3-d perspective camera params, and the per-hand
  // (L,R) presence logits + crop-frame boxes. On failure returns a zeroed
  // BodyPrediction and sets last_error().
  BodyPrediction Run(const CropInputs& in);

  // True if the session actually placed nodes on the platform accelerator EP.
  bool accelerator_active() const;

  // Whether the platform accelerator EP is compiled into this ORT build.
  static bool AcceleratorAvailable();

  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

}  // namespace hastur
