// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// Sam3dBodyPipeline — the M4 integration sync point. Wires every stage engine
// (person detector -> per-person crop + camera conditioning -> SAM 3D body
// regressor -> pose-corrective ONNX -> C++ MHR mesh -> perspective camera solve
// -> software rasterizer -> depth-ordered over-composite) into a single call:
//
//   FrameResult Run(const float* rgb, int W, int H, const PipelineParams&);
//
// Mirrors the reference orchestration (sam-3d-body @ c259bfc,
// sam_3d_body_estimator.process_one_image) but BODY-ONLY, single-person-loop.
// Hands are M7: the hand presence gate is computed and stored on the
// PersonResult, but the hand refiner is not run (TODO(M7) hook below).
//
// Session ownership: the pose-corrective session is owned via OrtSessionManager
// (shared Ort::Env, CPU — it is a dense fp32 matmul). The detector and body
// regressor keep their own self-managing engine wrappers (DetectorEngine /
// Sam3dBodyEngine), each of which already does the CoreML/CUDA-with-CPU-fallback
// session dance internally. All sessions are created lazily on the first Run().
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "MeshTypes.h"

namespace hastur {

// Per-frame pipeline tunables (mapped 1:1 from the OFX plugin params). The
// heavy, session-defining fields (model_dir, units) are read on the FIRST Run()
// and then fixed for the lifetime of the pipeline object; the OFX layer rebuilds
// the pipeline when those change (keyed like humbaba's _engineKey).
struct PipelineParams {
  // Colon-separated search path of directories holding the model/asset files
  // (person_detector.onnx, sam3dbody_body.onnx, mhr_assets.bin,
  // pose_corrective.onnx). $HASTUR_MODEL_DIR is appended automatically.
  std::string model_dir;
  ComputeUnits units = ComputeUnits::All;

  float detector_score_thresh = 0.5f;  // person score gate
  int max_people = 1;                  // cap the per-person loop (M4: 1)

  float grey = 0.6f;                   // neutral clay albedo (linear)
  int ssaa = 2;                        // rasterizer supersampling per axis

  bool override_camera = false;        // use focal_override / fov_override_deg
  float focal_override = 0.f;          // fx=fy in px (>0 wins)
  float fov_override_deg = 0.f;        // diagonal FOV (used if focal<=0)

  bool premultiply = false;            // emit premultiplied (associated) alpha

  float hand_tau = 0.5f;               // hand-presence gate (M7 hook only)
  int intra_threads = 0;               // ORT intra-op threads (0 = default)
};

class Sam3dBodyPipeline {
 public:
  Sam3dBodyPipeline();
  ~Sam3dBodyPipeline();

  Sam3dBodyPipeline(const Sam3dBodyPipeline&) = delete;
  Sam3dBodyPipeline& operator=(const Sam3dBodyPipeline&) = delete;

  // Runs the full body-only pipeline on a top-down, row-major HWC RGB image
  // (3 channels, float in [0,1]) of size W x H. Returns the depth-ordered
  // per-person results and the composited grey-mesh RGBA at frame resolution.
  // Never throws: on any stage failure it records last_error() and returns
  // whatever was produced so far (possibly an empty render).
  FrameResult Run(const float* rgb, int W, int H, const PipelineParams& p);

  // True once all required sessions/assets loaded successfully.
  bool ok() const;
  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;

  bool EnsureLoaded(const PipelineParams& p);
};

}  // namespace hastur
