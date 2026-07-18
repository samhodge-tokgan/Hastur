// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// HandRefinerEngine -- ONNX Runtime wrapper around the SAM-3D-Body hand-decoder
// 2nd pass exported by tools/export_hand.py (M7). It is the direct analogue of
// Sam3dBodyEngine: a FIXED batch=1 / 512x512 fp16 graph with the SAME three-input
// crop contract as the body regressor, and two outputs:
//
//   inputs : image[1,3,512,512]  (ImageNet-normalized CHW -- a HAND crop, pad 0.9)
//            ray_cond[1,2,32,32] (token-grid ray directions of the hand crop)
//            condition_info[1,3] (CLIFF cx/f, cy/f, b/f)
//   outputs: pred[1,519]         (hand-decoder MHR-head parameter vector)
//            wrist_global[1,2,3,3] (global wrist rotmats, joints [78 L, 42 R])
//
// The engine takes/returns fp32 and casts at the fp16 tensor boundary, exactly
// like Sam3dBodyEngine. It is created LAZILY by Sam3dBodyPipeline only when a hand
// is gated on, and if models/sam3dbody_hand.onnx is absent the pipeline simply
// skips hand refinement (graceful body-only fallback). Like the body graph, the
// interleaved MHR-refinement ops fall back to CPU under CoreML; that is expected.
#pragma once

#include <array>
#include <memory>
#include <string>

#include "MeshTypes.h"  // hastur::CropInputs, ComputeUnits

namespace hastur {

// The hand-decoder result for one crop: the 519-d MHR-head param vector (we slice
// the hand PCA[108] / scale[28] / shape[45] out of it) plus the two wrist global
// rotation matrices (row-major 3x3), joint order [0]=left(78), [1]=right(42).
struct HandPrediction {
  std::array<float, kParamDim> pred{};                 // 519
  std::array<std::array<float, 9>, 2> wrist_global{};  // [L,R] row-major 3x3
  bool valid{false};
};

class HandRefinerEngine {
 public:
  HandRefinerEngine(const std::string& model_path, ComputeUnits units,
                    int intra_threads);
  ~HandRefinerEngine();

  HandRefinerEngine(const HandRefinerEngine&) = delete;
  HandRefinerEngine& operator=(const HandRefinerEngine&) = delete;

  // Runs the hand decoder on one prepared hand crop. On failure returns a
  // HandPrediction with valid=false and sets last_error().
  HandPrediction Run(const CropInputs& in);

  // True if the session was created successfully.
  bool ok() const;
  bool accelerator_active() const;
  static bool AcceleratorAvailable();

  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

}  // namespace hastur
