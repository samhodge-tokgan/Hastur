// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// MhrModel.h -- C++/Eigen reimplementation of Meta's MHR (Momentum Human Rig)
// parametric body model. Takes a 519-d MHR-head parameter vector and produces
// the posed 18,439-vertex Mesh, replacing the shipped 664 MB TorchScript.
//
// Pipeline (exactly mirrors tools/mhr_meshmodule.py, which is validated
// bit-for-bit against the TorchScript):
//   pred[519]
//     -> decode: 6D global rot -> ZYX euler; continuous body pose -> euler
//        (mhr_utils); hand PCA -> euler; scale PCA -> scales; => model_params[204]
//     -> identity rest = base_shape + shape_basis . shape_coeffs
//     -> joint_parameters[889] = parameter_transform . [model_params, 0_45]
//     -> per-joint local skeleton state (trans+prerot*euler_quat+exp scale)
//     -> forward kinematics (prefix-multiply over the pmi levels, double precision)
//     -> + pose-corrective vertex offsets (from pose_corrective.onnx, via the hook)
//     -> linear blend skinning (sparse skin weights, per-joint transform)
//     -> keypoints = keypoint_mapping . [verts, joints]
//     -> /100 (cm->m) and verts/joints[...,{1,2}] *= -1 (camera-frame flip)
//
// Eigen is used for the dense linear algebra; the header keeps Eigen out of its
// public surface so callers only need MeshTypes.h.

#pragma once

#include <array>
#include <memory>

#include "MeshAssets.h"
#include "MeshTypes.h"

namespace hastur {

class MhrModel {
 public:
  explicit MhrModel(std::shared_ptr<const MeshAssets> assets);

  // Full decode of the 519-d vector to the 889-d MHR joint_parameters. This is
  // the exact input pose_corrective.onnx expects; production computes it here,
  // runs the ONNX, and feeds the result back into Run() as `pose_corrective`.
  std::array<float, 889> JointParameters(const std::array<float, kParamDim>& pred) const;

  // Decode the 519-d vector to the 204-d MHR model_params + 45-d shape coeffs
  // (exposed mainly for testing / debugging the decode stage).
  void Decode(const std::array<float, kParamDim>& pred,
              std::array<float, 204>& model_params,
              std::array<float, kShapeComps>& shape) const;

  // Posed mesh. `pose_corrective` (optional) is a pointer to kNumVerts*3 floats:
  // the pose-corrective vertex offsets in the model's internal cm frame (pre-flip),
  // as produced by pose_corrective.onnx. If null, correctives are treated as zero.
  Mesh Run(const std::array<float, kParamDim>& pred,
           const float* pose_corrective = nullptr) const;

  // Wrist-gate geometry for the M7 hand-refiner's wrist-angle criterion, computed
  // from a body pred[519] (sam3d_body.run_inference, "CRITERIA 1"):
  //   * ori_local[side]   = roma.euler_to_rotmat("XZY", body_pose[wrist eulers])
  //                         for side 0=left (params 41,43,42), 1=right (31,33,32).
  //   * lowarm_global[side]= the FK global joint rotation at the lower-arm joints
  //                         [76=left, 40=right] (== joint_global_rots there).
  // Both are row-major 3x3. The refiner combines lowarm_global with the model's
  // joint_rotation[wrist_twist] and the hand decoder's wrist_global to form the
  // fused local wrist rotation and compare its angle to ori_local.
  struct WristGate {
    std::array<std::array<float, 9>, 2> ori_local{};
    std::array<std::array<float, 9>, 2> lowarm_global{};
  };
  WristGate ComputeWristGate(const std::array<float, kParamDim>& pred) const;

 private:
  std::shared_ptr<const MeshAssets> a_;

  // Cached dimensions.
  int V_{}, J_{};

  // Shared decode helper producing model_params[204] and joint_parameters[889].
  void DecodeInternal(const std::array<float, kParamDim>& pred,
                      std::array<float, 204>& model_params,
                      std::array<float, kShapeComps>& shape,
                      std::array<float, 889>& joint_params) const;
};

}  // namespace hastur
