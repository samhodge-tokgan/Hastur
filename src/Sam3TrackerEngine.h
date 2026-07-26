// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Sam3TrackerEngine — the STATEFUL, in-order SAM 3 single-object video tracker core.
// It drives the 5 exported per-frame ONNX graphs (G1 backbone, G2 detector [seed only],
// G3 memory-attention, G4 SAM heads, G5 memory encoder) via ONNX Runtime and owns ALL
// recurrent memory in C++, mirroring sam3_tracker_base.py's track_step /
// _prepare_memory_conditioned_features / _encode_new_memory exactly.
//
// Recurrent state (per processed frame, held in `frames_`):
//   maskmem_features [1,64,72,72]  spatial memory (from G5)
//   maskmem_pos_enc  [1,64,72,72]  spatial memory pos enc (from G5)
//   obj_ptr          [1,256]       object pointer (from G4; seed approximates the
//                                  mask-prompt decoder with the memory-less G4 ptr)
//   object_score_logits [1,1]      obj presence logit
//   iou_score, eff_iou_score       memory-selection scores (SAM2Long frame_filter)
// Constant params (learned, not in any graph; loaded from .npy):
//   maskmem_tpos_enc [7,1,1,64], no_mem_embed [1,1,256],
//   obj_ptr_tpos_proj weight [64,256] + bias [64]
// The attended memory per frame is num_maskmem=7 (1 cond + up to 6 recent) plus up to
// max_obj_ptrs_in_encoder=16 object-pointer frames, each ptr split into 4 tokens of 64.
//
// Usage (mirrors the harness ti-loop):
//   eng.Seed(frame0_chw1008, seed_mask_1008, total_frames);   // ti=0
//   auto low = eng.Propagate(frame_i_chw1008);                // ti=1..N-1
// Frames are planar CHW float, 3x1008x1008, already normalised (mean/std 0.5).
// seed_mask is 1008x1008 float in [0,1] (the add_new_mask-resized detection mask).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "OrtSessionManager.h"

namespace hastur {

// config mirrored from the built model (probed):
constexpr int kNumMaskMem = 7;          // 1 cond + 6 recent
constexpr int kMemDim = 64;
constexpr int kHidden = 256;
constexpr int kMaxObjPtrs = 16;
constexpr float kMfThreshold = 0.01f;
constexpr int kGrid = 72;               // top-level feature grid
constexpr int kHW = kGrid * kGrid;      // 5184 tokens
constexpr int kImgSize = 1008;
constexpr int kLowRes = 288;
constexpr int kPtrTokens = kHidden / kMemDim;   // 4 tokens per obj_ptr

struct Sam3Paths {
  std::string g1, g3, g4, g5;
  std::string tpos, no_mem, proj_w, proj_b;   // constant .npy files
};

class Sam3TrackerEngine {
 public:
  Sam3TrackerEngine(const Sam3Paths& paths, Ep ep = Ep::Auto,
                    ComputeUnits units = ComputeUnits::All);
  ~Sam3TrackerEngine();

  Sam3TrackerEngine(const Sam3TrackerEngine&) = delete;
  Sam3TrackerEngine& operator=(const Sam3TrackerEngine&) = delete;

  // Seed the tracker on frame 0 from the detection mask; clears all recurrent state.
  // `frame_chw1008` = 3*1008*1008 planar float; `seed_mask1008` = 1008*1008 float [0,1].
  // Returns the seed high-res mask logits [1008*1008] (= mask*20-10).
  std::vector<float> Seed(const float* frame_chw1008, const float* seed_mask1008,
                          int total_frames);

  // Track a subsequent frame. Returns low-res mask logits (288*288).
  std::vector<float> Propagate(const float* frame_chw1008);

  bool accelerator_active() const { return accel_active_; }
  int num_frames() const { return static_cast<int>(frames_.size()); }

 private:
  struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    size_t elems() const {
      size_t n = 1; for (auto d : shape) n *= static_cast<size_t>(d); return n;
    }
  };
  struct OrtGraph {
    std::shared_ptr<OrtSessionManager::Handle> h;
    std::vector<std::string> in_names, out_names;
  };
  // one processed frame's recurrent record
  struct Frame {
    bool cond = false;
    Tensor mem;       // [1,64,72,72]
    Tensor mempos;    // [1,64,72,72]
    Tensor obj_ptr;   // [1,256]
    float osl = 0.f;  // object_score_logits scalar
    float iou = 0.f;
    float eff = 0.f;
  };

  void InitGraph(OrtGraph& g, std::shared_ptr<OrtSessionManager::Handle> h);
  std::vector<Tensor> RunFloat(
      OrtGraph& g, const std::vector<std::pair<std::string, const Tensor*>>& feeds);

  // graph calls
  void G1(const Tensor& image, Tensor& f0, Tensor& f1, Tensor& f2, Tensor& pos2);
  Tensor G3(const Tensor& src, const Tensor& src_pos, const Tensor& prompt,
            const Tensor& prompt_pos, int64_t num_obj_ptr_tokens);
  void G4(const Tensor& bf, const Tensor& hr0, const Tensor& hr1, Tensor& low,
          Tensor& high, Tensor& ious, Tensor& obj_ptr, Tensor& osl);
  void G5(const Tensor& pix_feat, const Tensor& high, const Tensor& osl, Tensor& mem,
          Tensor& mempos);

  // host-side helpers (mirror engine_ref)
  static Tensor FlattenHWBC(const Tensor& bchw);   // [1,C,72,72] -> [5184,1,C]
  static Tensor SeqToBCHW(const Tensor& seq);      // [5184,1,C] -> [1,C,72,72]
  std::vector<int> FrameFilter(int frame_idx) const;
  static void Sine1DPE(const std::vector<float>& pos, int dim, std::vector<float>& out);
  std::pair<float, float> EffIou(float osl, const float* ious3) const;
  static Tensor LoadNpyFloat(const std::string& path);

  OrtSessionManager mgr_;
  OrtGraph g1_, g3_, g4_, g5_;
  Ort::MemoryInfo mem_info_;
  bool accel_active_ = false;

  // constants
  Tensor tpos_;     // [7,1,1,64]
  Tensor no_mem_;   // [1,1,256]
  Tensor proj_w_;   // [64,256]
  Tensor proj_b_;   // [64]

  // recurrent state
  std::vector<Frame> frames_;   // index == frame_idx
  int total_frames_ = 0;
};

}  // namespace hastur
