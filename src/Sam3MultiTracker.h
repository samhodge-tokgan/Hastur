// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Sam3MultiTracker — N-object SAM 3 video tracker with host-side detect/associate.
// Phase 5d-ii: generalizes the validated single-object Sam3TrackerEngine to many
// objects, each holding its own independent recurrent memory (proven independent in
// M1). Memory is bounded per object: maskmem ring buffer (num_maskmem=7, i.e. 1 cond
// + up to 6 recent) and obj-ptr history capped at max_obj_ptrs_in_encoder=16 — the
// single-object version retained every frame; here the per-object `recent` deque is
// pruned to <= kMaxObjPtrs so storage is O(17) frames/object regardless of clip length.
//
// Per frame the driver does (host-side association, ported from sam3_video_base.py,
// kept OUT of the ONNX graphs):
//   BeginFrame -> G1 backbone (once, object-independent)
//   Detect     -> G2 detector (person text) + host sigmoid*presence threshold + mask-IoU NMS
//   associate  -> IoU(dets, propagated track masks); matched tracks keep id,
//                 unmatched high-conf dets (score>=new_det_thresh) spawn new tracks,
//                 tracks unmatched too long are pruned (keep-alive)
//   PropagateObj per matched/kept track (G3 memory-attn, G4 SAM heads, G5 mem-encoder)
//
// The graphs G1..G5 + constants are exactly those the single-object engine validated
// at IoU 0.997 vs the PyTorch golden; the per-object propagate math is a faithful copy.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "OrtSessionManager.h"

namespace hastur {

// config mirrored from the built SAM 3 model (probed) — same as single-object engine
constexpr int kMNumMaskMem = 7;      // 1 cond + 6 recent
constexpr int kMMemDim = 64;
constexpr int kMHidden = 256;
constexpr int kMMaxObjPtrs = 16;     // obj-ptr history cap == recent-deque cap
constexpr float kMMfThreshold = 0.01f;
constexpr int kMGrid = 72;
constexpr int kMHW = kMGrid * kMGrid;  // 5184
constexpr int kMImgSize = 1008;
constexpr int kMLowRes = 288;          // G4 low-res + G2 det-mask resolution
constexpr int kMPtrTokens = kMHidden / kMMemDim;  // 4

// association defaults (build_sam3_video_model, apply_temporal_disambiguation=True)
constexpr float kScoreThresh = 0.5f;    // score_threshold_detection
constexpr float kDetNmsIou = 0.1f;      // det_nms_thresh
constexpr float kNewDetThresh = 0.7f;   // new_det_thresh (spawn)
constexpr float kAssocIou = 0.1f;       // assoc_iou_thresh (det matches a track)
constexpr float kTrkAssocIou = 0.5f;    // trk_assoc_iou_thresh (track counts as matched)
constexpr int kKeepAliveMax = 30;       // init/max keep-alive; prune when it hits 0

struct MultiPaths {
  std::string g1, g2, g3, g4, g5;
  std::string tpos, no_mem, proj_w, proj_b;   // learned constants (.npy f32)
  std::string lang_feats, lang_mask;          // "person" text features (.npy f32)
};

// A host-side detection for the current frame.
struct Det {
  float score = 0.f;
  std::vector<uint8_t> bin288;    // 288*288 binary (mask logits > 0)
  std::vector<float> soft1008;    // 1008*1008 soft [0,1] for seeding a new track
  long area = 0;
};

class Sam3MultiTracker {
 public:
  Sam3MultiTracker(const MultiPaths& paths, Ep ep = Ep::Auto,
                   ComputeUnits units = ComputeUnits::All);
  ~Sam3MultiTracker();
  Sam3MultiTracker(const Sam3MultiTracker&) = delete;
  Sam3MultiTracker& operator=(const Sam3MultiTracker&) = delete;

  bool accelerator_active() const { return accel_active_; }

  // ---- generic tensor ----
  struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    size_t elems() const { size_t n = 1; for (auto d : shape) n *= (size_t)d; return n; }
  };
  // one processed frame's recurrent record, per object
  struct Frame {
    bool cond = false;
    int idx = 0;
    Tensor mem, mempos, obj_ptr;  // [1,64,72,72],[1,64,72,72],[1,256]
    float osl = 0.f, iou = 0.f, eff = 0.f;
  };
  // one object's independent, bounded recurrent state
  struct ObjState {
    int id = -1;
    int cond_idx = 0;
    int born = 0;
    bool active = true;
    int last_match = -1;         // last frame idx a detection matched this track
    int keep_alive = kKeepAliveMax;
    Frame cond_frame;            // single conditioning frame
    std::deque<Frame> recent;    // non-cond, pruned to <= kMMaxObjPtrs
    std::vector<float> last_low; // last propagated low-res logits (288*288)
    long last_area = 0;
    size_t max_recent = 0;       // high-water mark (to prove bounding)
  };

  // Run the shared backbone for a video frame (object-independent). Caches features.
  void BeginFrame(const float* frame_chw1008, int frame_idx, int total_frames);
  // Run the detector on the cached frame; host threshold + mask-IoU NMS.
  std::vector<Det> Detect(float score_thresh = kScoreThresh, float nms_iou = kDetNmsIou);

  // Seed a new object from a detection soft mask on the current (cached) frame.
  // Returns the seed high-res logits (1008*1008). Fills obj.last_low downsampled.
  void SeedObj(ObjState& obj, const float* seed_mask1008);
  // Propagate an existing object on the current (cached) frame. Returns low-res
  // logits (288*288); also stored in obj.last_low.
  // reverse=true mirrors the SAM3 `track_in_reverse` path (Phase 5d-iii bidirectional):
  // neighbor maskmem/obj-ptr frames are selected on the FUTURE side (idx>frame_idx,
  // toward the mid-clip seed) and the conditioning-frame object-pointer temporal
  // position is sign-negated (tpos_sign_mul=-1). Graphs are unchanged; only the
  // host-side memory-feed assembly differs.
  const std::vector<float>& PropagateObj(ObjState& obj, bool reverse = false);

 private:
  struct OrtGraph {
    std::shared_ptr<OrtSessionManager::Handle> h;
    std::vector<std::string> in_names, out_names;
  };
  void InitGraph(OrtGraph& g, std::shared_ptr<OrtSessionManager::Handle> h);
  std::vector<Tensor> RunFloat(
      OrtGraph& g, const std::vector<std::pair<std::string, const Tensor*>>& feeds);

  void G1(const Tensor& image);  // fills cur_*
  std::vector<Tensor> G2();      // pred_logits, pred_boxes, pred_masks, presence
  Tensor G3(const Tensor& src, const Tensor& src_pos, const Tensor& prompt,
            const Tensor& prompt_pos, int64_t num_obj_ptr_tokens);
  void G4(const Tensor& bf, const Tensor& hr0, const Tensor& hr1, Tensor& low,
          Tensor& high, Tensor& ious, Tensor& obj_ptr, Tensor& osl);
  void G5(const Tensor& pix_feat, const Tensor& high, const Tensor& osl, Tensor& mem,
          Tensor& mempos);

  static Tensor FlattenHWBC(const Tensor& bchw);
  static Tensor SeqToBCHW(const Tensor& seq);
  static void Sine1DPE(const std::vector<float>& pos, int dim, std::vector<float>& out);
  std::pair<float, float> EffIou(float osl, const float* ious3) const;
  std::vector<int> FrameFilterObj(const ObjState& o, int frame_idx, bool reverse) const;
  const Frame* GetFrame(const ObjState& o, int idx) const;
  static Tensor LoadNpyFloat(const std::string& path);

  OrtSessionManager mgr_;
  OrtGraph g1_, g2_, g3_, g4_, g5_;
  Ort::MemoryInfo mem_info_;
  bool accel_active_ = false;

  Tensor tpos_, no_mem_, proj_w_, proj_b_, lang_feats_;
  std::vector<uint8_t> lang_mask_bool_;  // [1,32] bool
  std::vector<int64_t> lang_mask_shape_;

  // cached per-frame backbone features (object-independent)
  Tensor cur_f0_, cur_f1_, cur_f2_, cur_pos2_;
  Tensor cur_detf0_, cur_detf1_, cur_detf2_, cur_detpos_;
  int cur_frame_idx_ = 0;
  int total_frames_ = 0;
};

}  // namespace hastur
