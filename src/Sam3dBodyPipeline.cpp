// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
#include "Sam3dBodyPipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "CameraSolver.h"
#include "CropAffine.h"
#include "DetectorEngine.h"
#include "HandRefinerEngine.h"
#include "MeshAssets.h"
#include "MhrModel.h"
#include "OrtSessionManager.h"
#include "Sam3dBodyEngine.h"
#include "SoftwareRasterizer.h"

namespace hastur {

namespace {

namespace fs = std::filesystem;

// Body-crop padding (reference: 1.25 for the body, 0.9 for hands).
constexpr float kBodyPad = 1.25f;
constexpr float kHandPad = 0.9f;

// 519-d pred slice offsets (MeshTypes.h layout).
constexpr int kShapeOff = kGlobal6D + kBodyCont;               // 266
constexpr int kScaleOff = kShapeOff + kShapeComps;             // 311
constexpr int kHandOff  = kScaleOff + kScaleComps;             // 339 (left54|right54)

// Reference thresholds (sam3d_body.run_inference).
constexpr float kWristAngleThresh = 1.4f;   // CRITERIA 1
constexpr float kHandBoxSizeThresh = 64.f;  // CRITERIA 2 (full-frame px)

// --- tiny row-major 3x3 helpers for the wrist-angle criterion ---------------
using Mat3 = std::array<float, 9>;
Mat3 Mat3MulT_A(const Mat3& a, const Mat3& b) {  // a^T * b
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += a[k * 3 + i] * b[k * 3 + j];
      r[i * 3 + j] = s;
    }
  return r;
}
Mat3 Mat3Mul(const Mat3& a, const Mat3& b) {  // a * b
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[k * 3 + j];
      r[i * 3 + j] = s;
    }
  return r;
}
// rotation_angle_difference(A,B) = acos((trace(A B^T) - 1)/2).
float RotAngleDiff(const Mat3& A, const Mat3& B) {
  float tr = 0.f;
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) tr += A[i * 3 + k] * B[i * 3 + k];  // A . B (== A B^T trace)
  float c = (tr - 1.f) * 0.5f;
  if (c > 1.f) c = 1.f;
  if (c < -1.f) c = -1.f;
  return std::acos(c);
}

// Split a colon-separated search path into existing directories, then append
// $HASTUR_MODEL_DIR (also colon-separated) so an env override always applies.
std::vector<std::string> SearchDirs(const std::string& model_dir) {
  std::vector<std::string> dirs;
  auto push_split = [&](const std::string& s) {
    size_t i = 0;
    while (i <= s.size()) {
      size_t j = s.find(':', i);
      if (j == std::string::npos) j = s.size();
      if (j > i) dirs.emplace_back(s.substr(i, j - i));
      i = j + 1;
    }
  };
  push_split(model_dir);
  if (const char* env = std::getenv("HASTUR_MODEL_DIR")) push_split(env);
  return dirs;
}

// First existing `dir/name` across the search dirs, else empty.
std::string FindFile(const std::vector<std::string>& dirs, const std::string& name) {
  for (const std::string& d : dirs) {
    std::error_code ec;
    fs::path p = fs::path(d) / name;
    if (fs::exists(p, ec)) return p.string();
  }
  return {};
}

float Sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// Coarse per-stage timing, printed to stderr only when HASTUR_PIPELINE_TIMING
// is set. Cheap std::chrono; zero cost when the env var is absent.
using Clock = std::chrono::steady_clock;
bool TimingOn() {
  static const bool on = std::getenv("HASTUR_PIPELINE_TIMING") != nullptr;
  return on;
}
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
void LogStage(const char* name, double ms) {
  if (TimingOn()) std::fprintf(stderr, "[hastur-timing] %-18s %8.1f ms\n", name, ms);
}

// Over-composite `fg` (premultiplied RGBA) on top of `acc` (premultiplied),
// both W*H*4. acc is modified in place. Standard Porter-Duff "over".
void OverComposite(std::vector<float>& acc, const std::vector<float>& fg) {
  const size_t n = acc.size();
  for (size_t i = 0; i + 3 < n; i += 4) {
    const float fa = fg[i + 3];
    const float inv = 1.f - fa;
    acc[i + 0] = fg[i + 0] + acc[i + 0] * inv;
    acc[i + 1] = fg[i + 1] + acc[i + 1] * inv;
    acc[i + 2] = fg[i + 2] + acc[i + 2] * inv;
    acc[i + 3] = fa + acc[i + 3] * inv;
  }
}

}  // namespace

struct Sam3dBodyPipeline::Impl {
  bool loaded = false;
  bool ok = false;

  // Fixed at first load.
  std::string model_dir;
  ComputeUnits units = ComputeUnits::All;

  std::unique_ptr<DetectorEngine> det;
  std::unique_ptr<Sam3dBodyEngine> body;
  std::shared_ptr<MeshAssets> assets;
  std::unique_ptr<MhrModel> mhr;

  // M7 hand refiner: located at load, created lazily on the first gated hand.
  std::string hand_path;                        // empty -> refinement disabled
  std::unique_ptr<HandRefinerEngine> hand;      // lazy
  bool hand_tried = false;                      // attempted lazy creation
  int intra_threads = 0;

  // joint_rotation[wrist_twist] rotmats for CRITERIA 1: [0]=left(joint77),
  // [1]=right(joint41), row-major 3x3. Loaded from mhr_wrist.bin if present;
  // when absent the wrist-angle criterion is skipped (box-size gate only).
  bool wrist_loaded = false;
  std::array<std::array<float, 9>, 2> joint_rotation_wrist{};

  // Refine the gated hands of `person` in place (updates person.pred.pred and
  // person.has_hands to reflect which hands were actually merged). Returns true
  // if any hand was merged (caller re-runs the MHR mesh on the updated pred).
  bool RefineHands(PersonResult& person, const float* rgb, int W, int H,
                   const CamInt& K, const PipelineParams& p);

  // Pose-corrective session, owned via the shared-env manager.
  std::unique_ptr<OrtSessionManager> ort;
  std::shared_ptr<OrtSessionManager::Handle> pose;
  std::string pose_in_name, pose_out_name;

  // Run pose_corrective.onnx: joint_parameters[889] -> offset[kNumVerts*3]
  // (cm, pre-flip). Returns false (offset untouched) on any failure.
  bool RunPoseCorrective(const std::array<float, 889>& jp,
                         std::vector<float>& offset);
};

Sam3dBodyPipeline::Sam3dBodyPipeline() : impl_(std::make_unique<Impl>()) {}
Sam3dBodyPipeline::~Sam3dBodyPipeline() = default;

bool Sam3dBodyPipeline::ok() const { return impl_ && impl_->ok; }

bool Sam3dBodyPipeline::EnsureLoaded(const PipelineParams& p) {
  Impl& s = *impl_;
  if (s.loaded) return s.ok;
  s.loaded = true;
  s.model_dir = p.model_dir;
  s.units = p.units;

  const std::vector<std::string> dirs = SearchDirs(p.model_dir);

  const std::string det_path = FindFile(dirs, "person_detector.onnx");
  const std::string body_path = FindFile(dirs, "sam3dbody_body.onnx");
  const std::string assets_path = FindFile(dirs, "mhr_assets.bin");
  const std::string pose_path = FindFile(dirs, "pose_corrective.onnx");
  // Hand refiner is OPTIONAL: absence just disables 2nd-pass refinement.
  s.hand_path = FindFile(dirs, "sam3dbody_hand.onnx");
  s.intra_threads = p.intra_threads;

  std::string missing;
  if (det_path.empty()) missing += " person_detector.onnx";
  if (body_path.empty()) missing += " sam3dbody_body.onnx";
  if (assets_path.empty()) missing += " mhr_assets.bin";
  if (pose_path.empty()) missing += " pose_corrective.onnx";
  if (!missing.empty()) {
    last_error_ = "model/asset files not found:" + missing +
                  " (set HASTUR_MODEL_DIR or the model-dir param)";
    return false;
  }

  try {
    s.det = std::make_unique<DetectorEngine>(det_path, p.units, p.intra_threads);
    s.body = std::make_unique<Sam3dBodyEngine>(body_path, p.units, p.intra_threads);
    s.assets = MeshAssets::Load(assets_path);
    s.mhr = std::make_unique<MhrModel>(s.assets);

    // Pose-corrective: dense fp32 matmul — run it on CPU (fast, avoids a
    // multi-hundred-MB CoreML compile), on the shared env.
    s.ort = std::make_unique<OrtSessionManager>("HasturPipeline");
    s.pose = s.ort->MakeSession(pose_path, p.units, Ep::Cpu, /*gpu_mem_limit=*/0,
                                /*lazy=*/false);
    Ort::Session& psess = s.pose->Get();  // throws if unavailable
    Ort::AllocatorWithDefaultOptions alloc;
    s.pose_in_name = psess.GetInputNameAllocated(0, alloc).get();
    s.pose_out_name = psess.GetOutputNameAllocated(0, alloc).get();
  } catch (const std::exception& e) {
    last_error_ = std::string("session/asset load failed: ") + e.what();
    return false;
  }

  // Optional wrist-twist joint_rotation constants for the wrist-angle criterion
  // (mhr_wrist.bin = 18 float32: left joint77 rotmat(9), right joint41 rotmat(9)).
  const std::string wrist_path = FindFile(dirs, "mhr_wrist.bin");
  if (!wrist_path.empty()) {
    std::error_code ec;
    auto sz = fs::file_size(wrist_path, ec);
    if (!ec && sz == sizeof(float) * 18) {
      std::FILE* f = std::fopen(wrist_path.c_str(), "rb");
      if (f) {
        float buf[18];
        if (std::fread(buf, sizeof(float), 18, f) == 18) {
          for (int i = 0; i < 9; ++i) {
            s.joint_rotation_wrist[0][i] = buf[i];
            s.joint_rotation_wrist[1][i] = buf[9 + i];
          }
          s.wrist_loaded = true;
        }
        std::fclose(f);
      }
    }
  }

  s.ok = true;
  return true;
}

bool Sam3dBodyPipeline::Impl::RunPoseCorrective(const std::array<float, 889>& jp,
                                                std::vector<float>& offset) {
  Sam3dBodyPipeline::Impl& s = *this;
  Ort::Session* psess = s.pose ? s.pose->TryGet() : nullptr;
  if (!psess) return false;
  try {
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> ishape{1, 889};
    // CreateTensor needs a non-const data pointer.
    std::array<float, 889> jp_copy = jp;
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem, jp_copy.data(), jp_copy.size(), ishape.data(), ishape.size());
    const char* in_names[] = {s.pose_in_name.c_str()};
    const char* out_names[] = {s.pose_out_name.c_str()};
    std::vector<Ort::Value> r = psess->Run(Ort::RunOptions{nullptr}, in_names, &in,
                                           1, out_names, 1);
    if (r.empty() || !r[0].IsTensor()) return false;
    const size_t want = static_cast<size_t>(kNumVerts) * 3;
    if (r[0].GetTensorTypeAndShapeInfo().GetElementCount() < want) return false;
    const float* out = r[0].GetTensorData<float>();
    offset.assign(out, out + want);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// M7 hand refiner. Mirrors the per-hand building block of sam3d_body.run_inference:
// crop each gated hand from hand_box (pad 0.9), run the hand decoder, gate on the
// wrist-angle + hand-box-size criteria, and merge the refined hand PCA / scale /
// shape into the body pred[519] via the reference's replace_hands_in_pose logic.
//
// Conventions that must be exact:
//   * hand_box[side] = [cx, cy, w, h] normalized to [0,1] of the 512 BODY crop;
//     side 0 = LEFT, 1 = RIGHT. Mapped to a full-frame square box via the body
//     crop's inverse affine, then re-cropped (pad 0.9) as its own 512 hand crop.
//   * L/R FLIP: the left hand is cropped from a HORIZONTALLY-MIRRORED frame (box
//     x also mirrored) so it is processed exactly like a right hand. The hand
//     decoder's RIGHT-slot output (hand[54:108]) is therefore the refined hand for
//     BOTH crops; for the left it becomes the body's LEFT slot (hand[0:54]).
//   * pred[519] hand block = [left54 | right54] at kHandOff.
bool Sam3dBodyPipeline::Impl::RefineHands(PersonResult& person, const float* rgb,
                                          int W, int H, const CamInt& K,
                                          const PipelineParams& p) {
  Impl& s = *this;
  if (s.hand_path.empty()) return false;

  // Lazy engine creation on the first gated hand of the whole frame.
  if (!s.hand && !s.hand_tried) {
    s.hand_tried = true;
    try {
      s.hand = std::make_unique<HandRefinerEngine>(s.hand_path, s.units,
                                                   s.intra_threads);
      if (!s.hand->ok()) {
        std::fprintf(stderr, "[hastur] hand refiner load failed: %s\n",
                     s.hand->last_error().c_str());
        s.hand.reset();
        s.hand_path.clear();  // don't retry every person
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[hastur] hand refiner exception: %s\n", e.what());
      s.hand.reset();
      s.hand_path.clear();
    }
  }
  if (!s.hand) return false;

  const std::array<float, kParamDim>& bpred = person.pred.pred;

  // Body crop affine (frame<->crop) for placing hand boxes back in the frame.
  CropInputs bcrop = MakeCrop(rgb, W, H, person.box, K, kBodyPad);
  const std::array<float, 6>& inv = bcrop.affine_inv;   // crop px -> frame px
  const float crop_per_frame = bcrop.affine[0];         // frame -> crop scale

  // Scale-flip constants (left hand crop is processed as a right hand).
  const float* scale_mean = s.assets->f32("scale_mean");    // (68,)
  const float* scale_comps = s.assets->f32("scale_comps");  // (28,68)
  const float sr_mean = scale_mean[8], sl_mean = scale_mean[9];
  const float sr_std = scale_comps[8 * 68 + 8], sl_std = scale_comps[9 * 68 + 9];

  // Wrist-angle gate geometry from the BODY pred (CRITERIA 1).
  MhrModel::WristGate wg = s.mhr->ComputeWristGate(bpred);

  // Lazily built mirrored frame (only if a left hand is refined).
  std::vector<float> flipped;

  // Per-side accumulation for the shared-shape/scale averaging.
  bool apply[2] = {false, false};
  std::array<float, 54> hand54[2]{};
  float hand_scale_coeff[2] = {0, 0};             // body pred scale[8]/[9]
  std::array<float, 10> shared_scale[2]{};        // pred scale[18:28]
  std::array<float, 5> shared_shape[2]{};         // pred shape[40:45]

  const int S = kBodyImageSize;

  for (int side = 0; side < 2; ++side) {
    const std::array<float, 4>& hb = person.pred.hand_box[side];  // cx,cy,w,h [0,1]
    const float cx_c = hb[0] * S, cy_c = hb[1] * S;               // crop px
    const float wpx_c = hb[2] * S, hpx_c = hb[3] * S;
    const float sq_c = std::max(wpx_c, hpx_c);                    // square, crop px
    if (sq_c <= 1.f) continue;

    // CRITERIA 2: hand box size in full-frame px must exceed the threshold.
    const float sq_frame = sq_c / std::max(crop_per_frame, 1e-6f);
    const bool box_ok = sq_frame > kHandBoxSizeThresh;
    if (!box_ok) continue;

    // Hand box center in full-frame px (crop -> frame via the body inverse affine).
    const float fx = inv[0] * cx_c + inv[1] * cy_c + inv[2];
    const float fy = inv[3] * cx_c + inv[4] * cy_c + inv[5];
    const float half = 0.5f * sq_frame;

    CropInputs hc;
    if (side == 0) {
      // LEFT: mirror the frame and the box so it is processed like a right hand.
      if (flipped.empty()) {
        flipped.assign(static_cast<size_t>(W) * H * 3, 0.f);
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            const float* src = rgb + (static_cast<size_t>(y) * W + (W - 1 - x)) * 3;
            float* dst = &flipped[(static_cast<size_t>(y) * W + x) * 3];
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
          }
      }
      const float fxm = (W - 1) - fx;
      BBox hbox{fxm - half, fy - half, fxm + half, fy + half, person.box.score};
      hc = MakeCrop(flipped.data(), W, H, hbox, K, kHandPad);
    } else {
      BBox hbox{fx - half, fy - half, fx + half, fy + half, person.box.score};
      hc = MakeCrop(rgb, W, H, hbox, K, kHandPad);
    }

    HandPrediction hp = s.hand->Run(hc);
    if (!hp.valid) continue;

    // Refined hand for THIS crop is always the decoder's RIGHT slot (hand[54:108]).
    for (int i = 0; i < 54; ++i) hand54[side][i] = hp.pred[kHandOff + 54 + i];

    // Hand scale coefficient: right crop -> body scale[8]; left crop -> body
    // scale[9] via the flip formula.
    const float rc8 = hp.pred[kScaleOff + 8];
    hand_scale_coeff[side] =
        (side == 0) ? (((sr_mean + sr_std * rc8) - sl_mean) / sl_std) : rc8;

    for (int i = 0; i < 10; ++i) shared_scale[side][i] = hp.pred[kScaleOff + 18 + i];
    for (int i = 0; i < 5; ++i) shared_shape[side][i] = hp.pred[kShapeOff + 40 + i];

    // Predicted GLOBAL wrist rotmat for this hand: decoder joint 42 (right slot).
    // For the left crop, un-flip it (negate rows 1,2) to get the body-left wrist.
    Mat3 pred_wrist = hp.wrist_global[1];
    if (side == 0)
      for (int r = 1; r < 3; ++r)
        for (int c = 0; c < 3; ++c) pred_wrist[r * 3 + c] *= -1.f;

    // CRITERIA 1: wrist-angle. fused_local = wrist_zero^T * pred_wrist, where
    // wrist_zero = lowarm_global * joint_rotation[wrist_twist]. Compare its angle
    // to the body decoder's own local wrist rotation. Skipped (pass) when the
    // joint_rotation constants are unavailable.
    bool angle_ok = true;
    float ang = -1.f;
    if (s.wrist_loaded) {
      Mat3 wrist_zero = Mat3Mul(wg.lowarm_global[side], s.joint_rotation_wrist[side]);
      Mat3 fused = Mat3MulT_A(wrist_zero, pred_wrist);
      ang = RotAngleDiff(wg.ori_local[side], fused);
      angle_ok = ang < kWristAngleThresh;
    }
    if (std::getenv("HASTUR_HAND_DEBUG")) {
      std::fprintf(stderr,
                   "[hastur-hand] side=%d(%s) sq_frame=%.1fpx box_ok=%d "
                   "wrist_loaded=%d angle=%.3frad angle_ok=%d\n",
                   side, side ? "R" : "L", sq_frame, box_ok, s.wrist_loaded,
                   ang, angle_ok);
    }
    // HASTUR_HAND_FORCE bypasses the wrist-angle criterion (box-size gate only).
    if (!angle_ok && !std::getenv("HASTUR_HAND_FORCE")) continue;

    apply[side] = true;
  }

  if (!apply[0] && !apply[1]) {
    person.has_hands = false;  // nothing merged after gating
    return false;
  }

  // --- merge into the body pred[519] (replace_hands_in_pose) ------------------
  std::array<float, kParamDim>& pred = person.pred.pred;
  if (apply[0])  // left
    for (int i = 0; i < 54; ++i) pred[kHandOff + i] = hand54[0][i];
  if (apply[1])  // right
    for (int i = 0; i < 54; ++i) pred[kHandOff + 54 + i] = hand54[1][i];
  if (apply[0]) pred[kScaleOff + 9] = hand_scale_coeff[0];
  if (apply[1]) pred[kScaleOff + 8] = hand_scale_coeff[1];

  // Shared scale[18:28] / shape[40:45]: mean of the applied hands (reference
  // weights each by its validity).
  const float denom = (apply[0] ? 1.f : 0.f) + (apply[1] ? 1.f : 0.f);
  if (denom > 0.f) {
    for (int i = 0; i < 10; ++i) {
      float v = 0.f;
      if (apply[0]) v += shared_scale[0][i];
      if (apply[1]) v += shared_scale[1][i];
      pred[kScaleOff + 18 + i] = v / denom;
    }
    for (int i = 0; i < 5; ++i) {
      float v = 0.f;
      if (apply[0]) v += shared_shape[0][i];
      if (apply[1]) v += shared_shape[1][i];
      pred[kShapeOff + 40 + i] = v / denom;
    }
  }
  return true;
}

FrameResult Sam3dBodyPipeline::Run(const float* rgb, int W, int H,
                                   const PipelineParams& p) {
  FrameResult result;
  result.render.width = W;
  result.render.height = H;
  result.render.data.assign(static_cast<size_t>(W) * H * 4, 0.f);
  if (!rgb || W <= 0 || H <= 0) {
    last_error_ = "invalid input frame";
    return result;
  }
  if (!EnsureLoaded(p)) return result;
  Impl& s = *impl_;

  // Camera intrinsics for the whole frame (shared by conditioning + projection).
  const CamInt K = DefaultCamInt(
      W, H, p.override_camera ? p.fov_override_deg : 0.f,
      p.override_camera ? p.focal_override : 0.f);

  // --- Stage 1: person detection ------------------------------------------
  auto t0 = Clock::now();
  Detections dets = s.det->Run(rgb, W, H, p.detector_score_thresh);
  LogStage("detector", Ms(t0, Clock::now()));
  if (dets.empty()) {
    last_error_ = "no persons detected";
    return result;  // valid (empty) frame; host passes source through
  }

  const int max_people = std::max(1, p.max_people);
  const int n_people = std::min(static_cast<int>(dets.size()), max_people);

  RasterOptions ropt;
  ropt.grey = p.grey;
  ropt.ssaa = std::max(1, p.ssaa);
  ropt.premultiply = true;  // composite in premultiplied space internally

  // --- Per-person loop ----------------------------------------------------
  for (int i = 0; i < n_people; ++i) {
    const BBox& box = dets[i];
    PersonResult person;
    person.box = box;

    // 2. crop + camera conditioning.
    auto tc = Clock::now();
    CropInputs crop = MakeCrop(rgb, W, H, box, K, kBodyPad);
    LogStage("crop", Ms(tc, Clock::now()));

    // 3. body regressor.
    auto tb = Clock::now();
    person.pred = s.body->Run(crop);
    LogStage("body-regressor", Ms(tb, Clock::now()));

    // 3b. hand-presence gate: sigmoid over the per-hand logits.
    bool any_hand = false;
    for (int h = 0; h < 2; ++h) {
      const float logit =
          std::max(person.pred.hand_logits[h][0], person.pred.hand_logits[h][1]);
      if (Sigmoid(logit) > p.hand_tau) any_hand = true;
    }
    person.has_hands = any_hand;

    // 3c. M7 hand refiner: when gated (and sam3dbody_hand.onnx is present), crop
    // each hand (pad 0.9, L/R flip), run the hand decoder, and merge the refined
    // hand PCA/scale/shape into person.pred.pred (wrist-angle + box-size gated).
    // The updated pred then flows through pose-corrective + MHR below, so the mesh
    // hands are re-solved. Body-only behavior is unchanged when the gate is off or
    // the hand model/asset is absent.
    if (any_hand) {
      auto th = Clock::now();
      const bool merged = s.RefineHands(person, rgb, W, H, K, p);
      LogStage("hand-refine", Ms(th, Clock::now()));
      (void)merged;
    }

    // 4. pose-corrective ONNX -> per-vertex offsets (cm, pre-flip).
    auto tp = Clock::now();
    std::array<float, 889> jp = s.mhr->JointParameters(person.pred.pred);
    std::vector<float> pc;
    const bool have_pc = s.RunPoseCorrective(jp, pc);
    LogStage("pose-corrective", Ms(tp, Clock::now()));

    // 5. MHR mesh (C++ FK+LBS), with the pose-corrective offsets injected.
    auto tm = Clock::now();
    person.mesh = s.mhr->Run(person.pred.pred, have_pc ? pc.data() : nullptr);
    LogStage("mhr-mesh", Ms(tm, Clock::now()));

    // 6. perspective camera solve (body default_scale = 1).
    person.cam = PerspectiveProjection(person.pred.pred_cam, box, kBodyPad, K,
                                        /*default_scale=*/1.f);

    result.people.push_back(std::move(person));
  }

  if (result.people.empty()) {
    last_error_ = "no person meshes produced";
    return result;
  }

  // --- 7. depth-ordered over-composite ------------------------------------
  // Paint far first, near last (near occludes far). cam_t.z is metric depth
  // (>0 in front); larger z = farther.
  std::vector<int> order(result.people.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return result.people[a].cam.cam_t[2] > result.people[b].cam.cam_t[2];
  });

  auto tr = Clock::now();
  std::vector<float>& acc = result.render.data;  // premultiplied accumulator
  for (int idx : order) {
    const PersonResult& person = result.people[idx];
    if (person.mesh.verts.empty()) continue;
    RgbaImage r = Render(person.mesh, person.cam, W, H, ropt);
    if (static_cast<int>(r.data.size()) != W * H * 4) continue;
    OverComposite(acc, r.data);  // r is premultiplied (ropt.premultiply=true)
  }
  LogStage("raster+composite", Ms(tr, Clock::now()));

  // acc is premultiplied. If the caller wants straight (un-associated) alpha,
  // un-premultiply the colour.
  if (!p.premultiply) {
    for (size_t i = 0; i + 3 < acc.size(); i += 4) {
      const float a = acc[i + 3];
      if (a > 1e-6f) {
        acc[i + 0] /= a;
        acc[i + 1] /= a;
        acc[i + 2] /= a;
      }
    }
  }

  last_error_.clear();
  return result;
}

}  // namespace hastur
