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

    // 3b. hand-presence gate (M7 hook): sigmoid over the per-hand logits. We
    // record it but DO NOT run the hand refiner in M4.
    // TODO(M7): if any gate > hand_tau, run the hand crop + sam3dbody_hand.onnx
    // refiner and fuse the wrist-centric hand pose into the MHR params.
    bool any_hand = false;
    for (int h = 0; h < 2; ++h) {
      const float logit =
          std::max(person.pred.hand_logits[h][0], person.pred.hand_logits[h][1]);
      if (Sigmoid(logit) > p.hand_tau) any_hand = true;
    }
    person.has_hands = any_hand;  // gate only; refiner deferred to M7

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
