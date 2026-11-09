// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "HandRefinerEngine.h"
#include "OrtModelLoad.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "OrtAccel.h"

namespace hastur {

namespace {

const char* ComputeUnitsString(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CpuAndGpu: return "CPUAndGPU";
    case ComputeUnits::CpuAndAne: return "CPUAndNeuralEngine";
    case ComputeUnits::CpuOnly: return "CPUOnly";
  }
  return "ALL";
}

constexpr int64_t kImg = kBodyImageSize;   // 512
constexpr int64_t kTok = kTokenGrid;       // 32
constexpr int64_t kParam = kParamDim;      // 519

// IEEE-754 binary16 <-> float (identical to Sam3dBodyEngine; kept local to keep
// the engines independent).
uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) {
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
  }
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half_mant = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) half_mant += 1;
    return static_cast<uint16_t>(sign | half_mant);
  }
  uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                        (mant >> 13));
  if (mant & 0x1000u) half += 1;
  return half;
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1F;
  const uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      int e = -1;
      uint32_t m = mant;
      do { m <<= 1; ++e; } while (!(m & 0x400u));
      m &= 0x3FFu;
      out = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

std::vector<uint16_t> ToHalf(const float* src, size_t n) {
  std::vector<uint16_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = FloatToHalf(src[i]);
  return out;
}

bool CopyOutput(const std::vector<std::string>& names,
                const std::vector<Ort::Value>& vals, const char* name,
                float* dst, size_t count) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] != name) continue;
    if (!vals[i].IsTensor()) return false;
    if (vals[i].GetTensorTypeAndShapeInfo().GetElementCount() < count) return false;
    const uint16_t* p = vals[i].GetTensorData<uint16_t>();  // fp16 payload
    for (size_t k = 0; k < count; ++k) dst[k] = HalfToFloat(p[k]);
    return true;
  }
  return false;
}

// Session on the platform accelerator (CoreML NeuralNetwork format: these graphs
// share the interleaved MHR-refinement fallback ops, so the fast build that
// offloads the supported majority beats MLProgram), CPU fallback. `cpu_env` set
// forces a plain CPU session. Returns null and sets `err` on failure.
std::unique_ptr<Ort::Session> MakeAccelSession(Ort::Env& env, const std::string& path,
                                               ComputeUnits units, int intra_threads,
                                               const char* cpu_env, bool& accel,
                                               std::string& err) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);
  accel = false;
  if (AcceleratorAvailable() && !std::getenv(cpu_env)) {
    try {
      AppendAccelerator(so, ComputeUnitsString(units), /*coreml_static=*/true,
                        /*coreml_mlprogram=*/false);
      accel = true;
    } catch (const Ort::Exception& e) {
      err = std::string(AcceleratorSubstr()) + " EP append failed: " + e.what();
    }
  }
  try {
    return hastur::MakeModelSession(env, path, so);
  } catch (const Ort::Exception& e) {
    if (!accel) {
      err = std::string("session create failed: ") + e.what();
      return nullptr;
    }
    err = std::string(AcceleratorSubstr()) +
          " session create failed (falling back to CPU): " + e.what();
  }
  Ort::SessionOptions cpu_so;
  cpu_so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) cpu_so.SetIntraOpNumThreads(intra_threads);
  accel = false;
  try {
    return hastur::MakeModelSession(env, path, cpu_so);
  } catch (const Ort::Exception& e) {
    err = std::string("session create failed: ") + e.what();
    return nullptr;
  }
}

std::vector<std::string> SessionNames(Ort::Session& s, bool inputs) {
  Ort::AllocatorWithDefaultOptions alloc;
  std::vector<std::string> out;
  const size_t n = inputs ? s.GetInputCount() : s.GetOutputCount();
  for (size_t i = 0; i < n; ++i) {
    Ort::AllocatedStringPtr p = inputs ? s.GetInputNameAllocated(i, alloc)
                                       : s.GetOutputNameAllocated(i, alloc);
    out.emplace_back(p.get());
  }
  return out;
}

}  // namespace

struct HandRefinerEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "HandRefiner"};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool accel_active = false;
};

HandRefinerEngine::HandRefinerEngine(const std::string& model_path,
                                     ComputeUnits units, int intra_threads)
    : impl_(std::make_unique<Impl>()) {
  // HASTUR_HAND_CPU=1 forces a plain CPU session (mirrors HASTUR_BODY_CPU).
  impl_->session = MakeAccelSession(impl_->env, model_path, units, intra_threads,
                                    "HASTUR_HAND_CPU", impl_->accel_active, last_error_);
  if (!impl_->session) return;
  impl_->input_names = SessionNames(*impl_->session, true);
  impl_->output_names = SessionNames(*impl_->session, false);
}

HandRefinerEngine::~HandRefinerEngine() = default;

bool HandRefinerEngine::ok() const { return impl_ && impl_->session != nullptr; }

bool HandRefinerEngine::accelerator_active() const {
  return impl_ && impl_->accel_active;
}

bool HandRefinerEngine::AcceleratorAvailable() {
  return hastur::AcceleratorAvailable();
}

HandPrediction HandRefinerEngine::Run(const CropInputs& in) {
  HandPrediction out{};
  if (!impl_ || !impl_->session) {
    last_error_ = "session not initialized";
    return out;
  }
  if (in.image.size() != static_cast<size_t>(3 * kImg * kImg)) {
    last_error_ = "image must be 3*512*512 floats";
    return out;
  }

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const std::array<int64_t, 4> img_shape{1, 3, kImg, kImg};
  const std::array<int64_t, 4> ray_shape{1, 2, kTok, kTok};
  const std::array<int64_t, 2> cond_shape{1, 3};

  std::vector<uint16_t> image = ToHalf(in.image.data(), in.image.size());
  std::vector<uint16_t> ray_cond = ToHalf(in.ray_cond.data(), in.ray_cond.size());
  std::vector<uint16_t> cond = ToHalf(in.condition_info.data(), in.condition_info.size());

  auto make_half = [&](std::vector<uint16_t>& buf, const int64_t* shape,
                       size_t rank) {
    return Ort::Value::CreateTensor(
        mem, buf.data(), buf.size() * sizeof(uint16_t), shape, rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  };
  Ort::Value t_image = make_half(image, img_shape.data(), img_shape.size());
  Ort::Value t_ray = make_half(ray_cond, ray_shape.data(), ray_shape.size());
  Ort::Value t_cond = make_half(cond, cond_shape.data(), cond_shape.size());

  std::vector<const char*> in_names;
  std::vector<Ort::Value> in_vals;
  in_names.reserve(impl_->input_names.size());
  for (const std::string& nm : impl_->input_names) {
    in_names.push_back(nm.c_str());
    if (nm == "image") in_vals.emplace_back(std::move(t_image));
    else if (nm == "ray_cond") in_vals.emplace_back(std::move(t_ray));
    else if (nm == "condition_info") in_vals.emplace_back(std::move(t_cond));
    else {
      last_error_ = "unexpected graph input: " + nm;
      return out;
    }
  }

  std::vector<const char*> out_names;
  for (const std::string& nm : impl_->output_names) out_names.push_back(nm.c_str());

  std::vector<Ort::Value> results;
  try {
    results = impl_->session->Run(Ort::RunOptions{nullptr}, in_names.data(),
                                  in_vals.data(), in_vals.size(),
                                  out_names.data(), out_names.size());
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("inference failed: ") + e.what();
    return out;
  }

  bool ok = true;
  ok &= CopyOutput(impl_->output_names, results, "pred", out.pred.data(), kParam);
  // wrist_global[1,2,3,3] row-major -> [L,R] x 9.
  ok &= CopyOutput(impl_->output_names, results, "wrist_global",
                   out.wrist_global[0].data(), 18);
  if (!ok) {
    last_error_ = "missing/short output tensor";
    return out;
  }
  out.has_wrist_kp2d = CopyOutput(impl_->output_names, results, "wrist_kp2d",
                                  out.wrist_kp2d.data(), 2);
  out.valid = true;
  return out;
}

// ---------------------------------------------------------------------------
struct KeypointPromptEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "KeypointPrompt"};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool accel_active = false;
};

KeypointPromptEngine::KeypointPromptEngine(const std::string& model_path,
                                           ComputeUnits units, int intra_threads)
    : impl_(std::make_unique<Impl>()) {
  impl_->session = MakeAccelSession(impl_->env, model_path, units, intra_threads,
                                    "HASTUR_PROMPT_CPU", impl_->accel_active, last_error_);
  if (!impl_->session) return;
  impl_->input_names = SessionNames(*impl_->session, true);
  impl_->output_names = SessionNames(*impl_->session, false);
}

KeypointPromptEngine::~KeypointPromptEngine() = default;

bool KeypointPromptEngine::ok() const { return impl_ && impl_->session != nullptr; }

PromptRerun KeypointPromptEngine::Run(const std::vector<float>& image_embeddings,
                                      const CropInputs& body_crop,
                                      const std::array<float, 3 * kNumPrompts>& keypoints,
                                      const std::array<float, kParamDim>& pred,
                                      const std::array<float, 3>& pred_cam) {
  PromptRerun out{};
  if (!ok()) {
    last_error_ = "session not initialized";
    return out;
  }
  const size_t emb_n = static_cast<size_t>(kEmbedDim) * kTok * kTok;
  if (image_embeddings.size() != emb_n) {
    last_error_ = "image_embeddings must be 1280*32*32 floats";
    return out;
  }
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<float, kParam + 3> prev{};
  std::copy(pred.begin(), pred.end(), prev.begin());
  std::copy(pred_cam.begin(), pred_cam.end(), prev.begin() + kParam);

  std::vector<uint16_t> emb = ToHalf(image_embeddings.data(), emb_n);
  std::vector<uint16_t> ray = ToHalf(body_crop.ray_cond.data(), body_crop.ray_cond.size());
  std::vector<uint16_t> cond = ToHalf(body_crop.condition_info.data(),
                                      body_crop.condition_info.size());
  std::vector<uint16_t> kp = ToHalf(keypoints.data(), keypoints.size());
  std::vector<uint16_t> pv = ToHalf(prev.data(), prev.size());
  const std::array<int64_t, 4> emb_shape{1, kEmbedDim, kTok, kTok};
  const std::array<int64_t, 4> ray_shape{1, 2, kTok, kTok};
  const std::array<int64_t, 2> cond_shape{1, 3};
  const std::array<int64_t, 3> kp_shape{1, kNumPrompts, 3};
  const std::array<int64_t, 2> prev_shape{1, kParam + 3};
  auto half = [&](std::vector<uint16_t>& b, const int64_t* shape, size_t rank) {
    return Ort::Value::CreateTensor(mem, b.data(), b.size() * sizeof(uint16_t), shape,
                                    rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  };

  std::vector<const char*> in_names;
  std::vector<Ort::Value> in_vals;
  for (const std::string& nm : impl_->input_names) {
    in_names.push_back(nm.c_str());
    if (nm == "image_embeddings") in_vals.emplace_back(half(emb, emb_shape.data(), 4));
    else if (nm == "ray_cond") in_vals.emplace_back(half(ray, ray_shape.data(), 4));
    else if (nm == "condition_info") in_vals.emplace_back(half(cond, cond_shape.data(), 2));
    else if (nm == "keypoints") in_vals.emplace_back(half(kp, kp_shape.data(), 3));
    else if (nm == "prev_estimate") in_vals.emplace_back(half(pv, prev_shape.data(), 2));
    else {
      last_error_ = "unexpected graph input: " + nm;
      return out;
    }
  }
  std::vector<const char*> out_names;
  for (const std::string& nm : impl_->output_names) out_names.push_back(nm.c_str());
  std::vector<Ort::Value> results;
  try {
    results = impl_->session->Run(Ort::RunOptions{nullptr}, in_names.data(),
                                  in_vals.data(), in_vals.size(), out_names.data(),
                                  out_names.size());
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("inference failed: ") + e.what();
    return out;
  }
  bool good = CopyOutput(impl_->output_names, results, "pred", out.pred.data(), kParam);
  good &= CopyOutput(impl_->output_names, results, "pred_cam", out.pred_cam.data(), 3);
  if (!good) {
    last_error_ = "missing/short output tensor";
    return out;
  }
  out.valid = true;
  return out;
}

}  // namespace hastur
