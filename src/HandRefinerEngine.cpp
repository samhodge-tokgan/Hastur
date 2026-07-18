// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "HandRefinerEngine.h"

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

}  // namespace

struct HandRefinerEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "HandRefiner"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions alloc;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool accel_active = false;
};

HandRefinerEngine::HandRefinerEngine(const std::string& model_path,
                                     ComputeUnits units, int intra_threads)
    : impl_(std::make_unique<Impl>()) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);

  bool used_accel = false;
  // HASTUR_HAND_CPU=1 forces a plain CPU session (mirrors HASTUR_BODY_CPU).
  if (AcceleratorAvailable() && !std::getenv("HASTUR_HAND_CPU")) {
    try {
      // Same rationale as the body engine: the hand graph shares the interleaved
      // MHR-refinement fallback ops, so use the CoreML NeuralNetwork format (fast
      // build, offloads the supported majority) rather than MLProgram.
      AppendAccelerator(so, ComputeUnitsString(units), /*coreml_static=*/true,
                        /*coreml_mlprogram=*/false);
      used_accel = true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string(AcceleratorSubstr()) + " EP append failed: " + e.what();
      used_accel = false;
    }
  }

  try {
    impl_->session =
        std::make_unique<Ort::Session>(impl_->env, OrtPath(model_path).c_str(), so);
  } catch (const Ort::Exception& e) {
    if (used_accel) {
      last_error_ = std::string(AcceleratorSubstr()) +
                    " session create failed (falling back to CPU): " + e.what();
      Ort::SessionOptions cpu_so;
      cpu_so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      if (intra_threads > 0) cpu_so.SetIntraOpNumThreads(intra_threads);
      used_accel = false;
      try {
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, OrtPath(model_path).c_str(), cpu_so);
      } catch (const Ort::Exception& e2) {
        last_error_ = std::string("session create failed: ") + e2.what();
        impl_->session.reset();
        return;
      }
    } else {
      last_error_ = std::string("session create failed: ") + e.what();
      impl_->session.reset();
      return;
    }
  }
  impl_->accel_active = used_accel;

  const size_t nin = impl_->session->GetInputCount();
  for (size_t i = 0; i < nin; ++i) {
    Ort::AllocatedStringPtr n = impl_->session->GetInputNameAllocated(i, impl_->alloc);
    impl_->input_names.emplace_back(n.get());
  }
  const size_t nout = impl_->session->GetOutputCount();
  for (size_t i = 0; i < nout; ++i) {
    Ort::AllocatedStringPtr n = impl_->session->GetOutputNameAllocated(i, impl_->alloc);
    impl_->output_names.emplace_back(n.get());
  }
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
  out.valid = true;
  return out;
}

}  // namespace hastur
