// Prints the available ONNX Runtime execution providers and confirms the
// platform accelerator EP (CoreML on macOS, CUDA on Linux/Windows) is present.
//
// Usage:
//   ort_check                 -> list providers; exit 0 if accelerator present
//   ort_check <model.onnx>    -> also load the model on the accelerator and run
//                                one inference, reporting timing + output stats.
//
// NOTE: this is a reusable EP smoke tool ported verbatim from humbaba. The dummy
// input shape below is hard-coded to a 504x504 NCHW RGB image (the depth-model
// convention). TODO(hastur): the SAM 3D Body detector/body/hand ViTs take different
// input shapes — adapt the fallback shape (or infer it per model input) when wiring
// the model smoke path into gpu-smoke. It compiles and runs the provider check as-is.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#if defined(__APPLE__)
static const char* kAccel = "CoreML";
#else
static const char* kAccel = "CUDA";
#endif

static bool AppendAccel(Ort::SessionOptions& so) {
#if defined(__APPLE__)
  // Static input shapes are required for CoreML to compile a dynamic-resolution ViT
  // graph (its internal reshape otherwise trips an axis-out-of-range at MLProgram
  // build time). Mirrors the engine's session options.
  std::unordered_map<std::string, std::string> opts = {
      {"MLComputeUnits", "ALL"},
      {"ModelFormat", "MLProgram"},
      {"RequireStaticInputShapes", "1"}};
  so.AppendExecutionProvider("CoreML", opts);
#else
  OrtCUDAProviderOptions cuda{};
  cuda.device_id = 0;
  so.AppendExecutionProvider_CUDA(cuda);
#endif
  return true;
}

int main(int argc, char** argv) {
  bool accel = false;
  std::cout << "ONNX Runtime available execution providers:\n";
  for (const auto& p : Ort::GetAvailableProviders()) {
    std::cout << "  - " << p << "\n";
    if (p.find(kAccel) != std::string::npos) accel = true;
  }
  std::cout << kAccel << " EP available: " << (accel ? "YES" : "NO") << "\n";

  if (argc < 2) return accel ? 0 : 1;

  // Model smoke test on the accelerator.
  const std::string model = argv[1];
  std::cout << "\nLoading model on " << kAccel << ": " << model << "\n";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_check");
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  bool on_accel = false;
  try {
    AppendAccel(so);
    on_accel = true;
  } catch (const Ort::Exception& e) {
    std::cout << "  accelerator append failed, using CPU: " << e.what() << "\n";
  }

  std::unique_ptr<Ort::Session> sess;
  try {
    // Ort::Session wants const ORTCHAR_T* (wchar_t on Windows); filesystem::path
    // ::c_str() yields the native char type on each platform.
    std::filesystem::path model_path(model);
    sess = std::make_unique<Ort::Session>(env, model_path.c_str(), so);
  } catch (const Ort::Exception& e) {
    std::cout << "  session create FAILED: " << e.what() << "\n";
    return 2;
  }
  std::cout << "  session created (" << (on_accel ? kAccel : "CPU") << ")\n";

  Ort::AllocatorWithDefaultOptions alloc;
  Ort::AllocatedStringPtr in_name = sess->GetInputNameAllocated(0, alloc);
  auto in_info = sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shp = in_info.GetShape();
  // Dynamo-exported graphs often expose no static rank -> GetShape() is empty.
  // TODO(hastur): 504x504 is the depth-model convention; adjust for SAM 3D Body inputs.
  if (shp.size() != 4) shp = {1, 3, 504, 504};
  shp[0] = 1;
  shp[1] = 3;
  shp[2] = 504;
  shp[3] = 504;
  size_t n = 1;
  for (auto d : shp) n *= static_cast<size_t>(d ? d : 1);
  std::vector<float> input(n, 0.5f);
  for (size_t i = 0; i < n; ++i) input[i] = 0.3f + 0.4f * float(i % 97) / 97.0f;

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in_t = Ort::Value::CreateTensor<float>(mem, input.data(), n, shp.data(), shp.size());

  size_t nout = sess->GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> out_holders;
  std::vector<const char*> out_names;
  for (size_t i = 0; i < nout; ++i) {
    out_holders.push_back(sess->GetOutputNameAllocated(i, alloc));
    out_names.push_back(out_holders.back().get());
  }
  const char* in_names[] = {in_name.get()};

  // Warm-up (kernel/graph compile), then timed run.
  try {
    sess->Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1, out_names.data(), nout);
  } catch (const Ort::Exception& e) {
    std::cout << "  warm-up run FAILED: " << e.what() << "\n";
    return 3;
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  auto outs = sess->Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1, out_names.data(), nout);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  auto& d = outs[0];
  auto oi = d.GetTensorTypeAndShapeInfo();
  auto os = oi.GetShape();
  size_t on = 1;
  for (auto v : os) on *= static_cast<size_t>(v);
  const float* od = d.GetTensorData<float>();
  float mn = od[0], mx = od[0];
  double sum = 0;
  for (size_t i = 0; i < on; ++i) { mn = std::min(mn, od[i]); mx = std::max(mx, od[i]); sum += od[i]; }

  std::cout << "  inference OK: " << ms << " ms\n";
  std::cout << "  output[0] shape: [";
  for (size_t i = 0; i < os.size(); ++i) std::cout << os[i] << (i + 1 < os.size() ? "," : "");
  std::cout << "]  min=" << mn << " max=" << mx << " mean=" << (sum / on) << "\n";
  return 0;
}
