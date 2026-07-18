# Linux (x86-64, CUDA) build & deployment

The plugin builds as a native `x86-64` `.ofx.bundle` that runs SAM 3D Body on
**NVIDIA CUDA** via ONNX Runtime. Verified on **Rocky Linux 8** with two RTX 3090s
(CUDA 12.6, cuDNN 9). Accelerator selection is automatic — CoreML on macOS, CUDA on
Linux — through [`src/OrtAccel.h`](../src/OrtAccel.h); if CUDA (or its runtime libs)
is unavailable the engines **fall back to CPU** with a message in `getLastError()`.

## Build prerequisites

- A C++17 compiler with `GLIBCXX_3.4.21+` (on Rocky 8 use **gcc-toolset-12**:
  `source /opt/rh/gcc-toolset-12/enable`). The stock RHEL/Rocky 8 gcc 8 is too old
  for some dependencies; 12 is what we build and test with.
- CMake ≥ 3.24.
- **CUDA Toolkit 12.x** and **cuDNN 9** installed (for the ONNX Runtime CUDA package
  the build fetches: `onnxruntime-linux-x64-gpu_cuda12`).

```bash
source /opt/rh/gcc-toolset-12/enable
cmake -S . -B build -DHASTUR_WITH_ONNX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ort_check Sam3dBody -j"$(nproc)"
```

The build assembles `build/Sam3dBody.ofx.bundle` with this layout:

```
Sam3dBody.ofx.bundle/
  Contents/
    Linux-x86-64/                     # <-- OFX spec arch dir: HYPHENS, not "Linux-x86_64"
      Sam3dBody.ofx              # RUNPATH=$ORIGIN
      libonnxruntime.so.1
      libonnxruntime_providers_cuda.so
      libonnxruntime_providers_tensorrt.so
      libonnxruntime_providers_shared.so
    Resources/                        # optional: bundled *.onnx models (release build)
```

> **Gotcha (cost us a debugging session):** the OFX packaging spec names the 64-bit
> Linux arch directory **`Linux-x86-64`** (hyphens). If you use `Linux-x86_64`
> (underscore) the binary is valid and `dlopen`s fine, but OFX hosts (Natron)
> **silently skip it** because they look for the binary under `Linux-x86-64/`.

## Runtime requirements (deployment)

The bundle ships ONNX Runtime + its CUDA/TensorRT provider libs (found via `$ORIGIN`),
but **not** the CUDA runtime itself — that would add ~1 GB of CUDA libs plus cuDNN's
multi-GB sublibraries. The host must provide, and the loader must be able to find:

- an **NVIDIA driver** (`libcuda.so.1`),
- **CUDA 12.x runtime**: `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`,
  `libcufft.so.11`, `libcurand.so.10`, `libnvrtc.so.12`,
- **cuDNN 9**: `libcudnn.so.9`.

Make them discoverable one of these ways:

```bash
# per-session
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib64:$LD_LIBRARY_PATH"
# or system-wide
echo /usr/local/cuda-12.6/lib64 | sudo tee /etc/ld.so.conf.d/cuda.conf && sudo ldconfig
```

If these are missing the plugin still loads and runs — the depth engine falls back to
the CPU execution provider (much slower for the ~1.3 GB HASTUR model). MoGe/AnyCalib
attempt CUDA first and fall back to CPU as well.

## Install location

Copy the bundle to any OFX plugin path. Both of these are scanned by OFX hosts:

```bash
cp -r build/Sam3dBody.ofx.bundle ~/OFX/Plugins/          # per-user
sudo cp -r build/Sam3dBody.ofx.bundle /usr/OFX/Plugins/  # system-wide
# or point OFX_PLUGIN_PATH at a directory containing the bundle:
export OFX_PLUGIN_PATH="$HOME/OFX/Plugins"
```

`cmake --install build` installs to `~/OFX/Plugins` by default.

### Models

Release installers are model-less (to stay under GitHub's 2 GB asset limit). Pull the
models into the installed bundle's `Contents/Resources/` with the bundled fetch script
(SHA-256-verified, idempotent):

```bash
# auto-detects the installed bundle; use sudo for a system-wide install dir
tools/fetch_models.sh            # or: sudo tools/fetch_models.sh
```

Or point the plugin's `modelFile` param at an existing `.onnx`.

## Headless verification (Natron)

Any OFX host works; we test with the portable Linux Natron
(`Natron-2.5.0-Linux-x86_64-no-installer`). `NatronRenderer` runs without a display.

```bash
export OFX_PLUGIN_PATH="$HOME/OFX/Plugins"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/lib64:$LD_LIBRARY_PATH"
export HASTUR_INPUT=test-assets/synthetic_test.png
export HASTUR_OUTPUT=/tmp/depth.exr
export HASTUR_MODEL_DIR="$HOME/hastur-models/sam3dbody_body.onnx"  # or bundle it in Resources/
export HASTUR_ACESCG=0 HASTUR_PROCRES=504
NatronRenderer --clear-openfx-cache -t tests/natron/render_depth.py < /dev/null
```

Expect `RESULT: PASS`, an EXR the **same size as the input**, and metric depth values
(decimeters). Watch the GPU with `nvidia-smi` during the run — the CUDA session uses
~2.4 GB VRAM for HASTUR at 504×504.

### Quick GPU smoke test (no host)

`ort_check` doubles as a standalone accelerator test:

```bash
LD_LIBRARY_PATH="build/Sam3dBody.ofx.bundle/Contents/Linux-x86-64:\
/usr/local/cuda-12.6/lib64:/usr/lib64" \
  ./build/ort_check ~/hastur-models/sam3dbody_body.onnx
# -> lists providers, creates a CUDA session, runs one inference, prints timing.
# Reference: 504x504 HASTUR ~72 ms, GPU ~55%, ~2.4 GB VRAM on an RTX 3090.
```
