# Host compatibility, isolation & release engineering

> This document is ported from humbaba (the Depth Anything 3 OFX plugin) and describes
> the reused build/isolation/packaging scaffold. The ONNX-Runtime isolation machinery is
> shared verbatim; the SAM 3D Body specifics are flagged below.

## Hastur additions

SAM 3D Body differs from the single-model depth plugin in ways that matter for
host coexistence:

- **Multi-session VRAM budgeting.** The pipeline co-loads *several* ONNX sessions at once
  — a person detector plus the body ViT and hand ViT (two large transformers resident
  simultaneously), and a pose-corrective refiner. Their VRAM footprint stacks, so the
  arena/`gpu_mem_limit` caps and the "share GPU with host" policy (§2.4 below) must budget
  for the **sum** of co-resident sessions, not one model. Expect a substantially larger
  peak than the depth plugin's single ~2.4 GB session.
- **Software rasterizer dependency.** Recovering and compositing the MHR mesh back into the
  OFX stream needs a **CPU/GPU software rasterizer** (mesh → image). That is an added
  runtime dependency (and code surface) beyond ONNX Runtime — track its own isolation /
  symbol-visibility story so it does not leak or clash inside the host process.
- **CUDA runners must be re-registered.** The self-hosted GPU runner set (`rocky8-gpu`,
  `winbox-3090`) currently serves **humbaba** and has to be **re-registered to this repo**
  before `gpu-smoke` runs here. See `docs/CI_RUNNERS.md`.

---

This plugin bundles a large third-party runtime (ONNX Runtime) plus, on Linux/Windows,
the NVIDIA CUDA stack, and it runs **inside another application's process** (Natron,
Nuke, Resolve, Flame, …). That raises three coupled problems:

1. **Release/CI artifacts** — how to build, package and distribute per-platform bundles
   (and the multi-GB models) reproducibly.
2. **Host CUDA compatibility** — Foundry Nuke and others ship *their own* CUDA/cuDNN and
   sometimes their own inference runtimes; we must not fight them.
3. **Shared-library clashes** — if the host, or a sibling OFX plugin, loads a *different*
   build of a library we also load (above all `libonnxruntime`), the dynamic loader can
   bind the wrong one and crash. Namespacing/visibility/static-linking mitigate this.

---

## 1. CI/CD artifacts

### Where we are
- `ci.yml` builds all three platforms (macOS arm64 / Linux x64 / Windows x64) on
  GitHub-hosted runners and uploads the `.ofx.bundle` as a per-OS artifact. Those runners
  are **CPU-only**, so they validate *build + layout + that the CUDA EP is compiled in* —
  not GPU inference.
- `release.yml` (from M5) assembles a macOS `.pkg` on tags. It predates the Linux/Windows
  ports and hit the **2 GB GitHub asset limit** because it stuffed the ~1.3 GB models into
  a single bundle.

### Target design
**A. Three-platform release matrix.** One workflow, `matrix: [macos-14, ubuntu-latest,
windows-latest]`, each producing a *model-less* bundle + a native installer:

| Platform | Installer | Notes |
|---|---|---|
| macOS arm64 | `.pkg` (pkgbuild) | ad-hoc signed now; Developer ID + notarization later |
| Linux x64 | `.tar.xz` (+ optional `.deb`/`.rpm`) | installs to `/usr/OFX/Plugins` |
| Windows x64 | `.zip` + Inno Setup/NSIS `.exe` | installs to `%CommonProgramFiles%\OFX\Plugins` |

> **OFX plug-in search paths.** These install locations are the ones defined by the
> [OpenFX packaging spec](https://openfx.readthedocs.io/en/master/Reference/ofxPackaging.html)
> (see also the `OFX_PLUGIN_PATH` note in [`ofxCore.h`](https://github.com/AcademySoftwareFoundation/openfx/blob/main/include/ofxCore.h)).
> A compliant host searches, in order: the `OFX_PLUGIN_PATH` env var (colon-separated on
> Unix, semicolon on Windows), then the OS default — **Linux** `/usr/OFX/Plugins`, **macOS**
> `/Library/OFX/Plugins`, **Windows** `%CommonProgramFiles%\OFX\Plugins` (many hosts also
> add the per-user `~/OFX/Plugins` / `~/Library/OFX/Plugins` by convention). So dropping the
> `Sam3dBody.ofx.bundle` in the platform path above — or pointing `OFX_PLUGIN_PATH` at
> it — is all any host needs to find it. Verified for **Flame 2027** on Linux: its `flame`
> binary references both `/usr/OFX/Plugins` and `OFX_PLUGIN_PATH`, so a bundle in the former
> is discovered with no extra config.

**B. Decouple models from the plugin (fixes the 2 GB limit — task #12). _Implemented._**
The `.onnx` files are platform-independent and each is < 2 GB, so they ship as **separate
release assets** under a dedicated, independently-versioned tag (`models-v1`) — never inside
the code bundle. Each installer built by `release.yml` is **model-less** (a few hundred MB)
and carries `tools/fetch_models.{sh,ps1}`, a SHA-256-pinned downloader that pulls the models
into the installed bundle's `Contents/Resources/` (which the plugin already searches). It is
idempotent (skips models already present with the right checksum). Models are a *versioned
dependency*: bump to `models-v2` and update the pinned checksums only when a model actually
changes, so a plugin release never re-exports (and never ships) untested weights. A future
enhancement is an in-plugin first-run downloader to a per-user cache
(`~/.cache/hastur-ofx` / `%LOCALAPPDATA%\hastur-ofx`) for a zero-step experience.

> **Public distribution — resolved (2026-07-13).** The repo (`samhodge-tokgan/Hastur`) is now
> **public**, so the `models-v1` release assets are **anonymously downloadable** — the
> `*/releases/download/*` URLs return 200 without auth. `fetch_models.{sh,ps1}` therefore work
> for any end user with a plain `curl` (no `gh`, no `GITHUB_TOKEN`); the `gh` path and an optional
> `GITHUB_TOKEN` remain only as conveniences (nicer progress / higher API rate limits). The models
> (`models-v1`, pinned by SHA-256) are the single source of truth. A public **Hugging Face** mirror
> (MoGe already loads from HF) remains an *optional* alternative — point `HASTUR_MODELS_BASE_URL` at it
> — useful only if GitHub bandwidth/quota ever becomes a concern; it is no longer required for
> anonymous access.

**C. Real GPU validation via self-hosted runners.** GitHub-hosted runners have no NVIDIA
GPU. Register the Rocky 8 and Windows 11 boxes as **self-hosted runners**
(labels `self-hosted,linux,cuda` / `self-hosted,windows,cuda`) and add a `gpu-smoke` job
that runs `ort_check <model>` and the headless-Natron render, asserting depth parity
against a golden value (mean ≈ 0.7026). This is the only way to catch EP/driver regressions
in CI.

**D. Caching & speed.** Cache the ONNX Runtime download and the OpenFX clone
(`actions/cache` keyed on `HASTUR_ORT_VERSION` + OFX tag). The Linux build script's
`rm -rf build` (which re-downloads ORT) should be replaced by incremental builds in CI.

**E. Signing (later).** macOS: Developer ID + `notarytool`. Windows: Authenticode
(`signtool`) on the `.ofx`, DLLs and installer. Unsigned plugins trigger Gatekeeper /
SmartScreen friction. Deferred until certs exist.

---

## 2. Host CUDA compatibility (Nuke et al.)

OFX plugins load into the host process; there is **no process isolation**. The host may
already have a CUDA context, its own cuDNN/cuBLAS, and possibly its own ML runtime.

### Foundry Nuke specifics
- Nuke ships and initialises **its own CUDA runtime and cuDNN** for its GPU nodes; its
  Inference/CopyCat nodes use **libtorch** (Torch), *not* ONNX Runtime — so a direct
  `libonnxruntime` clash *with Nuke itself* is unlikely, but a clash with **another OFX
  plugin that bundles ORT** is very possible.
- Nuke pins a CUDA major per release. **Verify against the exact target build** (Foundry's
  release notes / "Third-Party Licenses"); as a rough guide, recent Nuke (15/16) is on
  **CUDA 12.x**, older (13/14) on **CUDA 11.x**. Our Linux/Windows builds target **CUDA 12**,
  so they line up with current Nuke and with a driver new enough for CUDA 12.

### The real risks in-process
1. **Two ONNX Runtimes, same SONAME** — the headline risk. See §3.
2. **CUDA runtime version skew** — CUDA is forward-compatible *at the driver level*: many
   `libcudart` versions coexist in one process as long as the installed **driver** supports
   the newest. So our CUDA-12 provider is fine next to a host on CUDA 11 **iff the machine's
   driver supports 12**. Mixing majors is still best avoided; prefer matching the host.
3. **cuDNN** — the major version is in the SONAME (`libcudnn.so.9` vs `.8`), so cuDNN 8 and
   9 *coexist*. Two different cuDNN **9.x** builds share `libcudnn.so.9` and can clash —
   isolate ours (§3) or rely on the host's if compatible.
4. **VRAM contention** — the host manages the GPU; our ORT allocates independently and can
   OOM a busy comp. Mitigate: bound ORT with an arena/`gpu_mem_limit`-style cap, expose a
   "share GPU with host" resolution/precision knob (we already gate by proc resolution),
   and prefer `cudaMallocAsync`/arena-shrink.
5. **Driver/context init order** — let ORT create the context lazily; never call
   `cudaSetDevice`/reset from the plugin in a way that disturbs the host's context.

### Recommendation
Build **one artifact per CUDA major** that matters (start with CUDA 12; add a CUDA-11
variant only if a supported host needs it), document the host↔CUDA matrix, and test against
an actual **Nuke Non-commercial/Indie** install before claiming support. Keep GPU memory
polite by default.

> **Status (2026-07-13):** verified in **Nuke 16.0v8 on macOS** (arm64/CoreML) — all three
> plugins load and run, coexisting with Nuke's own libtorch stack (no shared-lib clash). The
> CUDA host-coexistence path (Nuke on Linux/Windows sharing cuDNN/CUDA with our CUDA-EP ORT)
> is the one still to validate against a real Nuke install. See `BACKLOG.md`.

---

## 3. Shared-library clashes: namespacing vs static linking

### Why it happens
The dynamic loader deduplicates libraries **by name**: SONAME on Linux, install-name on
macOS, **base file name** on Windows. If the host (or a sibling plugin) has already loaded
`libonnxruntime.so.1` / `onnxruntime.dll` from a *different* ORT version, our `NEEDED`/
delay-load reference to the same name is satisfied by **their** already-loaded module — an
ABI mismatch that crashes on the first differing symbol. Symbol **interposition**
(especially Linux `RTLD_GLOBAL`) is the second half of the problem: whoever is loaded first
can capture calls meant for the other.

### Options, least → most invasive

**(D0) Export only the OFX entry points — do this unconditionally.** Build the `.ofx` so
that *nothing but* `OfxGetNumberOfPlugins`/`OfxGetPlugin`/`OfxSetHost` is visible. Cheap,
and it stops *our* (and any statically-linked deps') symbols from interposing on the host
or siblings.
- Linux: `-fvisibility=hidden -fvisibility-inlines-hidden` + a version script
  (`{ global: OfxGetNumberOfPlugins; OfxGetPlugin; OfxSetHost; local: *; };`).
- macOS: `-fvisibility=hidden` + `-exported_symbols_list` with the three symbols.
- Windows: already the case — only `__declspec(dllexport)` (the OFX macros) are exported.

**(B) Give our bundled ORT a private name — recommended, decisive, modest effort.** Rename
the bundled runtime so it can **never** be deduplicated against the host's:
- Linux: `patchelf --set-soname libonnxruntime_hastur.so.1` on the bundled `.so`, and
  `patchelf --replace-needed libonnxruntime.so.1 libonnxruntime_hastur.so.1` on the `.ofx`
  (post-build). ORT loads its `onnxruntime_providers_*.so` itself, from its own directory,
  with local scope — their names are unusual enough to rarely clash, but they can be renamed
  the same way if needed.
- macOS: `install_name_tool -id @rpath/libonnxruntime_hastur.1.dylib` on the dylib +
  `-change` on the `.ofx`. (Two-level namespace already helps; this makes it airtight.)
- Windows: **required**, and the hardest — Windows dedups by base name, so even our
  full-path `LoadLibraryExW` returns the host's `onnxruntime.dll` if one is loaded. Rename
  the DLL to `onnxruntime_hastur.dll` and patch the `.ofx` import/delay-load name to match
  (the import descriptor string must be rewritten — a small PE-patch step in the build, or
  build ORT with a custom output name).

**(C) Static-link ONNX Runtime — most bulletproof, highest cost.** Build ORT
`--build_shared_lib OFF` and link it into the `.ofx`; there is then no shared `libonnxruntime`
to clash. Must be paired with (D0) so ORT's vendored deps (protobuf, abseil, onnx,
flatbuffers, re2, …) don't leak and collide with a sibling that also static-links ORT.
Costs: building ORT from source in CI (long, CUDA-heavy), a large `.ofx`, and the CUDA
provider still pulls dynamic `cudart`/`cublas`/`cudnn` (NVIDIA libs can't be practically
statically isolated). Reserve for when a real clash is observed and (B)+(D0) aren't enough.

### Recommendation
Adopt **(D0) + (B)** now: hide all non-OFX symbols and privately rename the bundled ORT on
all three platforms. That eliminates the two most likely field failures (ORT dedup + symbol
interposition) for a few post-build `patchelf`/`install_name_tool`/PE-patch steps, without
the cost of a from-source static ORT. Keep **(C)** as the escalation if a host or sibling is
found to still collide. Add a CI **"co-load" test**: a tiny OFX host harness that loads our
plugin *alongside* a second module linking a different `libonnxruntime`, asserting both load
and run — the only way to prove isolation works.

### Implementation status
- **(D0) done, all platforms.** The `.ofx` exports only `OfxGetNumberOfPlugins`/`OfxGetPlugin`
  (macOS: `-exported_symbols_list`; Linux: `--version-script`; Windows: already the case).
  Verified on macOS: **790 → 2** exported symbols, plugin still renders in Natron.
- **(B) done on macOS + Linux.** The bundled ORT is renamed to `libonnxruntime_hastur.*`
  (macOS `install_name_tool -id`/`-change` + re-sign; Linux `patchelf --set-soname` +
  `--replace-needed` on the `.ofx` and the provider `.so`s). Verified on macOS end-to-end
  (Natron render passes with the renamed runtime); Linux is exercised by the `build-linux`
  CI check (needs `patchelf`).
- **(B) deferred on Windows.** Renaming the DLL there also requires **PE-import patching of
  the provider DLLs** (they import from `onnxruntime.dll` by name), which is more involved.
  The delay-load hook already loads *our* `onnxruntime.dll` by full path, which is sufficient
  when no other ORT is present in the host (e.g. Nuke, which uses libtorch). Full Windows
  isolation (rename + provider import-table patch, or `AddDllDirectory` in a private
  namespace) is tracked as follow-up.
- **Not yet done:** the "co-load" clash test harness.

### Don't bundle the CUDA runtime
`cudart`/`cublas`/`cudnn` have fixed NVIDIA SONAMEs we can't rename, and CUDA context is
process-global. Isolation tricks don't really help there, so **rely on the host/system CUDA**
(matched by major version) rather than shipping our own — which is also why cuDNN is a
documented host prerequisite (see `docs/LINUX.md`, `docs/WINDOWS.md`).
