# Self-hosted GPU CI runners

GitHub-hosted runners have **no GPU**, so the `gpu-smoke` job in `.github/workflows/ci.yml`
(the only job that exercises real CUDA) runs on a **self-hosted** runner labelled `cuda`. The
job branches on `runner.os`, so the `cuda` label may be served by either a Windows or a Linux
box — whichever GPU host is currently up.

> **These runners currently serve humbaba.** The two boxes described below (`rocky8-gpu`,
> `winbox-3090`) are registered against `samhodge-tokgan/humbaba` today. They must be
> **added to this repo** (`samhodge-tokgan/Hastur`) — a runner can be registered to multiple
> repos, or moved — before `gpu-smoke` will run here. Use the registration commands at the
> bottom of this doc, substituting the Hastur repo. Only one GPU host is up at a time (they
> share the passed-through GPU).

> **gpu-smoke is a placeholder (M0).** SAM 3D Body outputs a **mesh**, not a depth map, so
> humbaba's cross-platform *depth-mean parity* golden (mean ≈ 0.7026) does **not** apply. The
> job currently just builds and runs `ort_check` on the GPU host. It is replaced with a
> geometric invariant check on a recovered mesh — `TODO(M9): golden_invariant.py` — when the
> inference stages land. The pre-staged model table below is retained for that milestone.

The model is **pre-staged** on the runner (not downloaded per run):

| OS      | Default model path                              | Override repo/runner var |
|---------|-------------------------------------------------|--------------------------|
| Windows | `C:\dev\hastur-models\sam3dbody_body.onnx`     | `HASTUR_MODEL_WIN`          |
| Linux   | `/opt/hastur-models/sam3dbody_body.onnx`       | `HASTUR_MODEL_LINUX`        |

---

## Windows runner (verified: `winbox-3090`, 2× RTX 3090, CUDA 12.8, cuDNN 9)

Install as a **Windows service** so it survives the SSH/RDP session ending — an
interactively launched `run.cmd` is killed when its parent session closes.

```powershell
# In C:\actions-runner (unzip the runner package there first):
#   Register AND install the service in one shot. --runasservice installs under
#   NT AUTHORITY\NETWORK SERVICE (no password needed) and auto-starts on boot.
.\config.cmd --url https://github.com/samhodge-tokgan/Hastur `
  --token <REG_TOKEN> --name winbox-3090 --labels cuda,gpu --runasservice --unattended --replace
```

### Gotchas discovered (all now handled)

1. **Clock skew breaks registration.** The VM booted ~1 h behind real time; the runner's
   session token is time-sensitive and the runner stayed *offline* until synced:
   ```powershell
   w32tm /resync /force
   ```
2. **The service account has a minimal environment** — three things the interactive shell
   had but `NETWORK SERVICE` did **not**:
   - **No `pwsh` (PowerShell 7)** on PATH → the `gpu-smoke` steps use `shell: powershell`
     (Windows PowerShell 5.1, always in `System32`), *not* `shell: pwsh`. All script
     constructs used (`Out-String`, `[regex]::Match`, `[math]::Abs`, `throw`) are 5.1-safe.
   - **Execution policy `Restricted`** → GitHub's generated temp `.ps1` is refused with
     `PSSecurityException / UnauthorizedAccess`. Fix machine-wide (covers `NETWORK SERVICE`):
     ```powershell
     Set-ExecutionPolicy -Scope LocalMachine -ExecutionPolicy RemoteSigned -Force
     ```
   - **`cmake` not on any PATH scope** (installed at `C:\Program Files\CMake\bin`). Add it to
     the **Machine** PATH, then **restart the service** so it re-reads the environment:
     ```powershell
     [Environment]::SetEnvironmentVariable('Path',
       ([Environment]::GetEnvironmentVariable('Path','Machine').TrimEnd(';') + ';C:\Program Files\CMake\bin'),
       'Machine')
     Restart-Service (Get-Service 'actions.runner.*').Name -Force
     ```
3. **cuDNN 9 must live in the CUDA `bin` dir (on PATH), not just bundled** — cuDNN 9 is
   multi-DLL and its siblings resolve via the default search path. See `docs/HOST_COMPATIBILITY.md`.
4. **`dumpbin`/build tools** — the VS 2022 generator finds the toolchain via the registry, so
   `cmake -G "Visual Studio 17 2022"` works without a Developer prompt.

---

## Linux runner (verified: `rocky8-gpu`, 2× RTX 3090, driver 610.43, CUDA 12, cuDNN 9)

Register as a **systemd service** (`svc.sh`), which is durable by design (survives logout and
reboot). The systemd unit runs as the **installing user** with a normal login-style PATH, so
the Windows service-account gotchas (pwsh, execution policy, cmake PATH) do **not** apply — the
Linux `gpu-smoke` branch is plain `bash`.

```bash
# 0) Sync the clock first (registration token is time-sensitive).
timedatectl status | grep -E 'synchronized|Local time'   # rocky8 was already NTP-synced

# 1) Prereqs: NVIDIA driver + CUDA 12.x runtime + cuDNN 9 on the ldconfig path;
#    cmake >= 3.20, gcc-c++, patchelf, git. patchelf is in EPEL on Rocky 8.
sudo dnf install -y epel-release
sudo dnf install -y cmake gcc-c++ patchelf git
nvidia-smi                                  # must show the GPU
ldconfig -p | grep -E 'libcudart|libcudnn'  # CUDA runtime + cuDNN 9 must resolve

# 2) Unpack the runner, then INSTALL IT UNDER /opt (see SELinux note below), not /home.
sudo mkdir -p /opt/actions-runner && sudo chown "$USER:$USER" /opt/actions-runner
cd /opt/actions-runner
curl -fSL -o runner.tar.gz \
  https://github.com/actions/runner/releases/download/v2.335.1/actions-runner-linux-x64-2.335.1.tar.gz
tar xzf runner.tar.gz && rm runner.tar.gz

# 3) Configure with the cuda label (config.sh generates svc.sh from a template).
./config.sh --url https://github.com/samhodge-tokgan/Hastur \
  --token <REG_TOKEN> --name rocky8-gpu --labels cuda,gpu --unattended --replace

# 4) Install + start the systemd service (run as your user).
sudo ./svc.sh install "$USER"
sudo ./svc.sh start
sudo ./svc.sh status         # want: Active: active (running)

# 5) Stage the model, OR point the repo var at wherever it already lives.
#    On rocky8 it was already at ~/hastur-models, so we set the var instead of copying:
#    gh variable set HASTUR_MODEL_LINUX --body /home/sam/hastur-models/sam3dbody_body.onnx
```

### Gotchas discovered (all now handled)

1. **SELinux blocks a runner under `/home` (`203/EXEC Permission denied`).** Rocky 8 runs
   SELinux *Enforcing*; the systemd unit (`init_t`) may not `exec` files labelled `user_home_t`,
   so a runner unpacked in `~/actions-runner` fails to start. Install it under **`/opt`** and
   relabel so the scripts get an exec-able context:
   ```bash
   sudo ./svc.sh uninstall            # if you already tried from ~
   sudo mv ~/actions-runner /opt/actions-runner
   sudo chown -R "$USER:$USER" /opt/actions-runner
   sudo restorecon -R /opt/actions-runner   # relabel out of user_home_t
   cd /opt/actions-runner && sudo ./svc.sh install "$USER" && sudo ./svc.sh start
   ```
2. **`svc.sh` doesn't exist until you run `config.sh`** — it's generated from
   `bin/systemd.svc.sh.template` during configuration. Configure first, then install the service.
3. **GCC 8 (Rocky 8 default) needs `-lstdc++fs` for `std::filesystem`.** The build otherwise
   fails to link with `undefined reference to std::filesystem::...path::_M_split_cmpts()`. This
   is handled *in the build* (`HASTUR_FS_LIB` in `CMakeLists.txt`, guarded by GCC version), not on
   the runner — no action needed, noted here so the link error isn't mistaken for a runner fault.
4. **`patchelf` is a configure-time hard requirement** even to build only `ort_check` (the
   isolation block's `FATAL_ERROR` runs during configuration). Install it (step 1).

### Registration/removal tokens

Generated from the repo (needs `repo` scope on the `gh` auth):

```bash
gh api -X POST repos/samhodge-tokgan/Hastur/actions/runners/registration-token --jq .token
gh api -X POST repos/samhodge-tokgan/Hastur/actions/runners/remove-token       --jq .token
```

> **One GPU host at a time.** The Windows and Rocky VMs share the same passed-through GPU, so
> only one is up at once. Whichever is running carries the `cuda` label and services
> `gpu-smoke`; the other shows *offline* in the runner list (harmless). To decommission a box
> for good, `./config.(cmd|sh) remove --token <REMOVE_TOKEN>` so it drops out of the list.
