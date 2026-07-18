<#
Copyright the Hastur authors.
SPDX-License-Identifier: LicenseRef-SAM-License

convert_models.ps1 — generate the Sam3dBody plugin's model set LOCALLY from your
own copy of Meta's gated SAM-3D-Body checkpoints (Windows counterpart of
convert_models.sh).

Hastur is MODEL-LESS by design: it ships NO Meta weights. You download the gated
SAM-3D-Body checkpoints yourself (your own HF token, having accepted the SAM
License + DINOv3 gated terms) and run this one script to EXPORT/EXTRACT the
plugin's runtime assets into a target model dir (default the installed bundle's
Contents\Resources). It runs the tools\ export scripts IN ORDER to populate:

    person_detector.onnx (+ .json)   person / bbox detector      (fast, no ckpt)
    mhr_assets.bin        (+ manifest) MHR static buffers (C++ FK+LBS)
    pose_corrective.onnx             MHR pose-corrective MLP
    sam3dbody_body.onnx              SAM-3D-Body ViT regressor   (SLOW, hours)
    sam3dbody_hand.onnx (+ .json)    hand-decoder 2nd pass       (SLOW, hours)
    mhr_wrist.bin                    wrist-twist rest rotations (from export_hand)

IDEMPOTENT: an existing target file is skipped (use -Force to regenerate). This
REPLACES the download role of fetch_models.ps1 for Hastur.

PREREQUISITES: a Python env with torch / onnx / onnxruntime and the sam-3d-body
package importable; ~7 GB free disk; and PATIENCE — the body + hand ViT exports
take multiple HOURS on CPU.

Usage:
  .\tools\convert_models.ps1 `
      -CkptDir  C:\path\sam-3d-body\checkpoints\sam-3d-body-dinov3 `
      -OutDir   "$env:CommonProgramFiles\OFX\Plugins\Sam3dBody.ofx.bundle\Contents\Resources" `
      -Python   C:\path\torch-env\Scripts\python.exe
#>
[CmdletBinding()]
param(
  [string]$CkptDir,
  [string]$Sam3dRoot = $env:SAM3D_ROOT,
  [string]$OutDir,
  [string]$Python,
  [string]$ToolsDir,
  [switch]$Force,
  [switch]$PoseFp16,
  [switch]$SkipHand
)
$ErrorActionPreference = "Stop"

# ─── locate the export tools ────────────────────────────────────────────────
if (-not $ToolsDir) { $ToolsDir = $PSScriptRoot }
if (-not (Test-Path (Join-Path $ToolsDir "export_detector.py")) -and $env:HASTUR_TOOLS_DIR) {
  $ToolsDir = $env:HASTUR_TOOLS_DIR
}
if (-not (Test-Path (Join-Path $ToolsDir "export_detector.py"))) {
  throw "Cannot find the Hastur export tools (export_detector.py ...) in '$ToolsDir'. " +
        "Run from a Hastur checkout's tools\ dir or set HASTUR_TOOLS_DIR / -ToolsDir."
}

# ─── resolve checkpoints + sam-3d-body root ─────────────────────────────────
if (-not $Sam3dRoot) {
  if ($CkptDir) { $Sam3dRoot = (Resolve-Path (Join-Path $CkptDir "..\..")).Path }
  else { throw "Provide -CkptDir (…\checkpoints\sam-3d-body-dinov3) or -Sam3dRoot." }
}
if (-not $CkptDir) { $CkptDir = Join-Path $Sam3dRoot "checkpoints\sam-3d-body-dinov3" }

$toolCkpt = Join-Path $Sam3dRoot "checkpoints\sam-3d-body-dinov3\model.ckpt"
$toolMhr  = Join-Path $Sam3dRoot "checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt"

if (-not (Test-Path (Join-Path $Sam3dRoot "sam_3d_body"))) {
  throw "sam_3d_body package not found under -Sam3dRoot '$Sam3dRoot'."
}
if (-not (Test-Path $toolCkpt) -or -not (Test-Path $toolMhr)) {
  throw "Gated checkpoints not found where the export tools resolve them:`n" +
        "  $toolCkpt`n  $toolMhr`n" +
        "Required layout: <Sam3dRoot>\checkpoints\sam-3d-body-dinov3\{model.ckpt,assets\mhr_model.pt}."
}

# ─── python ─────────────────────────────────────────────────────────────────
if (-not $Python) {
  $cand = Join-Path $Sam3dRoot "env\Scripts\python.exe"
  if (Test-Path $cand) { $Python = $cand }
  else { $Python = (Get-Command python -ErrorAction SilentlyContinue).Source }
}
if (-not $Python) { throw "No usable python. Pass -Python C:\path\torch-env\Scripts\python.exe" }

# ─── output (model) directory ───────────────────────────────────────────────
function Find-Resources {
  if ((Split-Path $ToolsDir -Leaf) -eq "Contents" -and (Split-Path $ToolsDir -Parent) -like "*.ofx.bundle") {
    return (Join-Path $ToolsDir "Resources")
  }
  foreach ($d in @((Join-Path $env:CommonProgramFiles "OFX\Plugins"),
                   "C:\Program Files\Common Files\OFX\Plugins")) {
    $b = Join-Path $d "Sam3dBody.ofx.bundle"
    if (Test-Path $b) { return (Join-Path $b "Contents\Resources") }
  }
  return $null
}
if (-not $OutDir) {
  $OutDir = Find-Resources
  if (-not $OutDir) {
    throw "No -OutDir and no installed Sam3dBody.ofx.bundle found. " +
          "Pass -OutDir <bundle>\Contents\Resources (then set HASTUR_MODEL_DIR if custom)."
  }
}
New-Item -ItemType Directory -Force $OutDir | Out-Null

$env:SAM3D_ROOT = $Sam3dRoot

Write-Host "== Hastur model generation (model-less; user-generated from gated weights) =="
Write-Host "  python      : $Python"
Write-Host "  tools       : $ToolsDir"
Write-Host "  sam3d-root  : $Sam3dRoot"
Write-Host "  checkpoints : $toolCkpt"
Write-Host "  out-dir     : $OutDir"
Write-Host "  force: $Force  pose-fp16: $PoseFp16  skip-hand: $SkipHand"
Write-Host ""
Write-Host "  LEGAL: assets are DERIVED from Meta's gated SAM-3D-Body weights (SAM License,"
Write-Host "  no-reverse-engineering) with embedded DINOv3 (own gated terms). Generate from"
Write-Host "  YOUR licensed copy; do not redistribute. TIME: body + hand exports take HOURS."

$script:step = 0
function Run-Step {
  param([string]$Out, [string]$Label, [string[]]$CmdArgs)
  $script:step++
  $dest = Join-Path $OutDir $Out
  if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0) -and (-not $Force)) {
    Write-Host ""; Write-Host "-- [$script:step] $Label -> $Out  [SKIP: exists] --"; return
  }
  Write-Host ""; Write-Host "-- [$script:step] $Label -> $Out --"
  Write-Host "   $Python $($CmdArgs -join ' ')"
  & $Python @CmdArgs
  if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
  if (-not (Test-Path $dest) -or (Get-Item $dest).Length -eq 0) {
    throw "$Label did not produce $dest"
  }
}

# ─── ordered export pipeline ────────────────────────────────────────────────
Run-Step "person_detector.onnx" "person detector (ssdlite)" `
  @((Join-Path $ToolsDir "export_detector.py"), "--model", "ssdlite",
    "--out", (Join-Path $OutDir "person_detector.onnx"))

Run-Step "mhr_assets.bin" "MHR static assets" `
  @((Join-Path $ToolsDir "extract_mhr_assets.py"),
    "--out", (Join-Path $OutDir "mhr_assets.bin"),
    "--manifest", (Join-Path $OutDir "mhr_assets.manifest.json"))

$poseArgs = @((Join-Path $ToolsDir "export_pose_corrective.py"),
              "--out", (Join-Path $OutDir "pose_corrective.onnx"))
if ($PoseFp16) { $poseArgs += "--fp16" }
Run-Step "pose_corrective.onnx" "pose-corrective MLP" $poseArgs

Run-Step "sam3dbody_body.onnx" "SAM-3D-Body regressor (faithful)" `
  @((Join-Path $ToolsDir "export_sam3dbody.py"), "--mode", "faithful",
    "--out", (Join-Path $OutDir "sam3dbody_body.onnx"))

if ($SkipHand) {
  Write-Host ""; Write-Host "-- hand decoder SKIPPED (-SkipHand): plugin runs body-only --"
} else {
  Run-Step "sam3dbody_hand.onnx" "hand-decoder 2nd pass + mhr_wrist.bin" `
    @((Join-Path $ToolsDir "export_hand.py"),
      "--out", (Join-Path $OutDir "sam3dbody_hand.onnx"))
}

Write-Host ""
Write-Host "== done. model set in: $OutDir =="
foreach ($f in @("person_detector.onnx","person_detector.json","mhr_assets.bin",
                 "mhr_assets.manifest.json","pose_corrective.onnx","sam3dbody_body.onnx",
                 "sam3dbody_hand.onnx","sam3dbody_hand.json","mhr_wrist.bin")) {
  $p = Join-Path $OutDir $f
  if (Test-Path $p) { "  [ok]      {0,-28} {1:N1} MB" -f $f, ((Get-Item $p).Length/1MB) }
  else { "  [missing] $f" }
}
Write-Host ""
Write-Host "The plugin resolves these from Contents\Resources, `$env:HASTUR_MODEL_DIR, or the"
Write-Host "'Model directory' param. If you wrote to a custom -OutDir, set HASTUR_MODEL_DIR."
