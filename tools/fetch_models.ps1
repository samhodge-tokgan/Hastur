<#
Copyright the Hastur authors.
SPDX-License-Identifier: LicenseRef-SAM-License

fetch_models.ps1 — stage the SAM 3D Body ONNX models + mesh assets into the
installed Sam3dBody.ofx.bundle's Contents\Resources.

── Hastur is MODEL-LESS ──────────────────────────────────────────────────────
Hastur ships NO Meta weights. The runtime assets are DERIVED from Meta's gated
SAM-3D-Body checkpoints (SAM License + embedded DINOv3 gated terms), which we must
NOT redistribute. So the PRIMARY path is to GENERATE them locally from your own
licensed copy:

    .\tools\convert_models.ps1 -CkptDir <checkpoints> -OutDir <Resources> -Python <env>

This script therefore does NOT download from a Hastur release (there is none — it
would redistribute derived Meta weights). It only (a) points you at convert_models
(the default, no-arg behaviour), and (b) OPTIONALLY pulls a set you already
generated from a mirror YOU host, via -BaseUrl / $env:HASTUR_MODELS_BASE_URL.

Usage:
  .\fetch_models.ps1 [-ResourcesDir <path>]                     # prints generate-locally guidance
  .\fetch_models.ps1 -BaseUrl https://your-mirror/path [-ResourcesDir <path>]

With no -ResourcesDir it locates the installed bundle in the standard OFX dirs.
#>
[CmdletBinding()]
param(
  [string]$ResourcesDir,
  [string]$BaseUrl = $env:HASTUR_MODELS_BASE_URL
)

$ErrorActionPreference = "Stop"

# The user-generated model set. No pinned checksums: generated on YOUR machine /
# mirror, not distributed by Hastur. A mirror may host an optional <name>.sha256
# sidecar which, when present, is verified.
$models = @(
  "person_detector.onnx", "person_detector.json",
  "sam3dbody_body.onnx",
  "sam3dbody_hand.onnx", "sam3dbody_hand.json",
  "mhr_assets.bin", "mhr_assets.manifest.json",
  "pose_corrective.onnx", "mhr_wrist.bin"
)

function Find-Resources {
  param([string]$Dir)
  if ($Dir) {
    if ($Dir -like "*\Contents\Resources") { return $Dir }
    if ($Dir -like "*.ofx.bundle")         { return (Join-Path $Dir "Contents\Resources") }
    return $Dir
  }
  $candidates = @(
    (Join-Path $env:CommonProgramFiles "OFX\Plugins"),
    "C:\Program Files\Common Files\OFX\Plugins"
  ) | Select-Object -Unique
  foreach ($d in $candidates) {
    $b = Join-Path $d "Sam3dBody.ofx.bundle"
    if (Test-Path $b) { return (Join-Path $b "Contents\Resources") }
  }
  return $null
}

$res = Find-Resources -Dir $ResourcesDir
if (-not $res) {
  throw "Could not find Sam3dBody.ofx.bundle in the standard OFX dirs. Pass -ResourcesDir <bundle>\Contents\Resources."
}

# ── primary path: no mirror configured -> generate locally ───────────────────
if (-not $BaseUrl) {
  Write-Host "Hastur is model-less: there is no download of Meta-derived weights." -ForegroundColor Yellow
  Write-Host ""
  Write-Host "Generate the model set LOCALLY from your own gated SAM-3D-Body checkpoints:"
  Write-Host ""
  Write-Host "  .\tools\convert_models.ps1 ``"
  Write-Host "      -CkptDir C:\path\sam-3d-body\checkpoints\sam-3d-body-dinov3 ``"
  Write-Host "      -OutDir  `"$res`" ``"
  Write-Host "      -Python  C:\path\torch-env\Scripts\python.exe"
  Write-Host ""
  Write-Host "See docs\MODELS.md for the full flow. To instead pull a set you already"
  Write-Host "generated onto a mirror you host, set -BaseUrl / `$env:HASTUR_MODELS_BASE_URL."
  exit 2
}

# ── optional mirror path: pull user-generated assets from your own host ──────
New-Item -ItemType Directory -Force $res | Out-Null
Write-Host "Staging user-generated models into: $res"
Write-Host "Mirror: $BaseUrl"
$hdr = @()
if ($env:GITHUB_TOKEN) { $hdr = @("-H", "Authorization: token $($env:GITHUB_TOKEN)") }
foreach ($name in $models) {
  $dest = Join-Path $res $name
  $tmp = "$dest.part"
  Write-Host "  [get]  $name"
  & curl.exe -fL --retry 3 --progress-bar @hdr -o $tmp "$BaseUrl/$name"
  if ($LASTEXITCODE -ne 0) {
    Remove-Item $tmp -ErrorAction SilentlyContinue
    Write-Host "  [warn] $name not on mirror (skipping)"; continue
  }
  $wantRaw = & curl.exe -fsSL @hdr "$BaseUrl/$name.sha256" 2>$null
  if ($LASTEXITCODE -eq 0 -and $wantRaw) {
    $want = ($wantRaw -split '\s+')[0].ToLower()
    $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
    if ($got -ne $want) {
      Remove-Item $tmp -Force
      throw "checksum mismatch for $name (want $want got $got)"
    }
    Write-Host "  [ok]   $name (sha256 verified)"
  } else {
    Write-Host "  [ok]   $name (no sha256 sidecar; unverified)"
  }
  Move-Item -Force $tmp $dest
}
Write-Host "Done. Model staging complete for $res"
