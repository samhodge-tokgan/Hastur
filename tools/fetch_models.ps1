<#
Copyright the Hastur authors.
SPDX-License-Identifier: Apache-2.0

fetch_models.ps1 — download/stage the SAM 3D Body ONNX models + mesh assets into the
installed Sam3dBody.ofx.bundle's Contents\Resources.

NOTE: Hastur is MODEL-LESS by design. The models below are USER-GENERATED locally
(exported from the SAM 3D Body checkpoints, or pulled from a private mirror you
control) — they are NOT committed and there is no public release tag yet. The manifest
checksums/sizes are placeholders (TODO) to be pinned once the export pipeline produces
stable weights. The download plumbing (gh / GITHUB_TOKEN / curl, HF mirror override) is
kept from humbaba so a gated release can drop straight in.

Usage:
  .\fetch_models.ps1 [-ResourcesDir <path>] [-Tag models-v1] [-BaseUrl <url>]

With no -ResourcesDir it locates the installed bundle in the standard OFX dirs.
Writing to %CommonProgramFiles%\OFX\Plugins needs an elevated PowerShell.
#>
[CmdletBinding()]
param(
  [string]$ResourcesDir,
  [string]$Tag = $(if ($env:HASTUR_MODELS_TAG) { $env:HASTUR_MODELS_TAG } else { "models-v1" }),
  [string]$BaseUrl
)

$ErrorActionPreference = "Stop"
$repo = "samhodge-tokgan/Hastur"
if (-not $BaseUrl) {
  $BaseUrl = if ($env:HASTUR_MODELS_BASE_URL) { $env:HASTUR_MODELS_BASE_URL }
             else { "https://github.com/$repo/releases/download/$Tag" }
}

# asset name, installed filename, sha256, bytes.
# TODO(models): fill in the real sha256 + byte size once the models are pinned.
$models = @(
  @{ asset="person_detector.onnx";  name="person_detector.onnx";  sha="TODO"; bytes=0 },
  @{ asset="sam3dbody_body.onnx";   name="sam3dbody_body.onnx";   sha="TODO"; bytes=0 },
  @{ asset="sam3dbody_hand.onnx";   name="sam3dbody_hand.onnx";   sha="TODO"; bytes=0 },
  @{ asset="mhr_assets.bin";        name="mhr_assets.bin";        sha="TODO"; bytes=0 },
  @{ asset="pose_corrective.onnx";  name="pose_corrective.onnx";  sha="TODO"; bytes=0 }
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
New-Item -ItemType Directory -Force $res | Out-Null

Write-Host "Installing models into: $res"
Write-Host "Source: $BaseUrl"
foreach ($m in $models) {
  $dest = Join-Path $res $m.name
  if ($m.sha -eq "TODO") {
    Write-Host "  [todo] $($m.name) — no pinned checksum yet (model-less; stage it locally)"
    continue
  }
  if (Test-Path $dest) {
    $h = (Get-FileHash -Algorithm SHA256 $dest).Hash.ToLower()
    if ($h -eq $m.sha) { Write-Host "  [skip] $($m.name) (already present, checksum OK)"; continue }
  }
  Write-Host "  [get]  $($m.asset) -> $($m.name) ($($m.bytes) bytes)"
  $tmp = "$dest.part"
  # Prefer the GitHub CLI, which handles auth for a gated/private release; fall back to
  # a plain/token'd curl of the download URL.
  $gh = Get-Command gh -ErrorAction SilentlyContinue
  if ($gh -and (gh auth status 2>$null; $LASTEXITCODE -eq 0)) {
    & gh release download $Tag --repo $repo --pattern $m.asset --output $tmp --clobber
    if ($LASTEXITCODE -ne 0) { throw "gh release download failed for $($m.asset)" }
  } else {
    $hdr = @()
    if ($env:GITHUB_TOKEN) { $hdr = @("-H", "Authorization: token $($env:GITHUB_TOKEN)") }
    & curl.exe -fL --retry 3 --progress-bar @hdr -o $tmp "$BaseUrl/$($m.asset)"
    if ($LASTEXITCODE -ne 0) { throw "download failed for $($m.asset)" }
  }
  $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
  if ($got -ne $m.sha) {
    Remove-Item $tmp -Force
    throw "checksum mismatch for $($m.asset): expected $($m.sha), got $got"
  }
  Move-Item -Force $tmp $dest
  Write-Host "  [ok]   $($m.name) verified"
}
Write-Host "Done. Model staging complete for $res"
