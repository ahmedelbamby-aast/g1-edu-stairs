param(
  [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]*$')][string]$RunId = ("metrics-local-" + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')),
  [switch]$LocalOnly
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $venv = Join-Path $root 'outputs/metrics-report-venv'
  $python = Join-Path $venv 'Scripts/python.exe'
  if (-not (Test-Path $python)) {
    uv venv --python 3.12 $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated metrics environment.' }
    uv pip install --python $python -r experiments/requirements-metrics.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install reporting dependencies.' }
  }
  $run = Join-Path $root "outputs/$RunId"
  & $python -m scripts.verify_training_readiness --output $run
  $status = $LASTEXITCODE
  & (Join-Path $PSScriptRoot 'smoke_launch_guard.ps1')
  if (-not $LocalOnly) { & (Join-Path $PSScriptRoot 'publish_local_stairs_run.ps1') -LocalRunPath $run }
  if ($status -ne 0) { throw "Local verification failed; inspect $run/logs." }
  Write-Output 'Local reporting checks passed. GPU checks still require the VM. No training started.'
} finally { Pop-Location }
