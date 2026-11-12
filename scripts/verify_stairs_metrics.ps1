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
    $requirements = if (Test-Path 'experiments/requirements-metrics.lock.txt') { 'experiments/requirements-metrics.lock.txt' } else { 'experiments/requirements-metrics.txt' }
    uv pip install --python $python -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Could not install reporting dependencies.' }
  }
  $run = Join-Path $root "outputs/$RunId"
  & $python -m scripts.verify_training_readiness --output $run
  $status = $LASTEXITCODE
  $guardStatus = 0
  try {
    & (Join-Path $PSScriptRoot 'smoke_launch_guard.ps1') *> (Join-Path $run 'logs/launch-guard.log')
  } catch {
    $guardStatus = 1
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $run 'logs/launch-guard.log')
  }
  & $python -c 'import sys; from pathlib import Path; from scripts.training_metrics import atomic_json,read_json,queue; p=Path(sys.argv[1]); d=read_json(p/"readiness.json"); ok=sys.argv[2]=="0"; d["launcher_guard"]="passed" if ok else "failed"; d["status"]="failed" if not ok else d["status"]; atomic_json(p/"readiness.json",d); (p/"exit-code.txt").write_text("0\n" if ok and d["status"]=="passed" else "1\n"); queue(p)' $run $guardStatus
  if ($LASTEXITCODE -ne 0) { throw 'Could not finalize verification metadata.' }
  if (-not $LocalOnly) { & (Join-Path $PSScriptRoot 'publish_local_stairs_run.ps1') -LocalRunPath $run }
  if ($status -ne 0 -or $guardStatus -ne 0) { throw "Local verification failed; inspect $run/logs." }
  Write-Output 'Local reporting checks passed. GPU checks still require the VM. No training started.'
} finally { Pop-Location }
