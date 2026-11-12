param(
  [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]*$')][string]$RunId,
  [Parameter(Mandatory)][ValidatePattern('^/[a-zA-Z0-9_./-]+$')][string]$InitActorOnnx,
  [ValidateSet('ascent','turn','descent','round_trip','student')][string]$Stage = 'round_trip',
  [int]$Seed = 42,
  [switch]$StartTraining,
  [int]$Port = 9191,
  [string]$KnownHostsFile = "$env:USERPROFILE/.ssh/kaggle-zrok-9191.known_hosts"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$origin = git -C $root remote get-url origin
if ($LASTEXITCODE -ne 0 -or $origin -notmatch 'github.com[:/]([^/]+/[^/]+?)(\.git)?$') { throw 'Cannot resolve origin.' }
$repo = $Matches[1]
$canPush = gh api "repos/$repo" --jq .permissions.push
if ($LASTEXITCODE -ne 0 -or $canPush -ne 'true') { throw 'GitHub login does not have write access to the result repository.' }
$sshOptions = @('-p', "$Port", '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
  '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
  '-o', "UserKnownHostsFile=$KnownHostsFile", '-i', "$env:USERPROFILE/.ssh/id_ed25519")
$remoteRepo = '/codex_download_here_work_here/g1-edu-stairs'
$prefix = "cd $remoteRepo && export LD_LIBRARY_PATH=/usr/local/nvidia/lib64 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl WARP_CACHE_PATH=/codex_download_here_work_here/cache/warp && "
function Invoke-ReadOnlyPreflight([string]$Command, [string]$Label) {
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    ssh @sshOptions root@127.0.0.1 ($prefix + $Command)
    $status = $LASTEXITCODE
    if ($status -eq 0) { return }
    if ($status -ne 255 -or $attempt -eq 3) { throw "$Label failed (exit $status); no training launched." }
    Write-Warning "$Label lost its SSH connection. Retrying the read-only check ($attempt/3)."
    Start-Sleep -Seconds $attempt
  }
}
foreach ($gpu in 0,1) {
  Invoke-ReadOnlyPreflight ".venv/bin/python -m scripts.preflight_training --teacher-onnx $InitActorOnnx --output /kaggle/working/g1-edu-stairs/$RunId-preflight --device cuda:$gpu" "GPU $gpu preflight"
}
Invoke-ReadOnlyPreflight ".venv/bin/python -m scripts.check_gpu_collective --output /kaggle/working/g1-edu-stairs/$RunId-preflight" 'Two-GPU communication preflight'
if (-not $StartTraining) { Write-Output 'Preflight passed. Training was not started. Use -StartTraining for an explicit launch.'; return }
$launch = "test ! -e /kaggle/working/g1-edu-stairs/$RunId-report && mkdir /kaggle/working/g1-edu-stairs/$RunId-launch.lock && { nohup .venv/bin/python -m scripts.run_mastery_experiment --run-id $RunId --stage $Stage --seed $Seed --init-actor-onnx $InitActorOnnx > /kaggle/working/g1-edu-stairs/$RunId.launch.log 2>&1 < /dev/null & }"
ssh @sshOptions root@127.0.0.1 ($prefix + $launch)
if ($LASTEXITCODE -ne 0) { throw 'Launch status is uncertain or the run ID is already used. Inspect the remote run log before retrying. Training launch is never retried automatically.' }
& (Join-Path $PSScriptRoot 'watch_stairs_results.ps1') -RunPrefix $RunId -Port $Port -KnownHostsFile $KnownHostsFile
