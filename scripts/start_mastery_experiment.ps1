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
  '-o', "UserKnownHostsFile=$KnownHostsFile", '-i', "$env:USERPROFILE/.ssh/id_ed25519")
$remoteRepo = '/codex_download_here_work_here/g1-edu-stairs'
$prefix = "cd $remoteRepo && export LD_LIBRARY_PATH=/usr/local/nvidia/lib64 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl WARP_CACHE_PATH=/codex_download_here_work_here/cache/warp && "
foreach ($gpu in 0,1) {
  ssh @sshOptions root@127.0.0.1 ($prefix + ".venv/bin/python -m scripts.preflight_training --teacher-onnx $InitActorOnnx --output /kaggle/working/g1-edu-stairs/$RunId-preflight --device cuda:$gpu")
  if ($LASTEXITCODE -ne 0) { throw "GPU $gpu preflight failed; no training launched." }
}
ssh @sshOptions root@127.0.0.1 ($prefix + ".venv/bin/python -m scripts.check_gpu_collective --output /kaggle/working/g1-edu-stairs/$RunId-preflight")
if ($LASTEXITCODE -ne 0) { throw 'Two-GPU communication preflight failed; no training launched.' }
if (-not $StartTraining) { Write-Output 'Preflight passed. Training was not started. Use -StartTraining for an explicit launch.'; return }
$launch = "nohup .venv/bin/python -m scripts.run_mastery_experiment --run-id $RunId --stage $Stage --seed $Seed --init-actor-onnx $InitActorOnnx > /kaggle/working/g1-edu-stairs/$RunId.launch.log 2>&1 < /dev/null &"
ssh @sshOptions root@127.0.0.1 ($prefix + $launch)
if ($LASTEXITCODE -ne 0) { throw 'Remote launch failed.' }
& (Join-Path $PSScriptRoot 'watch_stairs_results.ps1') -RunPrefix $RunId -Port $Port -KnownHostsFile $KnownHostsFile
