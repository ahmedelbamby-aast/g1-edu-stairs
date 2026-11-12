param(
  [Parameter(Mandatory)][ValidatePattern('^SHA256:[A-Za-z0-9+/]{43}$')][string]$Fingerprint,
  [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$Share,
  [int]$Port = 9191,
  [switch]$SkipBootstrap
)
$ErrorActionPreference = 'Stop'
$previousShare = $env:KAGGLE_ZROK_SHARE
try {
  $env:KAGGLE_ZROK_SHARE = $Share
  & "$PSScriptRoot/connect_kaggle.ps1" -Fingerprint $Fingerprint -Port $Port -ReplaceTunnel -Bootstrap:(-not $SkipBootstrap)
} finally { $env:KAGGLE_ZROK_SHARE = $previousShare }
& "$PSScriptRoot/provision_github_delivery.ps1" -Port $Port -KnownHostsFile "$env:USERPROFILE/.ssh/kaggle-zrok-$Port.known_hosts"
$sshOptions = @('-p',"$Port",'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
  '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
  '-o',"UserKnownHostsFile=$env:USERPROFILE/.ssh/kaggle-zrok-$Port.known_hosts",'-i',"$env:USERPROFILE/.ssh/id_ed25519")
$remote = @'
set -e
cd /codex_download_here_work_here/g1-edu-stairs
export MPLBACKEND=Agg MPLCONFIGDIR=/codex_download_here_work_here/cache/matplotlib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.restore_github_results
.venv/bin/python -m scripts.server_results_service start
.venv/bin/python -m scripts.server_results_service status
'@
ssh @sshOptions root@127.0.0.1 $remote
if ($LASTEXITCODE -ne 0) { throw 'Session recovery failed; inspect the restore and service status before training.' }
Write-Output 'Restored published results and started independent GPU-server plotting/uploading. No training started.'
