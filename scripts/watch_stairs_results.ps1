param(
  [switch]$Once,
  [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]*$')][string]$RunPrefix,
  [ValidateRange(5, 300)][int]$PollSeconds = 30,
  [int]$Port = 9191,
  [string]$IdentityFile = "$env:USERPROFILE/.ssh/id_ed25519",
  [string]$RemoteHost = 'root@127.0.0.1',
  [string]$KnownHostsFile = "$env:USERPROFILE/.ssh/kaggle-zrok-9191.known_hosts"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$sshOptions = @('-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
  '-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-i',$IdentityFile,
  '-o',"UserKnownHostsFile=$KnownHostsFile")
$origin = git -C $root remote get-url origin
if ($LASTEXITCODE -ne 0 -or $origin -notmatch 'github.com[:/]([^/]+/[^/]+?)(\.git)?$') { throw 'Cannot resolve origin.' }
$repo = $Matches[1]
gh api "repos/$repo" --silent
if ($LASTEXITCODE -ne 0) { throw 'GitHub authentication cannot access the result repository. Run gh auth login.' }
$receiptPath = Join-Path $root 'outputs/publish-receipts.jsonl'
New-Item -ItemType Directory -Force -Path (Split-Path $receiptPath) | Out-Null
do {
  try {
    $json = ssh -p $Port @sshOptions $RemoteHost 'cd /codex_download_here_work_here/g1-edu-stairs && .venv/bin/python -m scripts.training_metrics pending'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read upload queue; retaining all pending work for retry.' }
    $items = @($json | ConvertFrom-Json)
    foreach ($item in $items) {
      if ($RunPrefix -and $item.run_id -ne 'experiment-comparisons' -and -not $item.run_id.StartsWith($RunPrefix, [StringComparison]::Ordinal)) { continue }
      if ($item.run_id -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$' -or $item.digest -notmatch '^[a-f0-9]{64}$') {
        throw 'Rejected an invalid publication queue entry.'
      }
      & (Join-Path $PSScriptRoot 'sync_stairs_run.ps1') -RunId $item.run_id -Port $Port `
        -IdentityFile $IdentityFile -RemoteHost $RemoteHost -KnownHostsFile $KnownHostsFile | Out-Host
      $url = "https://github.com/$repo/releases/tag/run-$($item.run_id)"
      $ack = "cd /codex_download_here_work_here/g1-edu-stairs && .venv/bin/python -m scripts.training_metrics acknowledge --run-id $($item.run_id) --digest $($item.digest) --url $url"
      ssh -p $Port @sshOptions $RemoteHost $ack
      if ($LASTEXITCODE -ne 0) { throw 'Upload receipt was not accepted; the queue will retry.' }
      @{run=$item.run_id; digest=$item.digest; url=$url; uploaded_at=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath $receiptPath
      Write-Output "Published $($item.run_id)"
    }
  } catch {
    if ($Once) { throw }
    Write-Warning $_
  }
  if (-not $Once) { Start-Sleep -Seconds $PollSeconds }
} while (-not $Once)
