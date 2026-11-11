param(
  [Parameter(Mandatory)]
  [ValidatePattern('^SHA256:[A-Za-z0-9+/]{43}$')]
  [string]$Fingerprint,
  [int]$Port = 9191,
  [string]$IdentityFile = "$env:USERPROFILE/.ssh/id_ed25519",
  [string]$ZrokPath,
  [switch]$Bootstrap
)

$ErrorActionPreference = 'Stop'
$trustPath = Join-Path $env:USERPROFILE ".ssh/kaggle-zrok-$Port.known_hosts"
$pidFile = "$trustPath.pid"
$probe = "$trustPath.$([guid]::NewGuid()).probe"
$endpoint = "127.0.0.1:$Port"
$common = @('-p', "$Port", '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
  '-o', 'GlobalKnownHostsFile=NUL', '-o', 'HostKeyAlgorithms=ssh-ed25519')

function Get-VerifiedHost {
  if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe }
  # Learn the offered public key without sending authentication credentials.
  $probeArgs = $common + @('-o', 'PubkeyAuthentication=no', '-o', 'PreferredAuthentications=none',
    '-o', 'StrictHostKeyChecking=accept-new', '-o', "UserKnownHostsFile=$probe", 'root@127.0.0.1', 'true')
  ssh @probeArgs 2>$null | Out-Null
  if (!(Test-Path -LiteralPath $probe)) { return $false }
  $offered = ssh-keygen -lf $probe
  return (($offered -split '\s+')[1] -ceq $Fingerprint)
}

try {
  $listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if ($listener -and !(Get-VerifiedHost)) {
    $ownedPid = if (Test-Path -LiteralPath $pidFile) { Get-Content -LiteralPath $pidFile -Raw }
    $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if ($ownedPid -and $ownedPid.Trim() -eq "$($listener.OwningProcess)" -and $process.ProcessName -eq 'zrok2') {
      Stop-Process -Id $process.Id
      $process.WaitForExit(5000) | Out-Null
      $listener = $null
    } else {
      throw "Port $Port is already held by a different tunnel. Close the old tunnel and retry."
    }
  }
  if (!$listener) {
    if (!$ZrokPath) {
      $installed = Get-Command zrok2 -ErrorAction SilentlyContinue
      $ZrokPath = if ($installed) { $installed.Source } else { Join-Path $env:USERPROFILE 'bin/zrok2.exe' }
    }
    $share = $env:KAGGLE_ZROK_SHARE
    if (!$share) {
      $secret = Read-Host 'Paste the private zrok share from the Kaggle notebook' -AsSecureString
      $share = [Net.NetworkCredential]::new('', $secret).Password
    }
    if ($share -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid zrok share format.' }
    $forward = Start-Process -FilePath $ZrokPath -ArgumentList @(
      'access', 'private', $share, '--bind', $endpoint, '--headless', '--force-local'
    ) -WindowStyle Hidden -PassThru
    $forward.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
    $ready = $false
    for ($attempt = 0; $attempt -lt 6 -and !$ready -and !$forward.HasExited; $attempt++) {
      Start-Sleep -Seconds 2
      $ready = Get-VerifiedHost
    }
    if (!$ready) {
      if (!$forward.HasExited) { Stop-Process -Id $forward.Id }
      throw 'The tunnel did not return the verified Kaggle host key.'
    }
  }

  Move-Item -LiteralPath $probe -Destination $trustPath -Force
  $authentication = $common + @(
    '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$trustPath",
    '-i', $IdentityFile, 'root@127.0.0.1'
  )
  $remote = if ($Bootstrap) {
    @'
set -e
mkdir -p /codex_download_here_work_here
cd /codex_download_here_work_here
if [ -d g1-edu-stairs/.git ]; then
  git -C g1-edu-stairs pull --ff-only
else
  git clone https://github.com/ahmedelbamby-aast/g1-edu-stairs.git
fi
cd g1-edu-stairs
bash scripts/bootstrap_kaggle.sh
'@
  } else { "printf 'Verified Kaggle SSH connection.\n'" }
  ssh @authentication $remote
  if ($LASTEXITCODE -ne 0) { throw 'Kaggle SSH or bootstrap failed.' }
} finally {
  if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe }
}
