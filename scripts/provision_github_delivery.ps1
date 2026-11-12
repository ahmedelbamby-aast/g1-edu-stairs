param(
  [int]$Port = 9191,
  [string]$IdentityFile = "$env:USERPROFILE/.ssh/id_ed25519",
  [string]$KnownHostsFile = "$env:USERPROFILE/.ssh/kaggle-zrok-9191.known_hosts"
)
$ErrorActionPreference = 'Stop'
$sshOptions = @('-p',"$Port",'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
  '-o','ConnectTimeout=15','-o',"UserKnownHostsFile=$KnownHostsFile",'-i',$IdentityFile)
# Capture the existing CLI credential; never print it, put it on the command
# line, or store it in the repository, results, recovery bundles or Git config.
$credential = gh auth token
if ($LASTEXITCODE -ne 0 -or -not $credential -or $credential -match '\s') { throw 'A valid GitHub CLI login is required.' }
try {
  $credential | ssh @sshOptions root@127.0.0.1 'umask 077; mkdir -p /codex_download_here_work_here/credentials; chmod 700 /codex_download_here_work_here/credentials; cat > /codex_download_here_work_here/credentials/github-token; chmod 600 /codex_download_here_work_here/credentials/github-token'
  if ($LASTEXITCODE -ne 0) { throw 'Could not provision the private server credential.' }
} finally { Remove-Variable credential -ErrorAction SilentlyContinue }
Write-Output 'Provisioned GitHub delivery credential over verified SSH; no credential was logged.'
