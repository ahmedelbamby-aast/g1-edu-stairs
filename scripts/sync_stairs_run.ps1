param(
  [Parameter(Mandatory)]
  [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]*$')]
  [string]$RunId,
  [int]$Port = 9191,
  [string]$IdentityFile = "$env:USERPROFILE/.ssh/id_ed25519",
  [string]$RemoteHost = 'root@127.0.0.1',
  [string]$KnownHostsFile = "$env:USERPROFILE/.ssh/kaggle-zrok-9191.known_hosts",
  [switch]$Checkpoint,
  [switch]$LocalOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$bundles = '/codex_download_here_work_here/bundles'
$remoteRepo = '/codex_download_here_work_here/g1-edu-stairs'
$suffix = if ($Checkpoint) { '-checkpoint' } else { '' }
$mode = if ($Checkpoint) { 'checkpoint' } else { 'completed' }
$archive = "$RunId$suffix.tar.gz"
$tag = "run-$RunId$suffix"
$sshOptions = @('-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-i', $IdentityFile)
if (Test-Path -LiteralPath $KnownHostsFile) { $sshOptions += @('-o', "UserKnownHostsFile=$KnownHostsFile") }

$remoteCommand = "cd '$remoteRepo' && bash scripts/archive_stairs_run.sh '$RunId' '$mode'"
$commit = ssh -p $Port @sshOptions $RemoteHost $remoteCommand
if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[a-f0-9]{40}$') {
  throw "Could not archive $RunId in $mode mode. Check SSH and whether the run has a usable checkpoint or completion marker."
}

$staging = Join-Path ([IO.Path]::GetTempPath()) ("g1-stairs-" + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $staging
$bundlePath = Join-Path $staging $archive
$checksumPath = "$bundlePath.sha256"
Push-Location $root
try {
  scp -P $Port @sshOptions "${RemoteHost}:$bundles/$archive" "${RemoteHost}:$bundles/$archive.sha256" $staging
  if ($LASTEXITCODE -ne 0) { throw "Download failed for $RunId." }
  $expected = ((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0]
  if ((Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash -ne $expected) {
    throw "Checksum mismatch for $archive."
  }
  $backup = Join-Path $root "outputs/runs/$RunId"
  $null = New-Item -ItemType Directory -Force -Path $backup
  Move-Item -Force -LiteralPath $bundlePath -Destination $backup
  Move-Item -Force -LiteralPath $checksumPath -Destination $backup
  $bundlePath = Join-Path $backup $archive
  $checksumPath = "$bundlePath.sha256"
  $metricAsset = $null
  $members = @(tar -tzf $bundlePath)
  if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the verified archive.' }
  if ($members -contains "$RunId/metrics-report.zip") {
    tar -xzf $bundlePath -C $staging "$RunId/metrics-report.zip"
    if ($LASTEXITCODE -ne 0) { throw 'Could not extract the metrics report.' }
    $metricAsset = Join-Path $backup "$RunId-metrics.zip"
    Move-Item -Force -LiteralPath (Join-Path $staging "$RunId/metrics-report.zip") -Destination $metricAsset
    Remove-Item -LiteralPath (Join-Path $staging $RunId)
  }
  if ($LocalOnly) {
    Write-Output $bundlePath
    return
  }

  $origin = git remote get-url origin
  if ($LASTEXITCODE -ne 0 -or $origin -notmatch '^(https://github\.com/|git@github\.com:)([^/]+/[^/]+?)(\.git)?$') {
    throw 'The origin remote must point to a GitHub repository.'
  }
  $githubRepo = $Matches[2]
  $assets = @($bundlePath, $checksumPath)
  if ($metricAsset) { $assets += $metricAsset }
  gh api "repos/$githubRepo/releases/tags/$tag" --silent 2>$null
  if ($LASTEXITCODE -eq 0) {
    gh release upload $tag @assets --repo $githubRepo --clobber
  } else {
    $title = "G1 EDU stairs | $RunId"
    $notes = if ($Checkpoint) {
      "Rolling recovery copy containing the latest saved checkpoint and run settings. Source commit $commit."
    } else {
      "Run configuration, logs, checkpoints, ONNX export and evaluation produced on Kaggle. Source commit $commit."
    }
    gh release create $tag @assets --repo $githubRepo --target $commit --title $title --latest=false --notes $notes
  }
  if ($LASTEXITCODE -ne 0) { throw "GitHub upload failed for $RunId." }
  gh release view $tag --repo $githubRepo --json url --jq .url
} finally {
  Pop-Location
  foreach ($file in @((Join-Path $staging $archive), (Join-Path $staging "$archive.sha256"))) {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file }
  }
  Remove-Item -LiteralPath $staging
}
