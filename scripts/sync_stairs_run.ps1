param(
  [Parameter(Mandatory)]
  [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]*$')]
  [string]$RunId,
  [int]$Port = 9191,
  [string]$IdentityFile = "$env:USERPROFILE/.ssh/id_ed25519",
  [string]$RemoteHost = 'root@127.0.0.1'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$results = '/kaggle/working/g1-edu-stairs'
$bundles = '/codex_download_here_work_here/bundles'
$archive = "$RunId.tar.gz"
$tag = "run-$RunId"
$sshOptions = @('-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-i', $IdentityFile)

# Package completed runs after checkpoint and TensorBoard writes have stopped.
$remoteCommand = "set -eu; test -f '$results/$RunId/exit-code.txt'; " +
  "mkdir -p '$bundles'; tar --sort=name -czf '$bundles/$archive.tmp' -C '$results' '$RunId'; " +
  "mv '$bundles/$archive.tmp' '$bundles/$archive'; " +
  "cd '$bundles'; sha256sum '$archive' > '$archive.sha256'; " +
  "cat '$results/$RunId/git-commit.txt'"
$commit = ssh -p $Port @sshOptions $RemoteHost $remoteCommand
if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[a-f0-9]{40}$') {
  throw "Could not archive $RunId. Check that the Kaggle run has finished."
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

  $origin = git remote get-url origin
  if ($LASTEXITCODE -ne 0 -or $origin -notmatch '^(https://github\.com/|git@github\.com:)([^/]+/[^/]+?)(\.git)?$') {
    throw 'The origin remote must point to a GitHub repository.'
  }
  $githubRepo = $Matches[2]
  gh api "repos/$githubRepo/releases/tags/$tag" --silent 2>$null
  if ($LASTEXITCODE -eq 0) {
    gh release upload $tag $bundlePath $checksumPath --repo $githubRepo --clobber
  } else {
    $title = "G1 EDU stairs | $RunId"
    $notes = "Run configuration, logs, and any checkpoints, ONNX exports and evaluations produced on Kaggle. Source commit $commit."
    gh release create $tag $bundlePath $checksumPath --repo $githubRepo --target $commit --title $title --latest=false --notes $notes
  }
  if ($LASTEXITCODE -ne 0) { throw "GitHub upload failed for $RunId." }
  gh release view $tag --repo $githubRepo --json url --jq .url
} finally {
  Pop-Location
  foreach ($file in @($bundlePath, $checksumPath)) {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file }
  }
  Remove-Item -LiteralPath $staging
}
