param([Parameter(Mandatory)][string]$LocalRunPath)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$run = (Resolve-Path -LiteralPath $LocalRunPath).Path
$runId = Split-Path -Leaf $run
if ($runId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$') { throw 'Invalid run ID.' }
$commit = (Get-Content -LiteralPath (Join-Path $run 'git-commit.txt') -Raw).Trim()
if ($commit -notmatch '^[a-f0-9]{40}$' -or -not (Test-Path (Join-Path $run 'exit-code.txt'))) { throw 'Missing source commit or completion marker.' }
$ready = Get-Content (Join-Path $run 'publish-ready.json') -Raw | ConvertFrom-Json
if ($ready.digest -notmatch '^[a-f0-9]{64}$') { throw 'Missing queued content digest.' }
$receiptPath = Join-Path $run 'publish-receipt.json'
if (Test-Path $receiptPath) {
  $receipt = Get-Content $receiptPath -Raw | ConvertFrom-Json
  if ($receipt.digest -eq $ready.digest) { Write-Output $receipt.url; return }
}
$origin = git -C $root remote get-url origin
if ($LASTEXITCODE -ne 0 -or $origin -notmatch 'github.com[:/]([^/]+/[^/]+?)(\.git)?$') { throw 'Cannot resolve origin.' }
$repo = $Matches[1]
$backup = Join-Path $root "outputs/runs/$runId"
if ($backup.StartsWith($run + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or $backup -eq $run) {
  throw 'The archive destination must be outside the source run.'
}
New-Item -ItemType Directory -Force -Path $backup | Out-Null
$archive = Join-Path $backup "$runId.tar.gz"
tar -czf $archive -C (Split-Path -Parent $run) $runId
if ($LASTEXITCODE -ne 0) { throw 'Local run archive failed.' }
$checksum = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumPath = "$archive.sha256"
"$checksum  $runId.tar.gz" | Set-Content -LiteralPath $checksumPath -Encoding ascii
$assets = @($archive, $checksumPath)
if (Test-Path (Join-Path $run 'metrics-report.zip')) {
  $metricAsset = Join-Path $backup "$runId-metrics.zip"
  Copy-Item -LiteralPath (Join-Path $run 'metrics-report.zip') -Destination $metricAsset -Force
  $assets += $metricAsset
}
$tag = "run-$runId"
gh api "repos/$repo/releases/tags/$tag" --silent 2>$null
if ($LASTEXITCODE -eq 0) {
  gh release upload $tag @assets --repo $repo --clobber
} else {
  gh release create $tag @assets --repo $repo --target $commit --latest=false --title "G1 EDU | $runId" `
    --notes 'Scripted verification artifacts and raw reports. No training was performed by this verification. See readiness.json for passed and pending server checks.'
}
if ($LASTEXITCODE -ne 0) { throw 'GitHub upload failed; queued work is retained for retry.' }
$url = "https://github.com/$repo/releases/tag/$tag"
@{digest=$ready.digest; url=$url; uploaded_at=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json |
  Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Output $url
