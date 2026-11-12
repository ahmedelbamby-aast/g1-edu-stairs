# All SSH/GitHub commands are mocked. This test cannot launch remote training.
$ErrorActionPreference = 'Stop'
$global:StairLaunchSmoke = @{ commands=[Collections.Generic.List[string]]::new(); failAll=$false; retryOnce=$true }
function global:gh { $global:LASTEXITCODE = 0; 'true' }
function global:ssh {
  $command = $args[-1]
  $global:StairLaunchSmoke.commands.Add($command)
  if ($command -match 'nohup|scripts.run_mastery_experiment') { throw 'Training launch reached a preflight-only test.' }
  if ($global:StairLaunchSmoke.failAll -or $global:StairLaunchSmoke.retryOnce) {
    $global:StairLaunchSmoke.retryOnce = $false
    $global:LASTEXITCODE = 255
  } else { $global:LASTEXITCODE = 0 }
}
try {
  & "$PSScriptRoot/start_mastery_experiment.ps1" -RunId mocked-readiness -InitActorOnnx /synthetic/policy.onnx
  if ($global:StairLaunchSmoke.commands.Count -ne 5 -or -not ($global:StairLaunchSmoke.commands | Where-Object {$_ -match 'server_results_service start'})) { throw 'Retry or server-delivery preflight coverage was incorrect.' }
  $global:StairLaunchSmoke.failAll = $true
  $caught = $false
  try {
    & "$PSScriptRoot/start_mastery_experiment.ps1" -RunId mocked-unreachable -InitActorOnnx /synthetic/policy.onnx
  } catch { $caught = $true }
  if (-not $caught -or $global:StairLaunchSmoke.commands.Count -ne 8) { throw 'Transport failure did not stop after three attempts.' }
  'PASS launch guard: preflight retries bounded; no training command executed'
} finally {
  Remove-Item Function:\gh,Function:\ssh -ErrorAction SilentlyContinue
  Remove-Variable -Name StairLaunchSmoke -Scope Global -ErrorAction SilentlyContinue
}
