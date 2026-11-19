$runtimePath = Join-Path $PSScriptRoot 'run\background-training'
$processFile = Join-Path $runtimePath 'process.json'
$running = $false
if (Test-Path -LiteralPath $processFile) {
    $details = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $active = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$details.pid) -ErrorAction SilentlyContinue
    $running = [bool]($active -and $active.CommandLine.Contains([string]$details.arguments))
}
$statusFile = Join-Path $runtimePath 'status.json'
if (Test-Path -LiteralPath $statusFile) {
    $state = Get-Content -LiteralPath $statusFile -Raw -Encoding utf8 | ConvertFrom-Json
    $state | Add-Member -NotePropertyName running -NotePropertyValue $running -Force
    $state | ConvertTo-Json -Depth 4
} else { [pscustomobject]@{running=$running;phase='starting_or_not_launched'} | ConvertTo-Json }
