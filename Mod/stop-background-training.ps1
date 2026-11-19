$ErrorActionPreference = 'Stop'
$runtimePath = Join-Path $PSScriptRoot 'run\background-training'
$processFile = Join-Path $runtimePath 'process.json'
if (!(Test-Path -LiteralPath $processFile)) { Write-Output 'Background training is not running.'; return }
$details = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
$active = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$details.pid) -ErrorAction SilentlyContinue
if (!$active -or !$active.CommandLine.Contains([string]$details.arguments)) {
    Write-Output 'Background training is already stopped.'; return
}
Set-Content -LiteralPath (Join-Path $runtimePath 'stop.request') -Value 'User requested graceful stop and checkpoint' -Encoding utf8
for ($attempt=0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Milliseconds 500
    if (!(Get-Process -Id ([int]$details.pid) -ErrorAction SilentlyContinue)) {
        Write-Output 'Background training stopped; model and isolated world saved. Scorer remains available.'; return
    }
}
throw 'Graceful stop has not finished yet. Stop request remains pending; no process was force-killed.'
