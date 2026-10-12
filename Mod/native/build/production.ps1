$taskProductionObjects = @(Get-TaskOperatorObjects 'production' $taskProductionCode)
$taskDll = Join-Path $taskBuild 'vabpnn_capture.dll'
$taskDllStamp = $taskDll + '.sha256'
$taskBridge = Join-Path $taskNativeRoot 'capture_bridge.cu'
$taskLinkIdentity = @($taskCommonSignature) + $taskProductionCode +
    @((Get-TaskFileHash $taskBridge),(Get-TaskSourceHeaderSignature $taskBridge))
foreach ($taskObject in $taskProductionObjects) {
    $taskLinkIdentity += (Get-TaskFileHash $taskObject)
}
$taskLinkSignature = Get-TaskSignature $taskLinkIdentity
$taskDllStampLines = @()
if (Test-Path -LiteralPath $taskDllStamp) { $taskDllStampLines = @(Get-Content -LiteralPath $taskDllStamp) }
$taskDllCached = $taskDllStampLines.Count -eq 2 -and $taskDllStampLines[0] -eq $taskLinkSignature -and
    (Test-Path -LiteralPath $taskDll)
if ($taskDllCached) {
    $taskDllCached = (Get-TaskFileHash $taskDll) -eq $taskDllStampLines[1]
}
if (!$taskDllCached) {
    $taskPendingDll = Join-Path $taskBuild 'vabpnn_capture.pending.dll'
    $taskArguments = @('--shared') + $taskCommon + $taskProductionCode + @($taskBridge) +
        $taskProductionObjects + @('opengl32.lib', 'user32.lib', 'gdi32.lib', 'bcrypt.lib', '-o', $taskPendingDll)
    & $taskNvcc @taskArguments
    if ($LASTEXITCODE -ne 0) { throw 'Native CUDA DLL build failed' }
    Move-Item -LiteralPath $taskPendingDll -Destination $taskDll -Force
    [IO.File]::WriteAllLines($taskDllStamp, [string[]]@($taskLinkSignature,
        (Get-TaskFileHash $taskDll)), $taskUtf8)
} else { Write-Host 'Native DLL cache hit' }
$taskPackagedDll = Join-Path $taskOutput 'vabpnn_capture.dll'
if (!(Test-Path -LiteralPath $taskPackagedDll) -or
    (Get-TaskFileHash $taskPackagedDll) -ne
    (Get-TaskFileHash $taskDll)) {
    Copy-Item -LiteralPath $taskDll -Destination $taskPackagedDll -Force
}

