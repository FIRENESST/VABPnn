$taskNvcc = Join-Path $CudaRoot 'bin/nvcc.exe'
$taskVcvars = Join-Path $VsRoot 'VC/Auxiliary/Build/vcvars64.bat'
foreach ($taskRequired in @($taskNvcc, $taskVcvars, (Join-Path $JavaRoot 'include/jni.h'),
        (Join-Path $taskOperatorRoot 'GpuFrame.h'), (Join-Path $taskOperatorRoot 'FrameOperator.h'),
        (Join-Path $taskOperatorRoot 'File.cu'), (Join-Path $taskOperatorRoot 'NeuronModel.h'),
        (Join-Path $taskOperatorRoot 'NeuronModel.cu'), (Join-Path $taskOperatorRoot 'TemporalDecision.h'),
        (Join-Path $taskOperatorRoot 'TemporalDecision.cu'), (Join-Path $taskOperatorRoot 'JepaMotion.h'),
        (Join-Path $taskOperatorRoot 'JepaMotion.cu'))) {
    if (-not (Test-Path -LiteralPath $taskRequired)) { throw "Missing build dependency: $taskRequired" }
}
# Compile operator sources, excluding generated CUDA intermediates in VS output folders.
# VABPNN_OPERATOR_LIBRARY omits the standalone main() in File.cu.
$taskIgnoredDirectories = '(^|[\\/])(x64|Win32|Debug|Release|\.vs|build|obj)([\\/]|$)'
$taskOperatorSources = @(Get-ChildItem -LiteralPath $taskOperatorRoot -File -Recurse |
    Where-Object { $_.Extension -in @('.cu','.cpp') -and
        $_.FullName.Substring($taskOperatorRoot.Length) -notmatch $taskIgnoredDirectories } |
    Sort-Object FullName | Select-Object -ExpandProperty FullName)
$taskEnvironment = & cmd.exe /d /s /c ('call "' + $taskVcvars + '" >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
foreach ($taskLine in $taskEnvironment) {
    if ($taskLine -match '^([^=]+)=(.*)$' -and $Matches[1] -notin @('HOME','CODEX_HOME')) {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$taskCommon = @('--cudart', 'static', '-std=c++17', '-O3',
    '-D', 'VABPNN_OPERATOR_LIBRARY=1', '-Xcompiler', '/MD,/EHsc,/O2,/Zc:preprocessor',
    '-I', (Join-Path $JavaRoot 'include'), '-I', (Join-Path $JavaRoot 'include/win32'),
    '-I', (Join-Path $CudaRoot 'include/cccl'), '-I', $taskNativeRoot, '-I', $taskOperatorRoot)
$taskProductionCode = @('-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_89,code=sm_89', '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_120,code=compute_120')
$taskTestCode = @('-arch=sm_120')
$taskUtf8 = New-Object Text.UTF8Encoding($false)


$taskDependencyCache=@{}
$taskSdkRoots=@((Join-Path $JavaRoot 'include'),(Join-Path $JavaRoot 'include/win32'),
    (Join-Path $CudaRoot 'include/cccl'),(Join-Path $CudaRoot 'include'))+@($env:INCLUDE -split ';')
$taskIdentity=@('vabpnn-native-cache-v2')+@($taskCommon)+@($taskNvcc,$taskOperatorRoot,
    $env:VCToolsVersion,$env:WindowsSDKVersion,$env:INCLUDE,(Get-TaskFileHash $taskNvcc),
    (Get-TaskFileHash (Get-Command cl.exe -ErrorAction Stop).Source))
$taskSystemHeaders=@((Join-Path $JavaRoot 'include/jni.h'),(Join-Path $JavaRoot 'include/win32/jni_md.h'),
    (Join-Path $CudaRoot 'include/cuda_runtime.h'),(Join-Path $CudaRoot 'include/cuda_runtime_api.h'),
    (Join-Path $CudaRoot 'include/cccl/cub/version.cuh'),(Join-Path $CudaRoot 'include/cccl/thrust/version.h'))
foreach($taskHeader in ($taskSystemHeaders | Sort-Object -Unique)) {
    if(Test-Path -LiteralPath $taskHeader) {$taskIdentity+=@($taskHeader,(Get-TaskFileHash $taskHeader))}
}
$taskCommonSignature=Get-TaskSignature $taskIdentity
