param(
    [string]$OutputDirectory,
    [string]$OperatorRoot,
    [string]$CudaRoot = 'C:/CUDAToolkit',
    [string]$JavaRoot = 'C:/Java/JDK26',
    [string]$VsRoot = 'C:/Program Files/Microsoft Visual Studio/18/Community',
    [switch]$TestOperator,
    [switch]$Benchmark
)
$ErrorActionPreference = 'Stop'
$taskProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $taskProject 'build/generated/nativeResources/native/windows-x86_64' }
if (-not $OperatorRoot) { $OperatorRoot = Join-Path $taskProject '../CUDAkernel/SelfCudaKernel/SelfCudaKernel' }
$taskOperatorRoot = [IO.Path]::GetFullPath($OperatorRoot)
$taskOutput = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $taskOutput.StartsWith($taskProject + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Native output must stay in the project' }
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
[IO.Directory]::CreateDirectory($taskOutput) | Out-Null
$taskBuild = Join-Path $taskProject 'build/native'
[IO.Directory]::CreateDirectory($taskBuild) | Out-Null
Push-Location -LiteralPath $taskBuild
try {
    Write-Output ("Operator source directory: " + $taskOperatorRoot)
    $taskCommon = @('--cudart', 'static', '-std=c++17', '-O3',
        '-D', 'VABPNN_OPERATOR_LIBRARY=1', '-Xcompiler', '/MD,/EHsc,/O2,/Zc:preprocessor',
        '-I', (Join-Path $JavaRoot 'include'), '-I', (Join-Path $JavaRoot 'include/win32'),
        '-I', (Join-Path $CudaRoot 'include/cccl'), '-I', $PSScriptRoot, '-I', $taskOperatorRoot)
    $taskProductionCode = @('-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_89,code=sm_89', '-gencode=arch=compute_120,code=sm_120',
        '-gencode=arch=compute_120,code=compute_120')
    $taskTestCode = @('-arch=sm_120')
    $taskUtf8 = New-Object Text.UTF8Encoding($false)

    function Get-TaskSignature([string[]]$Items) {
        $taskDigest = [Security.Cryptography.SHA256]::Create()
        try {
            $taskBytes = [Text.Encoding]::UTF8.GetBytes(($Items -join "`n"))
            return [BitConverter]::ToString($taskDigest.ComputeHash($taskBytes)).Replace('-', '')
        } finally { $taskDigest.Dispose() }
    }
    # Use .NET directly: Gradle's Windows PowerShell environment may not
    # auto-load Microsoft.PowerShell.Utility (Get-FileHash).
    function Get-TaskFileHash([string]$Path) {
        $taskFile = [IO.File]::OpenRead($Path)
        $taskDigest = [Security.Cryptography.SHA256]::Create()
        try {
            return [BitConverter]::ToString($taskDigest.ComputeHash($taskFile)).Replace('-', '')
        } finally { $taskDigest.Dispose(); $taskFile.Dispose() }
    }
    # Header changes invalidate all translation units; a .cu change recompiles only that unit.
    # Track compiler/toolchain/flags as well, so a stale object cannot cross build variants.
    $taskHeaders = @(Get-ChildItem -LiteralPath $taskOperatorRoot -File -Recurse |
        Where-Object { $_.Extension -in @('.h','.hpp','.cuh') -and
            $_.FullName.Substring($taskOperatorRoot.Length) -notmatch $taskIgnoredDirectories } |
        Select-Object -ExpandProperty FullName)
    $taskHeaders += @(Get-ChildItem -LiteralPath $PSScriptRoot -File |
        Where-Object Extension -in @('.h','.hpp','.cuh') | Select-Object -ExpandProperty FullName)
    $taskHeaders += @((Join-Path $JavaRoot 'include/jni.h'), (Join-Path $JavaRoot 'include/win32/jni_md.h'),
        (Join-Path $CudaRoot 'include/cuda_runtime.h'), (Join-Path $CudaRoot 'include/cuda_runtime_api.h'),
        (Join-Path $CudaRoot 'include/cccl/cub/version.cuh'), (Join-Path $CudaRoot 'include/cccl/thrust/version.h'))
    $taskIdentity = @($taskCommon) + @($taskNvcc, $taskOperatorRoot, $env:VCToolsVersion,
        $env:WindowsSDKVersion, $env:INCLUDE, (Get-TaskFileHash $taskNvcc),
        (Get-TaskFileHash (Get-Command cl.exe -ErrorAction Stop).Source),
        (Get-TaskFileHash $PSCommandPath))
    foreach ($taskHeader in ($taskHeaders | Sort-Object -Unique)) {
        if (Test-Path -LiteralPath $taskHeader) {
            $taskIdentity += @($taskHeader, (Get-TaskFileHash $taskHeader))
        }
    }
    $taskCommonSignature = Get-TaskSignature $taskIdentity

    function Get-TaskOperatorObjects([string]$Kind, [string[]]$CodeArguments) {
        $taskCache = Join-Path $taskBuild ('obj/' + $Kind)
        [IO.Directory]::CreateDirectory($taskCache) | Out-Null
        $taskObjects = @()
        $taskCompiled = 0
        foreach ($taskSource in $taskOperatorSources) {
            $taskName = [IO.Path]::GetFileNameWithoutExtension($taskSource) + '-' +
                (Get-TaskSignature @($taskSource)).Substring(0,12) + '.obj'
            $taskObject = Join-Path $taskCache $taskName
            $taskStamp = $taskObject + '.sha256'
            $taskSignature = Get-TaskSignature (@($taskCommonSignature) + $CodeArguments +
                @($taskSource, (Get-TaskFileHash $taskSource)))
            $taskStampLines = @()
            if (Test-Path -LiteralPath $taskStamp) { $taskStampLines = @(Get-Content -LiteralPath $taskStamp) }
            $taskCached = $taskStampLines.Count -eq 2 -and $taskStampLines[0] -eq $taskSignature -and
                (Test-Path -LiteralPath $taskObject)
            if ($taskCached) {
                $taskCached = (Get-TaskFileHash $taskObject) -eq $taskStampLines[1]
            }
            if (!$taskCached) {
                Write-Host ('Compile ' + $Kind + ': ' + [IO.Path]::GetFileName($taskSource))
                $taskPendingObject = $taskObject + '.pending.obj'
                $taskArguments = $taskCommon + $CodeArguments + @('-c', $taskSource, '-o', $taskPendingObject)
                & $taskNvcc @taskArguments | Out-Host
                if ($LASTEXITCODE -ne 0) { throw ('CUDA object build failed: ' + $taskSource) }
                Move-Item -LiteralPath $taskPendingObject -Destination $taskObject -Force
                [IO.File]::WriteAllLines($taskStamp, [string[]]@($taskSignature,
                    (Get-TaskFileHash $taskObject)), $taskUtf8)
                $taskCompiled++
            }
            $taskObjects += $taskObject
        }
        Write-Host ('Operator cache ' + $Kind + ': compiled=' + $taskCompiled +
            ', reused=' + ($taskObjects.Count - $taskCompiled))
        return $taskObjects
    }

    $taskProductionObjects = @(Get-TaskOperatorObjects 'production' $taskProductionCode)
    $taskDll = Join-Path $taskBuild 'vabpnn_capture.dll'
    $taskDllStamp = $taskDll + '.sha256'
    $taskBridge = Join-Path $PSScriptRoot 'capture_bridge.cu'
    $taskLinkIdentity = @($taskCommonSignature) + $taskProductionCode +
        @((Get-TaskFileHash $taskBridge))
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

    if ($TestOperator -or $Benchmark) {
        # All numerical tests and the benchmark share one sm_120 operator build.
        $taskTestObjects = @(Get-TaskOperatorObjects 'test' $taskTestCode)
    }
    if ($TestOperator) {
        foreach ($taskTestName in @('operator_test','neuron_vote_test','temporal_decision_test','jepa_motion_test','online_learning_test','spike_connections_test','network_scope_test','player_position_test','block_controllers_test')) {
            $taskTestExe = Join-Path $taskBuild ($taskTestName + '.exe')
            $taskArguments = $taskCommon + $taskTestCode + $taskTestObjects +
                @('bcrypt.lib', (Join-Path $PSScriptRoot ($taskTestName + '.cu')), '-o', $taskTestExe)
            & $taskNvcc @taskArguments
            if ($LASTEXITCODE -ne 0) { throw ('CUDA test build failed: ' + $taskTestName) }
            & $taskTestExe
            if ($LASTEXITCODE -ne 0) { throw ('CUDA verification failed: ' + $taskTestName) }
        }
    }
    if ($Benchmark) {
        $taskBenchmarkExe = Join-Path $taskBuild 'frame_budget_benchmark.exe'
        $taskArguments = $taskCommon + $taskTestCode + $taskTestObjects +
            @('bcrypt.lib', (Join-Path $PSScriptRoot 'frame_budget_benchmark.cu'), '-o', $taskBenchmarkExe)
        & $taskNvcc @taskArguments
        if ($LASTEXITCODE -ne 0) { throw 'Frame benchmark build failed' }
        & $taskBenchmarkExe
        if ($LASTEXITCODE -ne 0) { throw 'Frame benchmark failed' }
    }
} finally { Pop-Location }

