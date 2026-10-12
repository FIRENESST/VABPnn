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
$taskNativeRoot=$PSScriptRoot
$taskBuild=Join-Path $taskProject 'build/native'
[IO.Directory]::CreateDirectory($taskOutput) | Out-Null
[IO.Directory]::CreateDirectory($taskBuild) | Out-Null
. (Join-Path $taskNativeRoot 'build/cache.ps1')
$taskLock=Get-TaskBuildLock
Push-Location -LiteralPath $taskBuild
try {
    Write-Output ("Operator source directory: "+$taskOperatorRoot)
    . (Join-Path $taskNativeRoot 'build/toolchain.ps1')
    . (Join-Path $taskNativeRoot 'build/production.ps1')
    . (Join-Path $taskNativeRoot 'build/verification.ps1')
} finally {
    Pop-Location
    $taskLock.Dispose()
}
