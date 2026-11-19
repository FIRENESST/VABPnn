param(
    [string]$SourceInstance = 'D:\Minecraft\JavaEdition\BlockHelm\.minecraft\versions\26.1-fabric-0.19.3',
    [string]$WorldName = '新的世界',
    [string]$JavaRoot = 'C:\Java\JDK26',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$runtimePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'run\background-training'))
$projectPath = Split-Path -Parent $PSScriptRoot
$argumentPath = Join-Path $runtimePath 'java-arguments.txt'
$processFile = Join-Path $runtimePath 'process.json'
$timeTag = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff')
if (Test-Path -LiteralPath $processFile) {
    $previous = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    $active = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$previous.pid) -ErrorAction SilentlyContinue
    if ($active -and $active.CommandLine.Contains($argumentPath)) {
        Write-Output ('Background training is already running, PID ' + $previous.pid)
        & (Join-Path $PSScriptRoot 'background-training-status.ps1')
        return
    }
}
$handler = [Net.Http.HttpClientHandler]::new(); $handler.UseProxy = $false
$http = [Net.Http.HttpClient]::new($handler); $http.Timeout = [TimeSpan]::FromSeconds(3)
try {
    try { $health = $http.GetStringAsync('http://127.0.0.1:9876/health').GetAwaiter().GetResult() | ConvertFrom-Json }
    catch {
        if (Get-NetTCPConnection -LocalPort 9876 -State Listen -ErrorAction SilentlyContinue) {
            throw 'Port 9876 is occupied by an unverified service; existing process was left unchanged.'
        }
        $scorerPath = Join-Path $projectPath 'Scorer'
        $scorerJar = Join-Path $scorerPath 'build\libs\vabpnn-scorer-0.5.0.jar'
        if (!(Test-Path -LiteralPath $scorerJar)) { throw 'Build Scorer 0.5.0 before starting background training.' }
        $scorerData = Join-Path $scorerPath 'data'
        $scorerProcess = Start-Process -FilePath (Join-Path $JavaRoot 'bin\java.exe') -ArgumentList @(
            '-jar', ('"' + $scorerJar + '"'), '--headless', '--data', ('"' + $scorerData + '"'), '--port', '9876'
        ) -WorkingDirectory $scorerPath -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $scorerData ('service-' + $timeTag + '-stdout.log')) `
            -RedirectStandardError (Join-Path $scorerData ('service-' + $timeTag + '-stderr.log'))
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            Start-Sleep -Milliseconds 250
            try { $health = $http.GetStringAsync('http://127.0.0.1:9876/health').GetAwaiter().GetResult() | ConvertFrom-Json; break } catch {}
        }
    }
    if ($health.service -ne 'vabpnn-scorer' -or $health.training_protocol -ne 'vabpnn-reward-v1') {
        throw 'A compatible local Scorer service is required.'
    }
} finally { $http.Dispose(); $handler.Dispose() }
New-Item -Path $runtimePath -ItemType Directory -Force | Out-Null
$markerPath = Join-Path $runtimePath '.vabpnn-background'
Set-Content -LiteralPath $markerPath -Value 'VABPnn isolated hidden background training; explicitly requested by user' -Encoding utf8
$manifestFile = Join-Path $runtimePath 'seed-manifest.json'
if (!(Test-Path -LiteralPath $manifestFile)) {
    $sourceWorld = Join-Path $SourceInstance ('saves\' + $WorldName)
    $savePath = Join-Path $runtimePath 'saves\training-world'
    if (!(Test-Path -LiteralPath $sourceWorld -PathType Container)) { throw ('Missing source world: ' + $sourceWorld) }
    if (Test-Path -LiteralPath $savePath) { throw 'Existing isolated save has no seed manifest; refusing to overwrite it.' }
    $sourceModel = @('release-1.4.2.bin', 'release-1.4.1.bin', 'release-1.4.0.bin') |
        ForEach-Object { Join-Path $SourceInstance ('vabpnn\models\' + $_) } |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (!$sourceModel) { throw 'No compatible formal model checkpoint found; no fresh model was silently substituted.' }
    $worldHashes = @{}
    Get-ChildItem -LiteralPath $sourceWorld -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($sourceWorld.Length + 1)
        $worldHashes[$relative] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
    New-Item -Path (Split-Path -Parent $savePath) -ItemType Directory -Force | Out-Null
    Copy-Item -LiteralPath $sourceWorld -Destination $savePath -Recurse
    $copiedId = Join-Path $savePath 'data\vabpnn-scorer-world-id.txt'
    if (Test-Path -LiteralPath $copiedId) {
        Move-Item -LiteralPath $copiedId -Destination (Join-Path $savePath 'data\vabpnn-scorer-source-world-id.txt')
    }
    $modelDirectory = Join-Path $runtimePath 'vabpnn\models'
    New-Item -Path $modelDirectory -ItemType Directory -Force | Out-Null
    # Retain the legacy filename so the Alpha loader can inherit the PNN without changing the seed.
    Copy-Item -LiteralPath $sourceModel -Destination (Join-Path $modelDirectory (Split-Path -Leaf $sourceModel))
    $sourceMemory = Join-Path $SourceInstance 'vabpnn\memory\experience-v1.vdb'
    $memoryHash = $null
    if (Test-Path -LiteralPath $sourceMemory) {
        $memoryDirectory = Join-Path $runtimePath 'vabpnn\memory'
        New-Item -Path $memoryDirectory -ItemType Directory -Force | Out-Null
        Copy-Item -LiteralPath $sourceMemory -Destination (Join-Path $memoryDirectory 'experience-v1.vdb')
        $memoryHash = (Get-FileHash -LiteralPath $sourceMemory -Algorithm SHA256).Hash
    }
    [pscustomobject]@{created_at=[DateTime]::UtcNow.ToString('o');source_model=$sourceModel;
        source_model_sha256=(Get-FileHash -LiteralPath $sourceModel -Algorithm SHA256).Hash;
        source_world=$sourceWorld;source_world_sha256=$worldHashes;source_memory=$sourceMemory;
        source_memory_sha256=$memoryHash} | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $manifestFile -Encoding utf8
}
$optionsFile = Join-Path $runtimePath 'options.txt'
if (!(Test-Path -LiteralPath $optionsFile)) {
    @('fullscreen:false','enableVsync:false','maxFps:30','renderDistance:8','simulationDistance:6',
      'pauseOnLostFocus:false','onboardAccessibility:false','soundCategory_master:0.0') |
        Set-Content -LiteralPath $optionsFile -Encoding utf8
}
if (!$SkipBuild) {
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & .\gradlew.bat --offline -PbackgroundTraining prepareBackgroundTraining *> (Join-Path $runtimePath 'prepare.log')
        if ($LASTEXITCODE -ne 0) { throw ('Background runtime build failed; see ' + (Join-Path $runtimePath 'prepare.log')) }
    } finally { Pop-Location }
}
$launch = Get-Content -LiteralPath (Join-Path $runtimePath 'launch-info.json') -Raw | ConvertFrom-Json
if ($launch.workingDirectory -ne $runtimePath -or !($launch.jvmArgs -contains '-Dvabpnn.background.training=true') `
    -or !($launch.jvmArgs -contains '-Dvabpnn.selftrain.modules=all') `
    -or !($launch.jvmArgs | Where-Object { $_ -like '-Dvabpnn.model.path=*release-2.0.0-alpha.bin' }) `
    -or !($launch.args -contains '--quickPlaySingleplayer')) { throw 'Invalid background launch configuration.' }
$backgroundResources = Join-Path $PSScriptRoot 'build\resources\backgroundTraining'
$addon = Get-Content -LiteralPath (Join-Path $backgroundResources 'fabric.mod.json') -Raw | ConvertFrom-Json
$mixins = Get-Content -LiteralPath (Join-Path $backgroundResources 'vabpnn.background.mixins.json') -Raw | ConvertFrom-Json
if ($addon.id -ne 'vabpnn-background-runtime' -or !$mixins.required -or $mixins.client.Count -ne 4 `
    -or !($launch.classpath -contains $backgroundResources)) { throw 'Required invisible-window/input mixins are missing.' }
$stopRequest = Join-Path $runtimePath 'stop.request'
if (Test-Path -LiteralPath $stopRequest) {
    Move-Item -LiteralPath $stopRequest -Destination (Join-Path $runtimePath ('stop.ack-' + $timeTag))
}
$stdoutFile = Join-Path $runtimePath ('java-' + $timeTag + '-stdout.log')
$stderrFile = Join-Path $runtimePath ('java-' + $timeTag + '-stderr.log')
$backgroundProcess = Start-Process -FilePath (Join-Path $JavaRoot 'bin\javaw.exe') `
    -ArgumentList ('@"' + $argumentPath + '"') -WorkingDirectory $runtimePath -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile -PassThru
[pscustomobject]@{pid=$backgroundProcess.Id;started_utc=[DateTime]::UtcNow.ToString('o');
    stdout=$stdoutFile;stderr=$stderrFile;arguments=$argumentPath} | ConvertTo-Json |
    Set-Content -LiteralPath $processFile -Encoding utf8
Write-Output ('Background training launched invisibly, PID ' + $backgroundProcess.Id)
Write-Output ('Status: ' + (Join-Path $runtimePath 'status.json'))
Write-Output 'Exploration stays OFF. Original world and checkpoints are preserved; this runner uses copies.'
