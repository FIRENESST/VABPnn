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
# Fingerprints follow effective compiler inputs; a test/helper edit is not a model edit.
function Get-TaskSourceHeaderSignature([string]$Source) {
    $taskSourcePath=[IO.Path]::GetFullPath($Source)
    if($taskDependencyCache.ContainsKey($taskSourcePath)) {return $taskDependencyCache[$taskSourcePath]}
    $taskSeen=New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $taskPending=New-Object 'Collections.Generic.Queue[string]'
    $taskSdkSeen=New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $taskPending.Enqueue($taskSourcePath)
    while($taskPending.Count) {
        $taskPath=$taskPending.Dequeue()
        if(!$taskSeen.Add($taskPath)) {continue}
        $taskText=[IO.File]::ReadAllText($taskPath)
        foreach($taskInclude in [regex]::Matches($taskText,'(?m)^\s*#\s*include\s*["<]([^">]+)[">]')) {
            $taskName=$taskInclude.Groups[1].Value
            $taskResolved=$null
            foreach($taskDirectory in @([IO.Path]::GetDirectoryName($taskPath),$taskNativeRoot,$taskOperatorRoot)) {
                $taskCandidate=[IO.Path]::GetFullPath((Join-Path $taskDirectory $taskName))
                if(Test-Path -LiteralPath $taskCandidate -PathType Leaf) {$taskResolved=$taskCandidate;break}
            }
            if($taskResolved) {$taskPending.Enqueue($taskResolved)}
            elseif($taskInclude.Value.Contains('"')) {
                # SDK headers are also valid when written with quotes. Hash the
                # directly included SDK file; it does not become a project header.
                foreach($taskDirectory in $taskSdkRoots) {
                    if(!$taskDirectory) {continue}
                    $taskCandidate=[IO.Path]::GetFullPath([IO.Path]::Combine($taskDirectory,$taskName))
                    if(Test-Path -LiteralPath $taskCandidate -PathType Leaf) {
                        [void]$taskSdkSeen.Add($taskCandidate);$taskResolved=$taskCandidate;break
                    }
                }
                if(!$taskResolved) {throw ('Unresolved header: '+$taskName+' in '+$taskPath)}
            }
        }
        # Unknown macro includes use the conservative project-wide header set.
        if([regex]::IsMatch($taskText,'(?m)^\s*#\s*include\s+[^"<\s]')) {
            foreach($taskDirectory in @($taskNativeRoot,$taskOperatorRoot)) {
                Get-ChildItem -LiteralPath $taskDirectory -File -Recurse |
                    Where-Object { $_.Extension -in @('.h','.hpp','.cuh') -and $_.FullName -notmatch $taskIgnoredDirectories } |
                    ForEach-Object {$taskPending.Enqueue($_.FullName)}
            }
        }
    }
    $taskItems=@('project-include-graph-v1')
    foreach($taskPath in (@($taskSeen)+@($taskSdkSeen) | Sort-Object -Unique)) {
        if($taskPath -ne $taskSourcePath) {$taskItems+=@($taskPath,(Get-TaskFileHash $taskPath))}
    }
    $taskSignature=Get-TaskSignature $taskItems
    $taskDependencyCache[$taskSourcePath]=$taskSignature
    return $taskSignature
}
function Get-TaskBuildLock {
    $taskLockPath=Join-Path $taskBuild '.native-build.lock'
    $taskWait=[Diagnostics.Stopwatch]::StartNew()
    while($true) {
        try {return [IO.File]::Open($taskLockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)}
        catch [IO.IOException] {
            if($taskWait.Elapsed.TotalSeconds -gt 120) {throw 'Another native build still owns this cache; retry after it finishes'}
            if($taskWait.ElapsedMilliseconds -lt 250) {Write-Host 'Waiting for the other native build to release the shared cache...'}
            Start-Sleep -Milliseconds 200
        }
    }
}
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
            @($taskSource, (Get-TaskFileHash $taskSource), (Get-TaskSourceHeaderSignature $taskSource)))
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

function Get-TaskLinkedProgram([string]$Source,[string]$Target,[string[]]$CodeArguments,
        [string[]]$Objects,[string[]]$Libraries) {
    $taskIdentity=@($taskCommonSignature)+$CodeArguments+$Libraries+
        @($Source,(Get-TaskFileHash $Source),(Get-TaskSourceHeaderSignature $Source))
    foreach($taskObject in $Objects) {$taskIdentity+=(Get-TaskFileHash $taskObject)}
    $taskSignature=Get-TaskSignature $taskIdentity
    $taskStamp=$Target+'.sha256'
    $taskLines=@()
    if(Test-Path -LiteralPath $taskStamp) {$taskLines=@(Get-Content -LiteralPath $taskStamp)}
    $taskCached=$taskLines.Count -eq 2 -and $taskLines[0] -eq $taskSignature -and (Test-Path -LiteralPath $Target)
    if($taskCached) {$taskCached=(Get-TaskFileHash $Target) -eq $taskLines[1]}
    if(!$taskCached) {
        $taskPendingTarget=$Target+'.pending'+[IO.Path]::GetExtension($Target)
        $taskArguments=$taskCommon+$CodeArguments+$Objects+$Libraries+@($Source,'-o',$taskPendingTarget)
        & $taskNvcc @taskArguments | Out-Host
        if($LASTEXITCODE -ne 0) {throw ('Native program build failed: '+$Source)}
        Move-Item -LiteralPath $taskPendingTarget -Destination $Target -Force
        [IO.File]::WriteAllLines($taskStamp,[string[]]@($taskSignature,(Get-TaskFileHash $Target)),$taskUtf8)
    } else {Write-Host ('Program cache hit: '+[IO.Path]::GetFileName($Target))}
    return $Target
}
