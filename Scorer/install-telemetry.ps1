param([string]$InstancePath='D:/Minecraft/JavaEdition/BlockHelm/.minecraft/versions/26.1-fabric-0.19.3')
$ErrorActionPreference='Stop'
$taskModFolder=[IO.Path]::GetFullPath((Join-Path $InstancePath 'mods'))
if(!(Test-Path -LiteralPath $taskModFolder -PathType Container)){throw "Minecraft mods directory missing: $taskModFolder"}
$taskSource=Join-Path $PSScriptRoot 'build/libs/vabpnn-telemetry-0.5.0.jar'
if(!(Test-Path -LiteralPath $taskSource)){throw 'Build the telemetry JAR first'}
$taskTarget=Join-Path $taskModFolder 'vabpnn-telemetry-0.5.0.jar'
$taskExisting=Get-ChildItem -LiteralPath $taskModFolder -File -Filter 'vabpnn-telemetry-*.jar'
$taskAllowed=@('vabpnn-telemetry-0.5.0.jar','vabpnn-telemetry-0.4.0.jar','vabpnn-telemetry-0.3.0.jar','vabpnn-telemetry-0.2.0.jar','vabpnn-telemetry-0.1.0.jar')
if($taskExisting | Where-Object {$_.Name -notin $taskAllowed}){throw 'An unknown telemetry version exists; choose which telemetry version to keep'}
$taskStamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff')
if(Test-Path -LiteralPath $taskTarget){
 $taskBackup=Join-Path $PSScriptRoot ('build/install-backup/'+$taskStamp)
 New-Item -ItemType Directory -Path $taskBackup -Force | Out-Null
 Copy-Item -LiteralPath $taskTarget -Destination $taskBackup
}
$taskHash=(Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash
$taskPending=Join-Path $taskModFolder 'vabpnn-telemetry-0.5.0.jar.installing'
Copy-Item -LiteralPath $taskSource -Destination $taskPending -Force
if((Get-FileHash -LiteralPath $taskPending -Algorithm SHA256).Hash -ne $taskHash){throw 'Staged JAR checksum mismatch'}
$taskRenamed=[Collections.Generic.List[object]]::new()
try {
 foreach($taskOld in $taskExisting | Where-Object {$_.Name -ne 'vabpnn-telemetry-0.5.0.jar'}){
  $taskDisabled=$taskOld.Name+'.'+$taskStamp+'.disabled'
  Rename-Item -LiteralPath $taskOld.FullName -NewName $taskDisabled
  $taskRenamed.Add(@{Original=$taskOld.Name;Disabled=$taskDisabled})
 }
 # Move one staged file between literal paths under the verified mods directory.
 Move-Item -LiteralPath $taskPending -Destination $taskTarget -Force
} catch {
 foreach($taskOld in $taskRenamed){Rename-Item -LiteralPath (Join-Path $taskModFolder $taskOld.Disabled) -NewName $taskOld.Original}
 throw
}
if((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash -ne $taskHash){throw 'Installed JAR checksum mismatch'}
$taskManifest=[ordered]@{installed_utc=[DateTimeOffset]::UtcNow.ToString('o');target=$taskTarget;telemetry_sha256=$taskHash;scorer_sha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'build/libs/vabpnn-scorer-0.5.0.jar') -Algorithm SHA256).Hash}
$taskManifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'build/deployment.json')
Write-Output ("Installed telemetry: "+$taskTarget)
Write-Output ("SHA256: "+$taskHash)