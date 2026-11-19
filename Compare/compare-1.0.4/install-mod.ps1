param([string]$InstancePath='D:\Minecraft\JavaEdition\BlockHelm\.minecraft\versions\26.1-fabric-0.19.3')
$ErrorActionPreference='Stop'
$taskRoot=[IO.Path]::GetFullPath('C:\Users\43828\Desktop\MyProject\VABPnn')
$taskProject=Join-Path $taskRoot 'Compare\compare-1.0.4'
$taskMods=[IO.Path]::GetFullPath((Join-Path $InstancePath 'mods'))
$taskModels=[IO.Path]::GetFullPath((Join-Path $InstancePath 'vabpnn\models'))
$taskName='vabpnn-compare-1.0.4.jar'
$taskSource=Join-Path $taskProject ('Mod\build\libs\'+$taskName)
$taskTarget=[IO.Path]::GetFullPath((Join-Path $taskMods $taskName))
$taskPending=[IO.Path]::GetFullPath($taskTarget+'.installing')
foreach($taskPath in @($taskTarget,$taskPending)){
 if(!$taskPath.StartsWith($taskMods+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Install outside verified mods directory'}
}
if(!(Test-Path -LiteralPath $taskMods -PathType Container)){throw 'Mods directory missing'}
if(!(Test-Path -LiteralPath $taskSource -PathType Leaf)){throw 'Build the release first'}
if(Test-Path -LiteralPath $taskTarget){throw 'Target exists; preserve it rather than overwrite'}
$taskHash=(Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash
# Compare fixed prior-source snapshots, never modifying previous version roots.
$taskPreserved=Get-Content -LiteralPath (Join-Path $taskProject 'unchanged-version-hashes.json') -Raw|ConvertFrom-Json
foreach($taskEntry in $taskPreserved){
 if((Get-FileHash -LiteralPath (Join-Path $taskRoot $taskEntry.path) -Algorithm SHA256).Hash -ne $taskEntry.sha256){throw ('Prior source changed: '+$taskEntry.path)}
}
$taskHistory=@(Get-ChildItem -LiteralPath $taskMods -File -Filter 'vabpnn*.jar*'|Where-Object Name -notmatch '\.installing$'|ForEach-Object{
 [ordered]@{name=$_.Name;hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
})
$taskModelHashes=@()
if(Test-Path -LiteralPath $taskModels){
 $taskModelHashes=@(Get-ChildItem -LiteralPath $taskModels -File|ForEach-Object{
  [ordered]@{name=$_.Name;hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash;bytes=$_.Length}
 })
}
$taskActive=@(Get-ChildItem -LiteralPath $taskMods -File -Filter 'vabpnn*.jar'|Where-Object Name -notlike 'vabpnn-telemetry-*')
if($taskActive.Count -ne 1 -or $taskActive[0].Name -ne 'vabpnn-compare-1.0.3.jar'){throw 'Unexpected active model package; original retained'}
$taskOld=$taskActive[0].FullName;$taskDisabled=[IO.Path]::GetFullPath($taskOld+'.disabled')
if(!$taskDisabled.StartsWith($taskMods+'\',[StringComparison]::OrdinalIgnoreCase)|| (Test-Path -LiteralPath $taskDisabled)){throw 'Old disabled path unsafe or already exists'}
Copy-Item -LiteralPath $taskSource -Destination $taskPending
if((Get-FileHash -LiteralPath $taskPending -Algorithm SHA256).Hash -ne $taskHash){throw 'Staged JAR differs'}
Move-Item -LiteralPath $taskOld -Destination $taskDisabled
try{Move-Item -LiteralPath $taskPending -Destination $taskTarget}catch{Move-Item -LiteralPath $taskDisabled -Destination $taskOld;throw}
foreach($taskEntry in $taskHistory){
 $taskCheck=Join-Path $taskMods $taskEntry.name
 if($taskEntry.name -eq $taskActive[0].Name){$taskCheck=$taskDisabled}
 if((Get-FileHash -LiteralPath $taskCheck -Algorithm SHA256).Hash -ne $taskEntry.hash){throw ('History package changed: '+$taskEntry.name)}
}
foreach($taskEntry in $taskModelHashes){
 if((Get-FileHash -LiteralPath (Join-Path $taskModels $taskEntry.name) -Algorithm SHA256).Hash -ne $taskEntry.hash){throw ('User model changed: '+$taskEntry.name)}
}
if((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash -ne $taskHash){throw 'Installed JAR differs'}
$taskProof=[ordered]@{schema=1;version='compare-1.0.4';installed_utc=[DateTimeOffset]::UtcNow.ToString('o');target=$taskTarget;sha256=$taskHash;prior_source_count=$taskPreserved.Count;prior_jar_count=$taskHistory.Count;user_model_count=$taskModelHashes.Count;unchanged_user_models=$taskModelHashes;history=$taskHistory;old_disabled=$taskDisabled}
$taskProof|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $taskProject 'Mod\build\mouse-deployment.json')
Write-Output ("Installed "+$taskTarget)
Write-Output ("Preserved "+$taskPreserved.Count+" source files, "+$taskHistory.Count+" old packages and "+$taskModelHashes.Count+" user model files; SHA256 "+$taskHash)
