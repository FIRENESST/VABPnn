[CmdletBinding(SupportsShouldProcess=$true)]
param([Parameter(Mandatory=$true)][ValidateSet('formal','beta','compare')][string]$Variant,
      [string]$ModsDirectory='D:\Minecraft\JavaEdition\BlockHelm\.minecraft\versions\26.1-fabric-0.19.3\mods')
$ErrorActionPreference='Stop'
$taskMods=[IO.Path]::GetFullPath($ModsDirectory)
$taskNames=@{formal='vabpnn-1.4.1.jar';beta='vabpnn-compare-1.0.0-beta.jar';compare='vabpnn-compare-1.0.0.jar'}
$taskSelected=$taskNames[$Variant]
if(!(Test-Path -LiteralPath $taskMods -PathType Container)){throw 'Mods directory missing'}
$taskActive=@(Get-ChildItem -LiteralPath $taskMods -File -Filter 'vabpnn-*.jar')
if($taskActive.Count -eq 1 -and $taskActive[0].Name -eq $taskSelected) {
    Write-Output ($taskSelected+' is already selected; restart Minecraft after switching.');return
}
if($taskActive.Count -ne 1 -or $taskActive[0].Name -notin $taskNames.Values){throw 'Unexpected active VABPnn jars; retain files'}
$taskEnable=[IO.Path]::GetFullPath((Join-Path $taskMods $taskSelected))
$taskEnableDisabled=[IO.Path]::GetFullPath($taskEnable+'.disabled')
$taskDisable=[IO.Path]::GetFullPath($taskActive[0].FullName)
$taskDisableTarget=[IO.Path]::GetFullPath($taskDisable+'.disabled')
foreach($taskPath in @($taskEnable,$taskEnableDisabled,$taskDisable,$taskDisableTarget)) {
    if(!$taskPath.StartsWith($taskMods+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Switch path escaped mods directory'}
}
if(!(Test-Path -LiteralPath $taskEnableDisabled) -or (Test-Path -LiteralPath $taskEnable) -or (Test-Path -LiteralPath $taskDisableTarget)){
    throw 'Preserve missing or existing variant files; cannot switch automatically'
}
if(!$PSCmdlet.ShouldProcess($taskMods,'Select '+$taskSelected+' and retain '+$taskActive[0].Name+' as .disabled')){return}
Move-Item -LiteralPath $taskDisable -Destination $taskDisableTarget
try{Move-Item -LiteralPath $taskEnableDisabled -Destination $taskEnable}
catch{Move-Item -LiteralPath $taskDisableTarget -Destination $taskDisable;throw}
Write-Output ('Selected '+$taskSelected+'. Restart Minecraft. All independent checkpoints are retained.')
