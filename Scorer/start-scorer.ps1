param([switch]$Headless,[int]$Port=9876)
$ErrorActionPreference='Stop'
$taskJar=Join-Path $PSScriptRoot 'build/libs/vabpnn-scorer-0.5.0.jar'
$taskJava='C:\Java\JDK26\bin\java.exe'
if(!(Test-Path -LiteralPath $taskJava)){ $taskJava=(Get-Command java.exe -ErrorAction Stop).Source }
$taskArguments=@('-jar',$taskJar,'--data',(Join-Path $PSScriptRoot 'data'),'--port',"$Port")
if($Headless){$taskArguments+='--headless'}
& $taskJava @taskArguments

