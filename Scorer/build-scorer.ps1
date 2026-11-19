$ErrorActionPreference='Stop'
$env:JAVA_HOME='C:\Java\JDK26'
Push-Location $PSScriptRoot
try { & .\gradlew.bat --offline --console=plain build; if($LASTEXITCODE -ne 0){throw "Scorer build failed: $LASTEXITCODE"} }
finally { Pop-Location }

