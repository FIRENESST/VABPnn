@echo off
setlocal
set "SCORER_JAVA=C:\Java\JDK26\bin\javaw.exe"
if not exist "%SCORER_JAVA%" set "SCORER_JAVA=javaw.exe"
start "VABPnn Scorer" "%SCORER_JAVA%" -jar "%~dp0build\libs\vabpnn-scorer-0.5.0.jar" --data "%~dp0data"
