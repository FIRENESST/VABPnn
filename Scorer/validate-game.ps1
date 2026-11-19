$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
$taskService=$null
try {
 $taskPort=19876
 $taskProbe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$taskPort)
 try { $taskProbe.Start() } finally { $taskProbe.Stop() }
 $taskData='build/game-test-scorer-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff')
 $taskService=Start-Process -FilePath 'C:/Java/JDK26/bin/java.exe' -ArgumentList @('-jar','build/libs/vabpnn-scorer-0.5.0.jar','--headless','--port',"$taskPort",'--data',$taskData) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'build/test-scorer-stdout.log' -RedirectStandardError 'build/test-scorer-stderr.log'
 $taskHandler=[Net.Http.HttpClientHandler]::new()
 $taskHandler.UseProxy=$false
 $taskHttp=[Net.Http.HttpClient]::new($taskHandler)
 $taskHttp.Timeout=[TimeSpan]::FromSeconds(2)
 $taskReady=$false
 for($taskAttempt=0;$taskAttempt -lt 20;$taskAttempt++){
  if($taskService.HasExited){throw 'Test Scorer exited before becoming ready'}
  try { $taskHealth=$taskHttp.GetStringAsync("http://127.0.0.1:$taskPort/health").GetAwaiter().GetResult() | ConvertFrom-Json; if($taskHealth.service -eq 'vabpnn-scorer'){$taskReady=$true;break} }catch{}
  Start-Sleep -Milliseconds 200
 }
 if(!$taskReady){throw 'Test Scorer not ready'}
 $env:JAVA_HOME='C:/Java/JDK26'
 & ./gradlew.bat --offline --console=plain runClientGameTest 2>&1 | Tee-Object -FilePath game-validation.log
 $taskResult=$LASTEXITCODE
 $taskHttp.GetStringAsync("http://127.0.0.1:$taskPort/v1/status").GetAwaiter().GetResult() | Set-Content -LiteralPath build/game-test-scorer-status.json
 if($taskResult -eq 0){
  $taskRewards=@(Get-Content -LiteralPath (Join-Path $taskData 'rewards.jsonl') | ForEach-Object {$_ | ConvertFrom-Json} | Where-Object {$_.valid})
  $taskCrafts=@($taskRewards | ForEach-Object {$_.crafting_evaluations})
  $taskPickups=@($taskRewards | ForEach-Object {$_.pickup_evaluations})
  foreach($taskExpected in @(
   @{Item='minecraft:wooden_axe';Kind='first_craft';Amount=1.0},
   @{Item='minecraft:wooden_axe';Kind='replacement_used';Amount=0.5},
   @{Item='minecraft:stone_axe';Kind='upgrade';Amount=1.5},
   @{Item='minecraft:iron_pickaxe';Kind='first_craft';Amount=1.0}
  )){
   if(!($taskCrafts | Where-Object {$_.item -eq $taskExpected.Item -and $_.kind -eq $taskExpected.Kind -and $_.raw_amount -eq $taskExpected.Amount})){throw "Missing actual Scorer reward: $($taskExpected.Item) $($taskExpected.Kind)"}
  }
  if(!($taskPickups | Where-Object {$_.item -eq 'minecraft:oak_log' -and $_.eligible_count -gt 0})){throw 'Harvested pickup did not earn an actual Scorer reward'}
  if(!($taskPickups | Where-Object {$_.item -eq 'minecraft:oak_log' -and $_.picked_count -gt 0 -and $_.eligible_count -eq 0})){throw 'Untracked redrop exclusion missing from actual Scorer rewards'}
  Write-Output 'PASS: real Scorer reward log confirms first crafts, upgrade, replacement after use, eligible pickup, and redrop exclusion'
 }
 exit $taskResult
} finally {
 if($null -ne $taskService -and !$taskService.HasExited){Stop-Process -Id $taskService.Id -ErrorAction SilentlyContinue}
 if($null -ne $taskHttp){$taskHttp.Dispose()}
 Pop-Location
}
