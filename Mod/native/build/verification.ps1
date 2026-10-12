if($TestOperator -or $Benchmark) {
    # The numerical programs and benchmark share one unchanged sm_120 operator build.
    $taskTestObjects=@(Get-TaskOperatorObjects 'test' $taskTestCode)
}
if($TestOperator) {
    $taskTests=@(Get-ChildItem -LiteralPath (Join-Path $taskNativeRoot 'tests') -File -Filter '*_test.cu' | Sort-Object Name)
    if(!$taskTests.Count) {throw 'No CUDA verification programs found'}
    foreach($taskTest in $taskTests) {
        $taskTarget=Join-Path $taskBuild ($taskTest.BaseName+'.exe')
        $taskProgram=Get-TaskLinkedProgram $taskTest.FullName $taskTarget $taskTestCode $taskTestObjects @('bcrypt.lib')
        & $taskProgram
        if($LASTEXITCODE -ne 0) {throw ('CUDA verification failed: '+$taskTest.BaseName)}
    }
}
if($Benchmark) {
    $taskSource=Join-Path $taskNativeRoot 'benchmarks/frame_budget_benchmark.cu'
    $taskTarget=Join-Path $taskBuild 'frame_budget_benchmark.exe'
    $taskProgram=Get-TaskLinkedProgram $taskSource $taskTarget $taskTestCode $taskTestObjects @('bcrypt.lib')
    & $taskProgram
    if($LASTEXITCODE -ne 0) {throw 'Frame benchmark failed'}
}
