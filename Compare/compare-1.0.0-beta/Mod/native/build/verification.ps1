if($TestOperator -or $Benchmark) {
    # Comparison tests and the full-graph benchmark share one sm_120 operator build.
    $taskTestObjects=@(Get-TaskOperatorObjects 'test' $taskTestCode)
}
if($TestOperator) {
    $taskNames=@('dense_block_transformer_test.cu','operator_test.cu','neuron_vote_test.cu','jepa_motion_test.cu')
    $taskTests=@(Get-ChildItem -LiteralPath (Join-Path $taskNativeRoot 'tests') -File -Filter '*_test.cu' | Where-Object Name -in $taskNames | Sort-Object Name)
    if(!$taskTests.Count) {throw 'No CUDA verification programs found'}
    foreach($taskTest in $taskTests) {
        $taskTarget=Join-Path $taskBuild ($taskTest.BaseName+'.exe')
        $taskProgram=Get-TaskLinkedProgram $taskTest.FullName $taskTarget $taskTestCode $taskTestObjects @('bcrypt.lib')
        & $taskProgram
        if($LASTEXITCODE -ne 0) {throw ('CUDA verification failed: '+$taskTest.BaseName)}
    }
}
if($Benchmark) {
    $taskSource=Join-Path $taskNativeRoot 'benchmarks/dense_compare_benchmark.cu'
    $taskTarget=Join-Path $taskBuild 'dense_compare_benchmark.exe'
    $taskProgram=Get-TaskLinkedProgram $taskSource $taskTarget $taskTestCode $taskTestObjects @('bcrypt.lib')
    & $taskProgram
    if($LASTEXITCODE -ne 0) {throw 'Frame benchmark failed'}
}
