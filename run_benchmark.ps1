$tasks = @("task1","task2","task3","task4","task5")

Write-Host "Starting CodeRedEnv Baseline Benchmark..."
Write-Host "------------------------------------------"
"{0,-10} | {1,-10} | {2,-10}" -f "Task","Score","Status"
Write-Host "------------------------------------------"

foreach ($task in $tasks) {
    # Run inference
    $output = python inference.py --task $task --seed 0 --max-steps 30 2>$null

    # Extract score from [GRADE]=...
    $match = $output | Select-String "\[GRADE\]=([0-9\.]+)"
    
    if ($match) {
        $score = $match.Matches[0].Groups[1].Value

        if ([double]$score -gt 0.20) {
            $status = "PASSED"
        } else {
            $status = "FAIL"
        }
    } else {
        $score = "0.0000"
        $status = "FAILED"
    }

    "{0,-10} | {1,-10} | {2,-10}" -f $task,$score,$status
}   