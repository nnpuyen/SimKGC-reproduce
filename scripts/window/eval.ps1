. (Join-Path $PSScriptRoot '_common.ps1')

$modelPath = 'bert'
$task = 'WN18RR'
$testPath = $null
$remainingArgs = [System.Collections.Generic.List[string]]::new()
$remainingArgs.AddRange([string[]]$args)

while ($remainingArgs.Count -ge 1) {
  $token = $remainingArgs[0]
  if ($token -eq '--model_path' -or $token -eq '--model-path') {
    $modelPath = $remainingArgs[1]
    $remainingArgs.RemoveRange(0, 2)
    continue
  }
  if ($token -eq '--task') {
    $task = $remainingArgs[1]
    $remainingArgs.RemoveRange(0, 2)
    continue
  }
  if ($token -eq '--test_path' -or $token -eq '--test-path') {
    $testPath = $remainingArgs[1]
    $remainingArgs.RemoveRange(0, 2)
    continue
  }
  if ($token -eq '--') {
    $remainingArgs.RemoveAt(0)
    break
  }
  if ($token.StartsWith('--')) {
    break
  }
  if ($modelPath -eq 'bert') {
    $modelPath = $token
  } elseif ($task -eq 'WN18RR') {
    $task = $token
  } elseif (-not $testPath) {
    $testPath = $token
  }
  $remainingArgs.RemoveAt(0)
}

$repoRoot = Get-RepoRoot
Write-Host "working directory: $repoRoot"

if (-not $env:DATA_DIR) {
  $env:DATA_DIR = Join-Path $repoRoot "data/$task"
}

$testPath = Join-Path $env:DATA_DIR 'test.txt.json'
if (-not (Test-Path $testPath)) {
  $testPath = Join-Path $env:DATA_DIR 'test.txt'
}

Write-Host "eval.sh config: task=$task model_path=$modelPath DATA_DIR=$env:DATA_DIR test_path=$testPath train_path=$($env:DATA_DIR)/train.txt.json"

$pythonArgs = @(
  '-u', 'evaluate.py',
  '--task', $task,
  '--eval-model-path', $modelPath,
  '--train-path', (Join-Path $env:DATA_DIR 'train.txt.json'),
  '--valid-path', $testPath
)
$pythonArgs += @($remainingArgs)

& python @pythonArgs
