. (Join-Path $PSScriptRoot '_common.ps1')

$task = 'WN18RR'
if ($args.Count -ge 1 -and -not $args[0].StartsWith('--')) {
  $task = $args[0]
}

$repoRoot = Get-RepoRoot
$trainPath = Join-Path $repoRoot "data/$task/train.txt"
$validPath = Get-ExistingFile @(
  (Join-Path $repoRoot "data/$task/valid.txt"),
  (Join-Path $repoRoot "data/$task/valid_w_label.txt")
)
$testPath = Get-ExistingFile @(
  (Join-Path $repoRoot "data/$task/test.txt"),
  (Join-Path $repoRoot "data/$task/test_w_label.txt")
)

if (-not $validPath) {
  throw "No valid file found for task $task"
}
if (-not $testPath) {
  throw "No test file found for task $task"
}

$pythonArgs = @(
  '-u', 'preprocess.py',
  '--task', $task,
  '--train-path', $trainPath,
  '--valid-path', $validPath,
  '--test-path', $testPath
)

& python @pythonArgs
