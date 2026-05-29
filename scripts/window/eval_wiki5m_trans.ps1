. (Join-Path $PSScriptRoot '_common.ps1')

$modelPath = 'bert'
$task = 'wiki5m_trans'
$repoRoot = Get-RepoRoot
Write-Host "working directory: $repoRoot"

if (-not $env:DATA_DIR) {
  $env:DATA_DIR = Join-Path $repoRoot "data/$task"
}

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
  } elseif ($task -eq 'wiki5m_trans') {
    $task = $token
  } elseif (-not $testPath) {
    $testPath = $token
  }
  $remainingArgs.RemoveAt(0)
}

$testPath = Join-Path $env:DATA_DIR 'test.txt.json'
$neighborWeight = 0.05

$pythonArgs = @(
  '-u', 'eval_wiki5m_trans.py',
  '--task', $task,
  '--is-test',
  '--eval-model-path', $modelPath,
  '--neighbor-weight', $neighborWeight,
  '--train-path', (Join-Path $env:DATA_DIR 'train.txt.json'),
  '--valid-path', $testPath
)
$pythonArgs += @($remainingArgs)

& python @pythonArgs
