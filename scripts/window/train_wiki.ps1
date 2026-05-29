. (Join-Path $PSScriptRoot '_common.ps1')

$repoRoot = Get-RepoRoot
Write-Host "working directory: $repoRoot"

$task = 'wiki5m_ind'
$remainingArgs = @($args)
if ($remainingArgs.Count -ge 1 -and -not $remainingArgs[0].StartsWith('--')) {
  $task = $remainingArgs[0]
  $remainingArgs = @($remainingArgs | Select-Object -Skip 1)
}

$outputDir = $env:OUTPUT_DIR
if (-not $outputDir) {
  $outputDir = Join-Path $repoRoot ("checkpoint/{0}_{1}" -f $task, (Get-DateStamp))
}
$dataDir = $env:DATA_DIR
if (-not $dataDir) {
  $dataDir = Join-Path $repoRoot "data/$task"
}

$pythonArgs = @(
  '-u', 'main.py',
  '--model-dir', $outputDir,
  '--pretrained-model', 'bert-base-uncased',
  '--pooling', 'mean',
  '--lr', '3e-5',
  '--train-path', (Join-Path $dataDir 'train.txt.json'),
  '--valid-path', (Join-Path $dataDir 'valid.txt.json'),
  '--task', $task,
  '--batch-size', '1024',
  '--print-freq', '20',
  '--additive-margin', '0.02',
  '--use-amp',
  '--use-self-negative',
  '--finetune-t',
  '--pre-batch', '0',
  '--directau-uniformity-scale', '4',
  '--epochs', '1',
  '--workers', '3',
  '--max-to-keep', '10'
)
$pythonArgs += $remainingArgs

& python @pythonArgs
