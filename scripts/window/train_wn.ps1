. (Join-Path $PSScriptRoot '_common.ps1')

$repoRoot = Get-RepoRoot
Write-Host "working directory: $repoRoot"

$task = 'WN18RR'
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
  '--pretrained-model', 'distilbert-base-uncased',
  '--pooling', 'mean',
  '--lr', '5e-5',
  '--use-link-graph',
  '--train-path', (Join-Path $dataDir 'train.txt.json'),
  '--valid-path', (Join-Path $dataDir 'valid.txt.json'),
  '--valid-label-path', (Join-Path $dataDir 'valid_w_label.txt'),
  '--task', $task,
  '--batch-size', '512',
  '--print-freq', '20',
  '--additive-margin', '0.02',
  '--use-amp',
  '--use-self-negative',
  '--pre-batch', '0',
  '--finetune-t',
  '--loss-type', 'alignment',
  '--use-uniformity-loss',
  '--directau-uniformity-scale', '4',
  '--epochs', '50',
  '--workers', '2',
  '--max-to-keep', '3'
)
$pythonArgs += @($args)

& python @pythonArgs
