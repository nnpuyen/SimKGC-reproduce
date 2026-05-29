. (Join-Path $PSScriptRoot '_common.ps1')

$repoRoot = Get-RepoRoot
Write-Host "working directory: $repoRoot"

$resumePath = if ($args.Count -ge 1) { $args[0] } else { $null }
$totalEpochs = if ($args.Count -ge 2) { $args[1] } else { $null }
$extraArgs = @($args | Select-Object -Skip 2)
$outputDir = $null

if ($extraArgs.Count -gt 0 -and -not $extraArgs[0].StartsWith('--')) {
  $outputDir = $extraArgs[0]
  $extraArgs = @($extraArgs | Select-Object -Skip 1)
}

if (-not $resumePath -or -not $totalEpochs) {
  Write-Host 'Usage: resume_wn_alignment_uniformity.ps1 <resume_path> <total_epochs> [output_dir]'
  exit 1
}

if (-not $outputDir) {
  if (Test-Path $resumePath -PathType Container) {
    $outputDir = $resumePath
  } else {
    $outputDir = Split-Path -Parent $resumePath
  }
}

$task = 'WN18RR'
if (-not $env:DATA_DIR) {
  $env:DATA_DIR = Join-Path $repoRoot "data/$task"
}

$env:RESUME_PATH = $resumePath
@'
import os
import sys
import torch

ckt_path = os.environ.get('RESUME_PATH', '')
if not ckt_path:
    print('Resume path is empty; cannot read epoch')
    sys.exit(0)

try:
    ckt = torch.load(ckt_path, map_location='cpu')
    print('Checkpoint epoch =', ckt.get('epoch'))
except Exception as exc:
    print('Failed to read checkpoint epoch:', exc)
'@ | python -

$pythonArgs = @(
  '-u', 'main.py',
  '--model-dir', $outputDir,
  '--resume',
  '--resume-path', $resumePath,
  '--pretrained-model', 'distilbert-base-uncased',
  '--pooling', 'mean',
  '--lr', '5e-5',
  '--use-link-graph',
  '--train-path', (Join-Path $env:DATA_DIR 'train.txt.json'),
  '--valid-path', (Join-Path $env:DATA_DIR 'valid.txt.json'),
  '--valid-label-path', (Join-Path $env:DATA_DIR 'valid_w_label.txt'),
  '--task', $task,
  '--batch-size', '512',
  '--print-freq', '20',
  '--additive-margin', '0.02',
  '--use-amp',
  '--pre-batch', '0',
  '--finetune-t',
  '--loss-type', 'alignment',
  '--use-uniformity-loss',
  '--directau-alpha', '1.0',
  '--directau-gamma', '1.0',
  '--directau-eps', '1e-12',
  '--directau-uniformity-scale', '5',
  '--no-negative-sampling',
  '--epochs', $totalEpochs,
  '--workers', '2',
  '--max-to-keep', '3'
)
$pythonArgs += $extraArgs

& python @pythonArgs
