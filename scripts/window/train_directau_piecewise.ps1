# PowerShell script to run DirectAU piecewise schedule on Windows
param(
    [string]$DataDir = "data/WN18RR",
    [string]$ModelDir = "checkpoint\WN18RR_directau_piecewise",
    [string]$Pretrained = "distilbert-base-uncased"
)

if (-not (Test-Path $ModelDir)) { New-Item -ItemType Directory -Path $ModelDir | Out-Null }

python main.py `
  --model-dir $ModelDir `
  --pretrained-model $Pretrained `
  --pooling mean `
  --lr 5e-5 `
  --use-link-graph `
  --train-path "$DataDir/train.txt.json" `
  --valid-path "$DataDir/valid.txt.json" `
  --valid-label-path "$DataDir/valid_w_label.txt" `
  --task WN18RR `
  --batch-size 512 `
  --print-freq 20 `
  --additive-margin 0.02 `
  --use-amp `
  --pre-batch 0 `
  --finetune-t `
  --loss-type alignment `
  --use-uniformity-loss `
  --directau-piecewise `
  --directau-piecewise-phases 15,15,20 `
  --directau-piecewise-values 4.5,5.5,6.5 `
  --directau-alpha 1 `
  --directau-gamma 1 `
  --directau-eps 1e-12 `
  --directau-uniformity-scale 4 `
  --no-negative-sampling `
  --epochs 50 `
  --workers 2 `
  --max-to-keep 3
