How to run the full pipeline (Windows PowerShell)
===============================================

This document explains how to run the full pipeline on Windows using the PowerShell helper scripts in `scripts/window`.

**Prerequisites**
- Python 3.8+ installed and on PATH.
- Git and network access to download datasets (if needed).
- (Optional) GPU + CUDA if you plan to use CUDA builds of PyTorch.

**1. Create and activate a virtual environment**

Open PowerShell in the repository root and run:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need a specific PyTorch build (CUDA or CPU), install it after activating the venv following the instructions at https://pytorch.org/.

**2. (Optional) Download dataset(s)**

For Wikidata5M there is a downloader script. Example:

```powershell
.\scripts\window\download_wikidata5m.ps1
```

Many datasets in `data/` are already included (e.g., WN18RR, FB15k237). If you add or change dataset locations, see the `DATA_DIR` environment variable notes below.

**3. Preprocess data**

Use the PowerShell preprocess script to convert raw files into JSON inputs for training and evaluation. The script defaults to `WN18RR` but accepts a task name argument.

Default (WN18RR):

```powershell
.\scripts\window\preprocess.ps1
```

Other tasks (example `FB15k237`):

```powershell
.\scripts\window\preprocess.ps1 FB15k237
```

This runs `preprocess.py` and writes JSON files under `data/<TASK>/` (e.g., `train.txt.json`, `valid.txt.json`, `test.txt.json`).

**4. Train**

Choose the appropriate training script under `scripts/window` for your dataset. Example scripts:

- [scripts/window/train_wn_infonce_original.ps1](scripts/window/train_wn_infonce_original.ps1) — WordNet (WN18RR)
- [scripts/window/train_wiki.ps1](scripts/window/train_wiki.ps1) — Wikidata / Wiki dataset
- [scripts/window/train_fb.ps1](scripts/window/train_fb.ps1) — FB15k237

Basic example (train WN18RR):

```powershell
$env:OUTPUT_DIR = "$(Join-Path (Get-Location) 'checkpoint/WN18RR_experiment')"
.\scripts\window\train_wn.ps1
```

To pass extra training options (they are forwarded to `main.py`), append them to the script call. Example:

```powershell
.\scripts\window\train_wn.ps1 --epochs 10 --lr 1e-4
```

Notes:
- The train scripts use `DATA_DIR` and `OUTPUT_DIR` environment variables if set; otherwise they use the default `data/<TASK>` and `checkpoint/<TASK>_<timestamp>` locations.
- Trained models and checkpoints are written under the `checkpoint/` folder.

**5. Evaluate**

Use the evaluation helper to run evaluation on a trained model.

Basic usage (model path then task name):

```powershell
.\scripts\window\eval.ps1 <model_path> <task>
```

Example:

```powershell
.\scripts\window\eval.ps1 checkpoint\WN18RR_2026-05-30-0056.19 WN18RR
```

You can also pass flags:

```powershell
.\scripts\window\eval.ps1 --model-path checkpoint\WN18RR_2026-05-30-0056.19 --task WN18RR
```

The script runs `evaluate.py` and writes/prints metrics; predictions are stored under `predictions/<TASK>/`.

**Environment variables**
- `DATA_DIR` — override the default data folder for a task. Example:

```powershell
$env:DATA_DIR = "D:\\KL\\SimKGC\\data\\WN18RR"
```

- `OUTPUT_DIR` — override where checkpoints are saved by training scripts.

**Helpful scripts**
- See `scripts/window/_common.ps1` for helper functions used by the Windows scripts.
- See `scripts/window/preprocess.ps1`, `scripts/window/train_wn.ps1`, and `scripts/window/eval.ps1` for exact argument forwarding and defaults.

**Troubleshooting**
- If `python` is not found after activating the venv, ensure the venv activation succeeded and check `Get-Command python`.
- If GPU is not used, confirm your installed PyTorch build has CUDA support.
- For missing data files, ensure you ran `preprocess.ps1` for the task or set `DATA_DIR` to the correct location.

**References**
- Scripts folder: [scripts/window](scripts/window)

---
Created to document running the full pipeline on Windows using the repository helper scripts.
