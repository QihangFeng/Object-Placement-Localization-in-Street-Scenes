# 20-Epoch Training Experiment

This folder contains the complete code and results for training all three models for 20 epochs.

## Directory Structure

```
20epoch/
├── notebooks/                        # Jupyter notebooks
│   ├── build_dataset.ipynb           # Data preprocessing pipeline
│   ├── placement_localization_pipeline.ipynb  # Core training & evaluation
│   └─�� results.ipynb                 # Results analysis & visualization
├── vulcan_scripts/                   # HPC training scripts (Vulcan cluster)
│   ├── build_dataset.py              # Data preprocessing
│   ├── train_candidate.py            # Stage 1: Candidate-only baseline
│   ├── train_supportsurface.py       # Stage 2: Support surface model
│   ├── train_hardconstraint.py       # Stage 3: Hard constraint model
│   ├── extract_metrics.py            # Extract metrics from checkpoints
│   └── submit_all.sh                 # SLURM job submission orchestration
├── results/
│   ├── metrics_all.json              # Full 20-epoch training metrics (all 3 models)
│   └── figures/                      # Training curves and comparison tables
└── requirements.txt                  # Python dependencies
```

## Key Results (20 Epochs)

| Model | Val Loss | Top-1 IoU | Top-5 IoU |
|-------|----------|-----------|-----------|
| Candidate | 6.321 | 0.0483 | 0.0749 |
| SupportSurface | 6.065 | 0.0768 | 0.1137 |
| HardConstraint | 5.935 | 0.0669 | 0.0997 |

**Best Val Top-1 IoU across all 20 epochs:** SupportSurface = **0.0808**

## How to Reproduce

### Prerequisites

1. **Python environment:**
   ```bash
   pip install -r requirements.txt
   ```
   For PyTorch with CUDA, follow: https://pytorch.org/get-started/locally/

2. **Cityscapes dataset:**
   - Register and download from https://www.cityscapes-dataset.com/
   - Required: `leftImg8bit` (train/val) and `gtFine` (train/val)

### Three-Stage Training Pipeline

```bash
# Stage 0: Build dataset
python vulcan_scripts/build_dataset.py --cityscapes_root <CITYSCAPES_PATH> --out_root ./bootplace_like_data

# Stage 1: Train candidate model
python vulcan_scripts/train_candidate.py --data_root ./bootplace_like_data --cityscapes_root <CITYSCAPES_PATH> --save_dir ./models --num_epochs 20

# Stage 2: Train support surface model
python vulcan_scripts/train_supportsurface.py --data_root ./bootplace_like_data --cityscapes_root <CITYSCAPES_PATH> --save_dir ./models --num_epochs 20

# Stage 3: Train hard constraint model
python vulcan_scripts/train_hardconstraint.py --data_root ./bootplace_like_data --cityscapes_root <CITYSCAPES_PATH> --save_dir ./models --num_epochs 20

# Extract metrics
python vulcan_scripts/extract_metrics.py --models_dir ./models --output ./results/metrics_all.json
```

### On Vulcan HPC (SLURM)

Edit paths in `vulcan_scripts/submit_all.sh` to match your environment, then:
```bash
bash vulcan_scripts/submit_all.sh
```

## Model Checkpoints

Trained model checkpoints are stored in `../models/`:
- `01_candidate_last.pt` / `01_candidate_best.pt`
- `02_supportsurface_last.pt` / `02_supportsurface_best.pt`
- `03_ss_hard_last.pt` / `03_ss_hard_best.pt`
