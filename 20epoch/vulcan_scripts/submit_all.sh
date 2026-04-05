#!/bin/bash
set -e

PYTHON="/scratch/zchai3/ece740_final__runtime/env__vlm_base/bin/python3"
SCRIPTS="/scratch/zchai3/ece740_placement/scripts"
DATA_ROOT="/scratch/zchai3/ece740_placement/bootplace_like_data"
CS_ROOT="/scratch/zchai3/ece740_final__runtime/datasets__raw/cityscapes/data"
MODELS="/scratch/zchai3/ece740_placement/models"
LOGS="/scratch/zchai3/ece740_placement/logs"
PARTITION="gpubase_bygpu_b1"

mkdir -p $LOGS $MODELS $DATA_ROOT

echo "=== Phase 1: Data preprocessing (~25 min) ==="
DATA_JID=$(sbatch --parsable \
  --job-name="build_data" \
  --partition=$PARTITION \
  --gres=gpu:1 \
  --cpus-per-task=32 \
  --mem=32G \
  --time=1:00:00 \
  --output="$LOGS/build_data_%j.log" \
  --error="$LOGS/build_data_%j.err" \
  --wrap="$PYTHON $SCRIPTS/build_dataset.py \
    --cityscapes_root $CS_ROOT \
    --out_root $DATA_ROOT")
echo "  build_data -> job $DATA_JID"

echo "=== Phase 2: Training (3 models parallel, ~10 min) ==="
TRAIN_CAND_JID=$(sbatch --parsable \
  --dependency=afterok:$DATA_JID \
  --job-name="train_cand" \
  --partition=$PARTITION \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=1:00:00 \
  --output="$LOGS/train_cand_%j.log" \
  --error="$LOGS/train_cand_%j.err" \
  --wrap="$PYTHON $SCRIPTS/train_candidate.py \
    --data_root $DATA_ROOT \
    --cityscapes_root $CS_ROOT \
    --save_dir $MODELS")
echo "  candidate -> job $TRAIN_CAND_JID"

TRAIN_SS_JID=$(sbatch --parsable \
  --dependency=afterok:$DATA_JID \
  --job-name="train_ss" \
  --partition=$PARTITION \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=1:00:00 \
  --output="$LOGS/train_ss_%j.log" \
  --error="$LOGS/train_ss_%j.err" \
  --wrap="$PYTHON $SCRIPTS/train_supportsurface.py \
    --data_root $DATA_ROOT \
    --cityscapes_root $CS_ROOT \
    --save_dir $MODELS")
echo "  supportsurface -> job $TRAIN_SS_JID"

TRAIN_HC_JID=$(sbatch --parsable \
  --dependency=afterok:$DATA_JID \
  --job-name="train_hc" \
  --partition=$PARTITION \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=1:00:00 \
  --output="$LOGS/train_hc_%j.log" \
  --error="$LOGS/train_hc_%j.err" \
  --wrap="$PYTHON $SCRIPTS/train_hardconstraint.py \
    --data_root $DATA_ROOT \
    --cityscapes_root $CS_ROOT \
    --save_dir $MODELS")
echo "  hardconstraint -> job $TRAIN_HC_JID"

echo "=== Phase 3: Extract metrics ==="
METRICS_JID=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_CAND_JID}:${TRAIN_SS_JID}:${TRAIN_HC_JID}" \
  --job-name="metrics" \
  --partition=$PARTITION \
  --gres=gpu:1 \
  --mem=4G \
  --time=0:05:00 \
  --output="$LOGS/metrics_%j.log" \
  --error="$LOGS/metrics_%j.err" \
  --wrap="$PYTHON $SCRIPTS/extract_metrics.py \
    --models_dir $MODELS \
    --output $MODELS/metrics_all.json")
echo "  metrics -> job $METRICS_JID"

echo ""
echo "Pipeline submitted: 5 jobs total"
echo "  Phase 1: build_data ($DATA_JID)"
echo "  Phase 2: train_cand ($TRAIN_CAND_JID), train_ss ($TRAIN_SS_JID), train_hc ($TRAIN_HC_JID)"
echo "  Phase 3: metrics ($METRICS_JID)"
echo ""
echo "Monitor:  squeue -u zchai3"
echo "Logs:     ls $LOGS/"
echo "Results:  $MODELS/metrics_all.json"
