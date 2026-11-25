#!/usr/bin/env bash
# Reproduce every result in REPORT.md (CPU: ~2-3 h on 4 cores; much faster on a GPU).
#   bash scripts/run_all_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=data/processed/affec_T50.npz
COMMON="--epochs 30 --patience 8 --threads 2"

# 0. synthetic sanity check (slides, objective 4)
[ -f outputs/synthetic_random_all/metrics.json ] || python src/train.py --data synthetic --samples 2000 --split random --out outputs/synthetic_random_all $COMMON

# stream 1: modality ablation, subject-independent, 3-fold
python scripts/run_ablation.py --data $DATA --labels stimulus4 --split subject --folds 3 $COMMON \
    > outputs/ablation.log 2>&1 &

# stream 2: main results
(
python src/train.py --data $DATA --labels stimulus4 --split subject --folds 5 --out outputs/main_stimulus4_subject $COMMON
python src/train.py --data $DATA --labels stimulus4 --split random  --folds 5 --out outputs/main_stimulus4_random  $COMMON
python src/train.py --data $DATA --labels stimulus6 --split subject --folds 3 --out outputs/stimulus6_subject $COMMON
python src/train.py --data $DATA --labels felt_va4  --split subject --folds 3 --out outputs/felt_va4_subject  $COMMON
) > outputs/main.log 2>&1 &
wait
echo "all experiments finished"
