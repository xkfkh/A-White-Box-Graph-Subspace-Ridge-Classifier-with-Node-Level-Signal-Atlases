#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/srcv16c/src_v16c_paper_experiments"
DATA_ROOT="/root/autodl-tmp/planetoid/data"
OUT_DIR="$ROOT/results/E05_ablation_same_splits_no_full"
SCRIPT="$ROOT/scripts/run_E05_ablation_same_splits_no_full.py"

DATASETS=(chameleon cornell texas wisconsin amazon-photo amazon-computers)
REPEATS=(0 1 2 3 4 5 6 7 8 9)

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/single_runs"
: > "$OUT_DIR/jobs.txt"

for d in "${DATASETS[@]}"; do
  for r in "${REPEATS[@]}"; do
    run_dir="$OUT_DIR/single_runs/${d}_r${r}"
    log_file="$OUT_DIR/logs/${d}_r${r}.log"
    echo "cd '$ROOT' && python '$SCRIPT' \
      --data-root '$DATA_ROOT' \
      --datasets '$d' \
      --repeats '$r' \
      --grid fast \
      --out-dir '$run_dir' \
      > '$log_file' 2>&1" >> "$OUT_DIR/jobs.txt"
  done
done

echo "Wrote: $OUT_DIR/jobs.txt"
echo "Run with:"
echo "  cd '$ROOT'"
echo "  cat '$OUT_DIR/jobs.txt' | xargs -I{} -P 4 bash -lc '{}'"
echo "For your 12-core / 90GB machine, -P 4 is safer for dense NumPy ablations; use -P 6 only if memory remains stable."
