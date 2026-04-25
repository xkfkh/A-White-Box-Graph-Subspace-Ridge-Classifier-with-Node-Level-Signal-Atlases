#!/usr/bin/env bash
set -euo pipefail

# 修改这里：必须是 AutoDL Linux 里面的数据目录，不是 Windows 的 D:\ 路径。
DATA_ROOT="/root/autodl-tmp/planetoid/data"
OUT_DIR="results_strong_baseline_cpu"
PY="python"

mkdir -p "$OUT_DIR/logs"
: > "$OUT_DIR/jobs.txt"

# squirrel 不写进 jobs；actor 也不写，因为最终表没有 actor。
for d in amazon-computers amazon-photo chameleon cornell texas wisconsin; do
  for r in 0 1 2 3 4 5 6 7 8 9; do
    echo "$PY run_strong_baselines_srcv16c_fair.py --data_root '$DATA_ROOT' --dataset $d --repeats $r --out_dir '$OUT_DIR' --device cpu --threads 2 --epochs 300 --patience 50 --eval_every 5 --feature_norm row_l1 > '$OUT_DIR/logs/${d}_r${r}.log' 2>&1" >> "$OUT_DIR/jobs.txt"
  done
done

echo "Wrote $OUT_DIR/jobs.txt"
echo "Run with, for example:"
echo "  cat $OUT_DIR/jobs.txt | xargs -I{} -P 4 bash -lc '{}'"
