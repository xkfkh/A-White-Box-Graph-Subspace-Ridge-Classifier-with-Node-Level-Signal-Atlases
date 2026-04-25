AutoDL CPU fair strongest-baseline rerun for SRC-v16c
====================================================

Files:
- run_strong_baselines_srcv16c_fair.py: main runner; only GraphSAGE/LINKX strongest methods.
- make_strong_baseline_jobs.sh: creates 60 CPU jobs: 6 datasets x 10 repeats, no squirrel.
- summarize_strong_baselines_srcv16c.py: merges results and computes mean/std.

Important:
Your Windows path D:\桌面\MSR实验复现与创新\planetoid\data cannot be read directly by AutoDL Linux.
Upload/copy that data folder to a Linux path, recommended:
  /root/autodl-tmp/planetoid/data
Then edit DATA_ROOT in make_strong_baseline_jobs.sh if your cloud path is different.

Fast default protocol:
- one config: lr=0.01, weight_decay=5e-4, hidden=64, dropout=0.5
- epochs=300, patience=50
- early stopping uses validation accuracy only
- test accuracy is computed only once after best validation epoch is selected

Run:
  cd /root/autodl-tmp/srcv16c_baseline
  bash make_strong_baseline_jobs.sh
  cat results_strong_baseline_cpu/jobs.txt | xargs -I{} -P 4 bash -lc '{}'
  python summarize_strong_baselines_srcv16c.py --out_dir results_strong_baseline_cpu

If CPU has many cores, use -P 6 or -P 8. If memory is tight, use -P 2.
