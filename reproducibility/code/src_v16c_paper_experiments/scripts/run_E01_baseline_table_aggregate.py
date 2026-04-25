#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E01 baseline table aggregation from provided PyG baseline CSV.
This script does not train anything; it aggregates precomputed fair baseline rows.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-csv', default='baselines/pyg_baselines_fair_best_by_val_partial(1).csv')
    p.add_argument('--out-dir', default='results/E01_baseline_table')
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.baseline_csv)
    rows = df.groupby(['dataset','method']).agg(n=('test_acc','size'), val_mean=('val_acc','mean'), val_std=('val_acc','std'), test_mean=('test_acc','mean'), test_std=('test_acc','std'), time_mean=('time_sec','mean'), params_mean=('params','mean')).reset_index()
    rows.to_csv(out/'E01_baseline_summary_by_method.csv', index=False, encoding='utf-8-sig')
    best = rows.sort_values(['dataset','test_mean'], ascending=[True,False]).groupby('dataset').head(1)
    best.to_csv(out/'E01_baseline_best_per_dataset.csv', index=False, encoding='utf-8-sig')
    print(f'Saved E01 baseline aggregation to {out}')

if __name__ == '__main__':
    main()
