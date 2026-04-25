#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E03 random split stability: src_v16c vs baseline CSV aggregation.

No leakage: every random split selects src_v16c hyper-parameters by validation only.
Baseline CSV is only aggregated; it is not used to tune src_v16c.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paperexp.core import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=None)
    p.add_argument('--datasets', nargs='+', default=['chameleon','squirrel','cornell','texas','wisconsin','amazon-photo','amazon-computers'])
    p.add_argument('--exclude', nargs='*', default=['actor'])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed0', type=int, default=20260419)
    p.add_argument('--grid', choices=['default','fast'], default='default')
    p.add_argument('--baseline-csv', default='pyg_baselines_fair_best_by_val_partial(1).csv')
    p.add_argument('--out-dir', default='results/E03_random_split_stability')
    args = p.parse_args()

    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    grid = grid_from_name(args.grid)
    rows, traces = [], []
    for dataset in args.datasets:
        if dataset in set(args.exclude):
            continue
        print(f'\n=== E03 {dataset} ===')
        graph = load_dataset(root, dataset)
        for rep in range(args.repeats):
            seed = args.seed0 + rep
            train_idx, val_idx, test_idx, policy = random_split_matching_protocol(graph, seed, prefer_fixed_counts=True)
            trace = []
            cfg, pred, info, metrics = select_by_validation(graph, train_idx, val_idx, test_idx, grid, trace=trace)
            row = dict(experiment='E03', dataset=dataset, repeat=rep, seed=seed, split_policy=policy,
                       num_nodes=graph.num_nodes, num_features=graph.num_features, num_classes=graph.num_classes,
                       train_size=len(train_idx), val_size=len(val_idx), test_size=len(test_idx),
                       method='src_v16c', **asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)),
                       **metrics, selection_rule='grid_selected_by_val_only_no_test')
            rows.append(row)
            for t in trace:
                t.update(dict(dataset=dataset, repeat=rep, seed=seed, split_policy=policy))
            traces.extend(trace)
            print(f"rep={rep} val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f} cfg={asdict(cfg)}")

    write_csv(out/'E03_src_v16c_random_split_rows.csv', rows)
    write_csv(out/'E03_src_v16c_candidate_trace.csv', traces)
    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby(['dataset','method']).agg(
            n=('test_acc','size'), val_mean=('val_acc','mean'), val_std=('val_acc','std'),
            test_mean=('test_acc','mean'), test_std=('test_acc','std')
        ).reset_index()
        summary.to_csv(out/'E03_src_v16c_summary.csv', index=False, encoding='utf-8-sig')

    # Aggregate provided PyG baseline CSV if present.
    bpath = Path(args.baseline_csv)
    if not bpath.exists():
        bpath = Path(__file__).resolve().parents[1] / 'baselines' / args.baseline_csv
    if bpath.exists():
        b = pd.read_csv(bpath)
        b = b[b['dataset'].isin(args.datasets)]
        bsum = b.groupby(['dataset','method']).agg(
            n=('test_acc','size'), val_mean=('val_acc','mean'), val_std=('val_acc','std'),
            test_mean=('test_acc','mean'), test_std=('test_acc','std'), time_mean=('time_sec','mean')
        ).reset_index()
        bsum.to_csv(out/'E03_pyg_baseline_summary_from_csv.csv', index=False, encoding='utf-8-sig')
        allsum = pd.concat([summary if rows else pd.DataFrame(), bsum], ignore_index=True)
        allsum.to_csv(out/'E03_combined_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\nSaved E03 to {out}')

if __name__ == '__main__':
    main()
