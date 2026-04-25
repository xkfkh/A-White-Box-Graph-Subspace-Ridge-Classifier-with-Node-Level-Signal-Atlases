#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E07 coarse parameter scan.

Writes every candidate result. Selection is by val only; test is recorded only
for analysis after the fact.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paperexp.core import *


def parse_list(s, typ=float):
    return [typ(x) for x in str(s).split(',') if str(x).strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=None)
    p.add_argument('--datasets', nargs='+', default=['chameleon'])
    p.add_argument('--splits', type=int, default=10)
    p.add_argument('--top-k', default='2000,4000,6000,8000')
    p.add_argument('--dims', default='32,48,64,96,128')
    p.add_argument('--energy', default='0.90,0.95,0.99')
    p.add_argument('--w', default='0.2,0.3,0.4,0.5,0.6,0.7,0.8')
    p.add_argument('--alpha-sets', default='0.01|0.1|1.0;0.05|0.5|5.0;0.1|1.0|10.0')
    p.add_argument('--out-dir', default='results/E07_param_scan')
    args = p.parse_args()
    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    alpha_sets = [tuple(float(x) for x in part.split('|')) for part in args.alpha_sets.split(';')]
    grid = dict(top_k=parse_list(args.top_k, int), dim=parse_list(args.dims, int), energy=parse_list(args.energy, float), alphas=alpha_sets, w=parse_list(args.w, float))
    rows, best_rows = [], []
    for dataset in args.datasets:
        graph = load_dataset(root, dataset)
        for split in range(args.splits):
            if has_fixed_splits(graph):
                train_idx, val_idx, test_idx = load_fixed_split(graph, split)
                policy = 'fixed_geom_gcn_split'
            else:
                train_idx, val_idx, test_idx, policy = random_split_matching_protocol(graph, 2000+split, prefer_fixed_counts=False)
            trace = []
            cfg, pred, info, metrics = select_by_validation(graph, train_idx, val_idx, test_idx, grid, trace=trace)
            for t in trace:
                t.update(dict(experiment='E07', dataset=dataset, split=split, split_policy=policy))
            rows.extend(trace)
            best_rows.append(dict(experiment='E07', dataset=dataset, split=split, split_policy=policy, **asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)), **metrics))
            print(f"{dataset} split={split} best val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f} cfg={asdict(cfg)}")
    write_csv(out/'E07_all_candidates.csv', rows)
    write_csv(out/'E07_best_by_val.csv', best_rows)
    pd.DataFrame(best_rows).groupby('dataset').agg(n=('test_acc','size'), val_mean=('val_acc','mean'), val_std=('val_acc','std'), test_mean=('test_acc','mean'), test_std=('test_acc','std')).reset_index().to_csv(out/'E07_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\nSaved E07 to {out}')

if __name__ == '__main__':
    main()
