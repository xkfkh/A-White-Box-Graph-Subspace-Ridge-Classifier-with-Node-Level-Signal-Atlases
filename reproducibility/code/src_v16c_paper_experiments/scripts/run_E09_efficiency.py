#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E09 efficiency experiment.

Measures src_v16c wall-clock and Python peak memory. Also aggregates baseline
runtime/parameter counts from provided PyG CSV if available.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paperexp.core import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=None)
    p.add_argument('--datasets', nargs='+', default=['chameleon','squirrel','cornell','texas','wisconsin','amazon-photo','amazon-computers'])
    p.add_argument('--splits', type=int, default=3, help='use fewer splits by default for efficiency table')
    p.add_argument('--grid', choices=['default','fast'], default='fast')
    p.add_argument('--baseline-csv', default='pyg_baselines_fair_best_by_val_partial(1).csv')
    p.add_argument('--out-dir', default='results/E09_efficiency')
    args = p.parse_args()
    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    grid = grid_from_name(args.grid)
    rows = []
    for dataset in args.datasets:
        graph = load_dataset(root, dataset)
        for split in range(args.splits):
            if has_fixed_splits(graph):
                train_idx, val_idx, test_idx = load_fixed_split(graph, split)
                policy = 'fixed_geom_gcn_split'
            else:
                train_idx, val_idx, test_idx, policy = random_split_matching_protocol(graph, 9000+split, prefer_fixed_counts=False)
            def run_once():
                return select_by_validation(graph, train_idx, val_idx, test_idx, grid)
            (cfg, pred, info, metrics), eff = measure_run(run_once)
            rows.append(dict(experiment='E09', dataset=dataset, split=split, method='src_v16c', split_policy=policy,
                             train_size=len(train_idx), val_size=len(val_idx), test_size=len(test_idx),
                             num_nodes=graph.num_nodes, num_features=graph.num_features, num_classes=graph.num_classes,
                             **asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)), **metrics, **eff,
                             epoch_trainable_params=0, stored_coefficients_proxy=info['stored_coefficients_proxy'],
                             requires_epoch_training=False, selection_rule='grid_selected_by_val_only_no_test'))
            print(f"{dataset} split={split} time={eff['time_sec']:.2f}s peak={eff['python_peak_memory_mb']:.1f}MB test={metrics['test_acc']:.4f}")
    write_csv(out/'E09_src_v16c_efficiency_rows.csv', rows)
    df = pd.DataFrame(rows)
    df.groupby(['dataset','method']).agg(n=('test_acc','size'), time_mean=('time_sec','mean'), time_std=('time_sec','std'), peak_mem_mean=('python_peak_memory_mb','mean'), test_mean=('test_acc','mean'), stored_coefficients_proxy_mean=('stored_coefficients_proxy','mean')).reset_index().to_csv(out/'E09_src_v16c_efficiency_summary.csv', index=False, encoding='utf-8-sig')

    bpath = Path(args.baseline_csv)
    if not bpath.exists():
        bpath = Path(__file__).resolve().parents[1] / 'baselines' / args.baseline_csv
    if bpath.exists():
        b = pd.read_csv(bpath)
        b = b[b['dataset'].isin(args.datasets)]
        bsum = b.groupby(['dataset','method']).agg(n=('test_acc','size'), time_mean=('time_sec','mean'), time_std=('time_sec','std'), params_mean=('params','mean'), val_mean=('val_acc','mean'), test_mean=('test_acc','mean')).reset_index()
        bsum['requires_epoch_training'] = True
        bsum.to_csv(out/'E09_pyg_baseline_efficiency_from_csv.csv', index=False, encoding='utf-8-sig')
    print(f'\nSaved E09 to {out}')

if __name__ == '__main__':
    main()
