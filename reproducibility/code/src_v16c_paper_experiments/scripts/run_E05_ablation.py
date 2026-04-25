#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E05 ablation study for src_v16c.

Ablations are selected by validation independently. Test is never used for selection.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paperexp.core import *


def ablation_grid(base_grid, ablation):
    g = {k:list(v) for k,v in base_grid.items()}
    variant = 'full'
    if ablation == 'full':
        pass
    elif ablation in ['no_p3','no_sym','no_highpass','raw_only','row_lowpass_only']:
        variant = ablation
    elif ablation == 'pca_only':
        g['w'] = [1.0]
    elif ablation == 'ridge_only':
        g['w'] = [0.0]
    elif ablation == 'single_alpha':
        g['alphas'] = [(1.0,)]
    elif ablation == 'fixed_dim64_energy095':
        g['dim'] = [64]; g['energy'] = [0.95]
    elif ablation == 'no_energy_fixed_dim':
        # energy=1.0 approximates using max dim without early truncation.
        g['energy'] = [1.0]
    elif ablation == 'small_grid_v16c_core':
        g['top_k'] = [6000]; g['dim'] = [64]; g['energy'] = [0.95]; g['alphas'] = [(0.01,0.1,1.0)]; g['w'] = [0.5]
    else:
        raise ValueError(ablation)
    return g, variant


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=None)
    p.add_argument('--datasets', nargs='+', default=['chameleon','squirrel','cornell','texas','wisconsin','amazon-photo','amazon-computers'])
    p.add_argument('--splits', type=int, default=10)
    p.add_argument('--grid', choices=['default','fast'], default='fast')
    p.add_argument('--ablations', nargs='+', default=['full','no_p3','no_sym','no_highpass','pca_only','ridge_only','single_alpha','fixed_dim64_energy095','raw_only','row_lowpass_only'])
    p.add_argument('--out-dir', default='results/E05_ablation')
    args = p.parse_args()
    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    base_grid = grid_from_name(args.grid)
    rows, traces = [], []
    for dataset in args.datasets:
        print(f'\n=== E05 {dataset} ===')
        graph = load_dataset(root, dataset)
        for split in range(args.splits):
            if has_fixed_splits(graph):
                train_idx, val_idx, test_idx = load_fixed_split(graph, split)
                policy = 'fixed_geom_gcn_split'
            else:
                train_idx, val_idx, test_idx, policy = random_split_matching_protocol(graph, 1000+split, prefer_fixed_counts=False)
            for ab in args.ablations:
                g, variant = ablation_grid(base_grid, ab)
                trace = []
                cfg, pred, info, metrics = select_by_validation(graph, train_idx, val_idx, test_idx, g, feature_variant=variant, trace=trace)
                rows.append(dict(experiment='E05', dataset=dataset, split=split, split_policy=policy,
                                 ablation=ab, method='src_v16c_ablation',
                                 train_size=len(train_idx), val_size=len(val_idx), test_size=len(test_idx),
                                 **asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)), **metrics,
                                 pca_dims_json=json.dumps(info['pca_dims'], ensure_ascii=False),
                                 selection_rule='ablation_grid_selected_by_val_only_no_test'))
                for t in trace:
                    t.update(dict(dataset=dataset, split=split, ablation=ab))
                traces.extend(trace)
                print(f"split={split} {ab:20s} val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f}")
    write_csv(out/'E05_ablation_rows.csv', rows)
    write_csv(out/'E05_ablation_candidate_trace.csv', traces)
    df = pd.DataFrame(rows)
    df.groupby(['dataset','ablation']).agg(n=('test_acc','size'), val_mean=('val_acc','mean'), val_std=('val_acc','std'), test_mean=('test_acc','mean'), test_std=('test_acc','std')).reset_index().to_csv(out/'E05_ablation_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\nSaved E05 to {out}')

if __name__ == '__main__':
    main()
