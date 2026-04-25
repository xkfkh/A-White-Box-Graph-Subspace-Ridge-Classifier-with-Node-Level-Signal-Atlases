#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E11 sample-efficiency learning curve.

Compares src_v16c with closed-form representative baselines under exactly the
same generated few-shot train/val/test masks.

For GNN/other external baselines, the script additionally writes the split
indices so an external PyG runner can reuse the exact masks.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from paperexp.core import *


def save_split_npz(out, dataset, k, repeat, train_idx, val_idx, test_idx, n):
    mask_tr = np.zeros(n, dtype=bool); mask_va = np.zeros(n, dtype=bool); mask_te = np.zeros(n, dtype=bool)
    mask_tr[train_idx] = True; mask_va[val_idx] = True; mask_te[test_idx] = True
    path = out / 'fewshot_masks' / dataset
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / f'{dataset}_fewshot_k{k}_repeat{repeat}.npz', train_mask=mask_tr, val_mask=mask_va, test_mask=mask_te,
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=None)
    p.add_argument('--datasets', nargs='+', default=['chameleon','squirrel','cornell','texas','wisconsin','amazon-photo','amazon-computers'])
    p.add_argument('--k-list', default='1,2,3,5,8,10,15,20')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--val-per-class', type=int, default=30)
    p.add_argument('--seed0', type=int, default=11000)
    p.add_argument('--grid', choices=['default','fast'], default='fast')
    p.add_argument('--out-dir', default='results/E11_sample_efficiency')
    args = p.parse_args()
    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    grid = grid_from_name(args.grid)
    k_list = [int(x) for x in args.k_list.split(',')]
    rows, traces = [], []
    for dataset in args.datasets:
        graph = load_dataset(root, dataset)
        classes = np.unique(graph.y)
        for k in k_list:
            for rep in range(args.repeats):
                seed = args.seed0 + 1000*k + rep
                train_idx, val_idx, test_idx = fewshot_split(graph.y, train_per_class=k, seed=seed, val_per_class=args.val_per_class)
                save_split_npz(out, dataset, k, rep, train_idx, val_idx, test_idx, graph.num_nodes)
                trace = []
                cfg, pred, info, metrics = select_by_validation(graph, train_idx, val_idx, test_idx, grid, trace=trace)
                rows.append(dict(experiment='E11', dataset=dataset, train_per_class=k, repeat=rep, seed=seed,
                                 method='src_v16c', train_size=len(train_idx), val_size=len(val_idx), test_size=len(test_idx),
                                 **asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)), **metrics,
                                 selection_rule='fewshot_grid_selected_by_val_only_no_test'))
                for t in trace:
                    t.update(dict(dataset=dataset, train_per_class=k, repeat=rep, method='src_v16c'))
                traces.extend(trace)
                for b in simple_baseline_grid(graph, train_idx, val_idx, test_idx):
                    rows.append(dict(experiment='E11', dataset=dataset, train_per_class=k, repeat=rep, seed=seed,
                                     method=b['method'], train_size=len(train_idx), val_size=len(val_idx), test_size=len(test_idx),
                                     val_acc=b['val_acc'], test_acc=b['test_acc'], alpha=b['alpha'], selection_rule=b['selection_rule']))
                print(f"{dataset} k={k} rep={rep} src16c val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f}")
    write_csv(out/'E11_sample_efficiency_rows.csv', rows)
    write_csv(out/'E11_src_v16c_candidate_trace.csv', traces)
    df = pd.DataFrame(rows)
    df.groupby(['dataset','method','train_per_class']).agg(n=('test_acc','size'), val_mean=('val_acc','mean'), val_std=('val_acc','std'), test_mean=('test_acc','mean'), test_std=('test_acc','std')).reset_index().to_csv(out/'E11_learning_curve_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\nSaved E11 to {out}')
    print('Few-shot masks for external PyG baselines were saved under results/E11_sample_efficiency/fewshot_masks/')

if __name__ == '__main__':
    main()
