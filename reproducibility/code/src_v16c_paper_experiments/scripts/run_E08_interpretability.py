#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E08 interpretability experiment.

Statistics:
- selected fusion weight w distribution
- feature-block Fisher importance distribution
- per-class PCA retained dimensions and variance
- confusion directions and top-2 margins
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
    p.add_argument('--datasets', nargs='+', default=['chameleon','squirrel','cornell','texas','wisconsin'])
    p.add_argument('--splits', type=int, default=10)
    p.add_argument('--grid', choices=['default','fast'], default='default')
    p.add_argument('--out-dir', default='results/E08_interpretability')
    args = p.parse_args()
    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    grid = grid_from_name(args.grid)
    summary_rows, block_rows, pca_rows, node_rows, confusion_rows = [], [], [], [], []
    for dataset in args.datasets:
        graph = load_dataset(root, dataset)
        for split in range(args.splits):
            if has_fixed_splits(graph):
                train_idx, val_idx, test_idx = load_fixed_split(graph, split)
                policy = 'fixed_geom_gcn_split'
            else:
                train_idx, val_idx, test_idx, policy = random_split_matching_protocol(graph, 3000+split, prefer_fixed_counts=False)
            cfg, pred, info, metrics = select_by_validation(graph, train_idx, val_idx, test_idx, grid)
            summary_rows.append(dict(dataset=dataset, split=split, split_policy=policy, **asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)), **metrics, pca_dims_json=json.dumps(info['pca_dims'], ensure_ascii=False)))
            fish = info['fisher_scores']
            sel = set(int(i) for i in info['selector'])
            for name, st, ed in info['feature_blocks']:
                idx = np.arange(st, ed)
                block_rows.append(dict(dataset=dataset, split=split, block=name, block_dim=int(ed-st), fisher_sum=float(np.sum(fish[idx])), fisher_mean=float(np.mean(fish[idx])), selected_count=int(sum(int(i) in sel for i in idx)), selected_ratio=float(sum(int(i) in sel for i in idx)/max(1,ed-st))))
            for cls, dim in info['pca_dims'].items():
                pca_rows.append(dict(dataset=dataset, split=split, class_label=int(cls), pca_dim=int(dim), retained_variance=float(info['retained_variance'][int(cls)])))
            scores = info['scores']
            classes = np.unique(graph.y[train_idx])
            order = np.argsort(scores, axis=1)
            gap = scores[np.arange(scores.shape[0]), order[:,1]] - scores[np.arange(scores.shape[0]), order[:,0]]
            for node in test_idx:
                yt, yp = int(graph.y[node]), int(pred[node])
                node_rows.append(dict(dataset=dataset, split=split, node_idx=int(node), true_label=yt, pred_label=yp, correct=int(yt==yp), top1=int(classes[order[node,0]]), top2=int(classes[order[node,1]]), top2_gap=float(gap[node])))
                if yt != yp:
                    confusion_rows.append(dict(dataset=dataset, split=split, true_label=yt, pred_label=yp, count=1))
            print(f"{dataset} split={split} val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f} w={cfg.w}")
    write_csv(out/'E08_selected_config_summary.csv', summary_rows)
    write_csv(out/'E08_feature_block_fisher_importance.csv', block_rows)
    write_csv(out/'E08_pca_dimension_by_class.csv', pca_rows)
    write_csv(out/'E08_test_node_margin_diagnostics.csv', node_rows)
    cdf = pd.DataFrame(confusion_rows)
    if len(cdf):
        cdf.groupby(['dataset','true_label','pred_label']).size().reset_index(name='count').to_csv(out/'E08_confusion_direction_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\nSaved E08 to {out}')

if __name__ == '__main__':
    main()
