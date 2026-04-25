#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare detailed branching structure of src_v9 vs src_v10.

Outputs detailed CSVs showing:
  - which split / class generated which branches
  - root vs extra branch counts
  - branch confuser direction
  - basis dimension, member size, seed size, class scales
  - algorithm-specific fields (v9 score scale/bias, v10 radius/gating/gain)
  - per-branch member node lists and exploded rows
  - out-of-fold confusion counts used to drive branching
  - within-class overlap between branches

Place this file under project/scripts and run:
    python run_compare_branch_details_src_v9_src_v10.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import csv
import numpy as np
from scipy import sparse


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            pass
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'scripts').exists() and (p / 'src_v9').exists() and (p / 'src_v10').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts, src_v9, src_v10')


def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError('Cannot locate drive root containing planetoid/data')


def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def json_dumps_int_list(arr):
    return json.dumps([int(x) for x in np.asarray(arr, dtype=np.int64).tolist()], ensure_ascii=False)


def robust_overlap(a, b):
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    inter = len(sa & sb)
    union = max(1, len(sa | sb))
    return inter / union, inter, union


def build_features_generic(mod, data_base, dataset='chameleon'):
    raw_root = mod.find_raw_root(data_base)
    X_raw, y, A_sym = mod.load_chameleon_raw(raw_root)
    P = mod.row_normalize(A_sym)
    PX = np.asarray(P @ X_raw)
    P2X = np.asarray(P @ PX)
    F = np.hstack([
        mod.row_l2_normalize(X_raw),
        mod.row_l2_normalize(PX),
        mod.row_l2_normalize(P2X),
        mod.row_l2_normalize(X_raw - PX),
        mod.row_l2_normalize(PX - P2X),
    ])
    return raw_root, F, y


def run_v9(mod, raw_root, F, y, dim_candidates, num_splits):
    summary_rows = []
    branch_rows = []
    member_rows = []
    confusion_rows = []
    overlap_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])

        best_val = -1.0
        best_test = -1.0
        best_dim = None
        best_branch_models = None
        best_meta = None

        for dim in dim_candidates:
            branch_models, meta = mod.fit_adaptive_branch_subspaces(
                F, y, train_idx, classes, max_dim=dim,
                internal_n_folds=4,
                internal_seed=0,
                hard_residual_quantile=0.75,
                hard_margin_quantile=0.25,
                min_branch_size=10,
                expansion_ratio=2.0,
                max_extra_branches=2,
                repel_strength=0.08,
                extra_branch_bias=0.05,
            )
            class_scores, _ = mod.class_scores_from_branch_models(F, branch_models, classes)
            pred = classes[np.argmin(class_scores, axis=1)]
            val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
            test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_dim = int(dim)
                best_branch_models = branch_models
                best_meta = meta

        total_branches = 0
        extra_branches = 0
        for c in classes:
            bc = len(best_branch_models[int(c)])
            eb = sum(1 for b in best_branch_models[int(c)] if b.get('kind') == 'extra')
            total_branches += bc
            extra_branches += eb

        summary_rows.append({
            'algorithm': 'src_v9',
            'split': int(split),
            'best_dim': int(best_dim),
            'val_acc': float(best_val),
            'test_acc': float(best_test),
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
        })

        # OOF confusion counts used to decide branching
        root_conf = best_meta.get('root_confusion_counts', {}) if best_meta else {}
        incoming = best_meta.get('incoming_counts', {}) if best_meta else {}
        class_scales = best_meta.get('class_scales', {}) if best_meta else {}

        for true_c, conf_map in root_conf.items():
            for pred_c, cnt in conf_map.items():
                confusion_rows.append({
                    'algorithm': 'src_v9',
                    'split': int(split),
                    'kind': 'root_confusion',
                    'true_class': int(true_c),
                    'other_class': int(pred_c),
                    'count': int(cnt),
                })
        for target_c, conf_map in incoming.items():
            for from_c, cnt in conf_map.items():
                confusion_rows.append({
                    'algorithm': 'src_v9',
                    'split': int(split),
                    'kind': 'incoming_to_class',
                    'true_class': int(target_c),
                    'other_class': int(from_c),
                    'count': int(cnt),
                })

        for c in classes:
            branches = best_branch_models[int(c)]
            member_sets = []
            for b_idx, b in enumerate(branches):
                member_idx = np.asarray(b.get('member_idx', []), dtype=np.int64)
                total_branches += 0
                branch_rows.append({
                    'algorithm': 'src_v9',
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': str(b.get('kind', '')),
                    'confuser_class': int(b.get('confuser_class', -1)) if b.get('confuser_class', None) is not None else -1,
                    'basis_dim': int(b.get('basis', np.empty((F.shape[1], 0))).shape[1]),
                    'member_size': int(member_idx.size),
                    'seed_size': int(b.get('seed_size', -1)) if 'seed_size' in b else -1,
                    'class_scale': float(class_scales.get(int(c), np.nan)),
                    'score_scale': float(b.get('score_scale', np.nan)),
                    'score_bias': float(b.get('score_bias', np.nan)),
                    'radius': np.nan,
                    'extra_bias_scale': np.nan,
                    'gate_strength': np.nan,
                    'median_gain': np.nan,
                    'member_idx_json': json_dumps_int_list(member_idx),
                })
                for node_idx in member_idx:
                    member_rows.append({
                        'algorithm': 'src_v9',
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'kind': str(b.get('kind', '')),
                        'confuser_class': int(b.get('confuser_class', -1)) if b.get('confuser_class', None) is not None else -1,
                        'node_idx': int(node_idx),
                    })
                member_sets.append((b_idx, b.get('kind', ''), member_idx, b.get('confuser_class', -1)))

            for i in range(len(member_sets)):
                for j in range(i + 1, len(member_sets)):
                    i_idx, i_kind, i_members, i_conf = member_sets[i]
                    j_idx, j_kind, j_members, j_conf = member_sets[j]
                    ratio, inter, union = robust_overlap(i_members, j_members)
                    overlap_rows.append({
                        'algorithm': 'src_v9',
                        'split': int(split),
                        'class_label': int(c),
                        'branch_i': int(i_idx),
                        'branch_i_kind': str(i_kind),
                        'branch_i_confuser': int(i_conf) if i_conf is not None else -1,
                        'branch_j': int(j_idx),
                        'branch_j_kind': str(j_kind),
                        'branch_j_confuser': int(j_conf) if j_conf is not None else -1,
                        'intersection': int(inter),
                        'union': int(union),
                        'jaccard_overlap': float(ratio),
                    })

    return summary_rows, branch_rows, member_rows, confusion_rows, overlap_rows


def run_v10(mod, raw_root, F, y, dim_candidates, num_splits):
    summary_rows = []
    branch_rows = []
    member_rows = []
    confusion_rows = []
    overlap_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])

        best_val = -1.0
        best_test = -1.0
        best_dim = None
        best_branch_models = None
        best_meta = None

        for dim in dim_candidates:
            branch_models, meta = mod.fit_conservative_adaptive_branches(
                F, y, train_idx, classes, max_dim=dim,
                internal_n_folds=4,
                internal_seed=0,
                high_res_quantile=0.85,
                low_margin_quantile=0.15,
                min_seed_size=8,
                min_branch_size=12,
                local_quantile=0.35,
                max_group_frac=0.45,
                max_extra_branches=2,
                min_gain_ratio=0.10,
                repel_strength=0.03,
                extra_bias_scale=0.10,
                gate_strength=2.5,
            )
            scores, _ = mod.class_scores_from_branch_models(F, branch_models, classes)
            pred = classes[np.argmin(scores, axis=1)]
            val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
            test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_dim = int(dim)
                best_branch_models = branch_models
                best_meta = meta

        total_branches = 0
        extra_branches = 0
        for c in classes:
            bc = len(best_branch_models[int(c)])
            eb = sum(1 for b in best_branch_models[int(c)] if b.get('kind') == 'extra')
            total_branches += bc
            extra_branches += eb

        summary_rows.append({
            'algorithm': 'src_v10',
            'split': int(split),
            'best_dim': int(best_dim),
            'val_acc': float(best_val),
            'test_acc': float(best_test),
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
        })

        confusion_map = best_meta.get('confusion_counts', {}) if best_meta else {}
        class_scales = best_meta.get('class_scales', {}) if best_meta else {}
        branch_debug = best_meta.get('branch_debug', {}) if best_meta else {}

        for true_c, conf_map in confusion_map.items():
            for pred_c, cnt in conf_map.items():
                confusion_rows.append({
                    'algorithm': 'src_v10',
                    'split': int(split),
                    'kind': 'root_confusion',
                    'true_class': int(true_c),
                    'other_class': int(pred_c),
                    'count': int(cnt),
                })

        for c in classes:
            branches = best_branch_models[int(c)]
            member_sets = []
            for b_idx, b in enumerate(branches):
                member_idx = np.asarray(b.get('member_idx', []), dtype=np.int64)
                branch_rows.append({
                    'algorithm': 'src_v10',
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': str(b.get('kind', '')),
                    'confuser_class': int(b.get('confuser_class', -1)) if b.get('confuser_class', None) is not None else -1,
                    'basis_dim': int(b.get('basis', np.empty((F.shape[1], 0))).shape[1]),
                    'member_size': int(member_idx.size),
                    'seed_size': int(b.get('seed_size', -1)) if 'seed_size' in b else -1,
                    'class_scale': float(class_scales.get(int(c), np.nan)),
                    'score_scale': np.nan,
                    'score_bias': np.nan,
                    'radius': float(b.get('radius', np.nan)),
                    'extra_bias_scale': float(b.get('extra_bias_scale', np.nan)),
                    'gate_strength': float(b.get('gate_strength', np.nan)),
                    'median_gain': float(b.get('median_gain', np.nan)) if 'median_gain' in b else np.nan,
                    'member_idx_json': json_dumps_int_list(member_idx),
                })
                for node_idx in member_idx:
                    member_rows.append({
                        'algorithm': 'src_v10',
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'kind': str(b.get('kind', '')),
                        'confuser_class': int(b.get('confuser_class', -1)) if b.get('confuser_class', None) is not None else -1,
                        'node_idx': int(node_idx),
                    })
                member_sets.append((b_idx, b.get('kind', ''), member_idx, b.get('confuser_class', -1)))

            for dbg in branch_debug.get(int(c), []):
                confusion_rows.append({
                    'algorithm': 'src_v10',
                    'split': int(split),
                    'kind': 'branch_debug',
                    'true_class': int(c),
                    'other_class': int(dbg.get('confuser', -1)),
                    'count': int(dbg.get('seed_size', -1)),
                })

            for i in range(len(member_sets)):
                for j in range(i + 1, len(member_sets)):
                    i_idx, i_kind, i_members, i_conf = member_sets[i]
                    j_idx, j_kind, j_members, j_conf = member_sets[j]
                    ratio, inter, union = robust_overlap(i_members, j_members)
                    overlap_rows.append({
                        'algorithm': 'src_v10',
                        'split': int(split),
                        'class_label': int(c),
                        'branch_i': int(i_idx),
                        'branch_i_kind': str(i_kind),
                        'branch_i_confuser': int(i_conf) if i_conf is not None else -1,
                        'branch_j': int(j_idx),
                        'branch_j_kind': str(j_kind),
                        'branch_j_confuser': int(j_conf) if j_conf is not None else -1,
                        'intersection': int(inter),
                        'union': int(union),
                        'jaccard_overlap': float(ratio),
                    })

    return summary_rows, branch_rows, member_rows, confusion_rows, overlap_rows


def main():
    parser = argparse.ArgumentParser(description='Compare detailed branching structure of src_v9 vs src_v10')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    parser.add_argument('--src-v9-file', type=str, default=None)
    parser.add_argument('--src-v10-file', type=str, default=None)
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)

    default_v9 = find_first_existing([
        project_root / 'src_v9' / 'algo1_adaptive_branch_pca_src_v9.py',
        project_root / 'src_v9' / 'algo1_adaptive_branch_pca.py',
    ])
    default_v10 = find_first_existing([
        project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10.py',
        project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10(1).py',
    ])

    if default_v9 is None:
        raise FileNotFoundError('Cannot find src_v9 algorithm file')
    if default_v10 is None:
        raise FileNotFoundError('Cannot find src_v10 algorithm file')

    src_v9_file = Path(args.src_v9_file) if args.src_v9_file else default_v9
    src_v10_file = Path(args.src_v10_file) if args.src_v10_file else default_v10
    data_base = Path(args.data_base) if args.data_base else (drive_root / 'planetoid' / 'data' / args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else (project_root / 'scripts' / f'results_compare_branch_details_src_v9_src_v10_{args.dataset}')
    out_dir.mkdir(parents=True, exist_ok=True)

    mod_v9 = import_module_from_path(src_v9_file, 'src_v9_algo')
    mod_v10 = import_module_from_path(src_v10_file, 'src_v10_algo')

    raw_root_v9, F_v9, y_v9 = build_features_generic(mod_v9, data_base, dataset=args.dataset)
    raw_root_v10, F_v10, y_v10 = build_features_generic(mod_v10, data_base, dataset=args.dataset)

    # Sanity: should use the same raw ordering / labels / feature shape
    if F_v9.shape != F_v10.shape or not np.array_equal(y_v9, y_v10):
        raise RuntimeError('src_v9 and src_v10 produced inconsistent feature or label arrays')

    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root_v9}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F_v9.shape[1]}')

    v9_summary, v9_branches, v9_members, v9_conf, v9_overlap = run_v9(
        mod_v9, raw_root_v9, F_v9, y_v9, args.dims, args.num_splits
    )
    v10_summary, v10_branches, v10_members, v10_conf, v10_overlap = run_v10(
        mod_v10, raw_root_v10, F_v10, y_v10, args.dims, args.num_splits
    )

    summary_rows = v9_summary + v10_summary
    branch_rows = v9_branches + v10_branches
    member_rows = v9_members + v10_members
    confusion_rows = v9_conf + v10_conf
    overlap_rows = v9_overlap + v10_overlap

    write_csv(out_dir / 'comparison_summary_by_split.csv', summary_rows)
    write_csv(out_dir / 'branch_subspace_summary_src_v9_src_v10.csv', branch_rows)
    write_csv(out_dir / 'branch_member_rows_src_v9_src_v10.csv', member_rows)
    write_csv(out_dir / 'oof_confusion_and_branch_debug_src_v9_src_v10.csv', confusion_rows)
    write_csv(out_dir / 'branch_overlap_within_class_src_v9_src_v10.csv', overlap_rows)

    # Compact per-class branch counts / extra counts
    per_class_count_rows = []
    for algo in ['src_v9', 'src_v10']:
        algo_branch_rows = [r for r in branch_rows if r['algorithm'] == algo]
        for split in sorted(set(r['split'] for r in algo_branch_rows)):
            split_rows = [r for r in algo_branch_rows if r['split'] == split]
            for c in sorted(set(r['class_label'] for r in split_rows)):
                rows_c = [r for r in split_rows if r['class_label'] == c]
                per_class_count_rows.append({
                    'algorithm': algo,
                    'split': int(split),
                    'class_label': int(c),
                    'branch_count': int(len(rows_c)),
                    'extra_branch_count': int(sum(1 for r in rows_c if r['kind'] == 'extra')),
                    'extra_confusers_json': json.dumps([int(r['confuser_class']) for r in rows_c if r['kind'] == 'extra'], ensure_ascii=False),
                })
    write_csv(out_dir / 'branch_count_by_class_src_v9_src_v10.csv', per_class_count_rows)

    # Mean summary per algorithm
    mean_rows = []
    for algo in ['src_v9', 'src_v10']:
        rows = [r for r in summary_rows if r['algorithm'] == algo]
        vals = np.asarray([r['val_acc'] for r in rows], dtype=np.float64)
        tests = np.asarray([r['test_acc'] for r in rows], dtype=np.float64)
        mean_rows.append({
            'algorithm': algo,
            'num_splits': int(len(rows)),
            'val_mean': float(np.mean(vals)),
            'val_std': float(np.std(vals)),
            'test_mean': float(np.mean(tests)),
            'test_std': float(np.std(tests)),
            'avg_total_branches': float(np.mean([r['total_branches'] for r in rows])),
            'avg_extra_branches': float(np.mean([r['extra_branches'] for r in rows])),
        })
    write_csv(out_dir / 'comparison_mean_summary_src_v9_src_v10.csv', mean_rows)

    print() 
    for r in mean_rows:
        print(
            f"{r['algorithm']}: val_mean={r['val_mean']:.4f} +- {r['val_std']:.4f}, "
            f"test_mean={r['test_mean']:.4f} +- {r['test_std']:.4f}, "
            f"avg_branches={r['avg_total_branches']:.2f} (extra={r['avg_extra_branches']:.2f})"
        )

    print('\nWrote:')
    print(f'  {out_dir / "comparison_summary_by_split.csv"}')
    print(f'  {out_dir / "comparison_mean_summary_src_v9_src_v10.csv"}')
    print(f'  {out_dir / "branch_count_by_class_src_v9_src_v10.csv"}')
    print(f'  {out_dir / "branch_subspace_summary_src_v9_src_v10.csv"}')
    print(f'  {out_dir / "branch_member_rows_src_v9_src_v10.csv"}')
    print(f'  {out_dir / "oof_confusion_and_branch_debug_src_v9_src_v10.csv"}')
    print(f'  {out_dir / "branch_overlap_within_class_src_v9_src_v10.csv"}')


if __name__ == '__main__':
    main()


