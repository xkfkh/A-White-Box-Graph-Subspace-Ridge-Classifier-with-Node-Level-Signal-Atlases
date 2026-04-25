#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run src_v13 per-class dimension + coverage-guided adaptive branch with audit.

Outputs are intentionally similar to previous v10/v12 audit scripts:
  - split summary
  - test node predictions
  - misclassified test nodes
  - branch summaries and member rows
  - per-class dimension search trace
  - OOF confusion / branch debug
  - extra branch effect summary and rewrite nodes
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            pass
        return
    keys, seen = [], set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in keys})


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
        if (p / 'scripts').exists() and (p / 'src_v13').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts and src_v13')


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


def resolve_default_paths(dataset: str, src_v13_file: str | None):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v13_perclass_dim_coverage_audit_{dataset}'

    if src_v13_file:
        src_path = Path(src_v13_file)
    else:
        src_path = find_first_existing([
            project_root / 'src_v13' / 'algo1_multihop_pca_perclass_dim_coverage_branch_src_v13.py',
            project_root / 'src_v13' / 'algo1_multihop_pca_perclass_dim_coverage_branch.py',
        ])
    if src_path is None or not src_path.exists():
        raise FileNotFoundError('Cannot find src_v13 algorithm file. Please pass --src-v13-file explicitly.')
    return project_root, drive_root, data_base, out_dir, src_path


def pairwise_branch_geometry(branch_a, branch_b):
    mu_a = np.asarray(branch_a['mu'], dtype=np.float64)
    mu_b = np.asarray(branch_b['mu'], dtype=np.float64)
    B_a = np.asarray(branch_a['basis'], dtype=np.float64)
    B_b = np.asarray(branch_b['basis'], dtype=np.float64)

    center_l2 = float(np.linalg.norm(mu_a - mu_b))
    center_sq = float(np.sum((mu_a - mu_b) ** 2))
    if B_a.size == 0 or B_b.size == 0:
        overlap = 0.0
        min_angle = np.nan
        mean_cos2 = 0.0
    else:
        cross = B_a.T @ B_b
        s = np.linalg.svd(cross, compute_uv=False)
        overlap = float(np.linalg.norm(cross, ord='fro'))
        mean_cos2 = float(np.mean(s * s)) if s.size > 0 else 0.0
        min_angle = float(np.degrees(np.arccos(np.clip(np.max(s), -1.0, 1.0)))) if s.size > 0 else np.nan
    return center_l2, center_sq, overlap, mean_cos2, min_angle


def compute_exposures(mod, A_sym, y, train_idx, classes):
    P = mod.row_normalize(A_sym)
    n = A_sym.shape[0]
    cnum = len(classes)
    train_exp_1 = np.zeros((n, cnum), dtype=np.float64)
    train_exp_2 = np.zeros((n, cnum), dtype=np.float64)
    true_exp_1 = np.zeros((n, cnum), dtype=np.float64)
    true_exp_2 = np.zeros((n, cnum), dtype=np.float64)
    for pos, c in enumerate(classes):
        c = int(c)
        mask_true = (y == c).astype(np.float64)
        mask_train = np.zeros(n, dtype=np.float64)
        cls_train = train_idx[y[train_idx] == c]
        mask_train[cls_train] = 1.0

        true_exp_1[:, pos] = np.asarray(P @ mask_true).ravel()
        true_exp_2[:, pos] = np.asarray(P @ true_exp_1[:, pos]).ravel()
        train_exp_1[:, pos] = np.asarray(P @ mask_train).ravel()
        train_exp_2[:, pos] = np.asarray(P @ train_exp_1[:, pos]).ravel()
    return true_exp_1, true_exp_2, train_exp_1, train_exp_2


def scores_with_removed_branch(mod, F, branch_models, classes, remove_class, remove_branch_idx):
    reduced = {}
    for c in classes:
        c = int(c)
        reduced[c] = []
        for i, b in enumerate(branch_models[c]):
            if c == int(remove_class) and i == int(remove_branch_idx):
                continue
            reduced[c].append(b)
    return mod.class_scores_from_branch_models(F, reduced, classes)


def run_audit(mod, dataset='chameleon', data_base=None, out_dir=None, dim_candidates=None, num_splits=10):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    raw_root = mod.find_raw_root(data_base)
    X_raw, y, A_sym, original_node_ids = mod.load_chameleon_raw(raw_root)

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

    branch_kwargs = dict(
        dim_search_kwargs=dict(
            max_passes=2,
            coverage_weight=0.010,
            explained_weight=0.004,
            fp_weight=0.020,
            dim_penalty=0.001,
            max_val_acc_drop=0.0,
            max_break_rate=0.015,
        ),
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

    split_rows = []
    node_rows = []
    mis_rows = []
    branch_rows = []
    member_rows = []
    dim_trace_rows = []
    oof_rows = []
    extra_effect_rows = []
    extra_rewrite_rows = []
    branch_pair_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        class_to_pos = {int(c): pos for pos, c in enumerate(classes)}

        branch_models, meta = mod.fit_perclass_coverage_adaptive_branches(
            F, y, train_idx, val_idx, classes,
            dim_candidates=dim_candidates,
            **branch_kwargs,
        )
        scores, best_kind, best_branch_idx = mod.class_scores_from_branch_models(F, branch_models, classes)
        pred = classes[np.argmin(scores, axis=1)]

        root_only = {int(c): [branch_models[int(c)][0]] for c in classes}
        root_scores, root_kind, root_branch_idx = mod.class_scores_from_branch_models(F, root_only, classes)
        root_pred = classes[np.argmin(root_scores, axis=1)]

        val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
        test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
        root_val_acc = float(np.mean(root_pred[val_idx] == y[val_idx]))
        root_test_acc = float(np.mean(root_pred[test_idx] == y[test_idx]))

        total_branches = sum(len(branch_models[int(c)]) for c in classes)
        extra_branches = sum(1 for c in classes for b in branch_models[int(c)] if b['kind'] == 'extra')

        full_correct = pred[test_idx] == y[test_idx]
        root_correct = root_pred[test_idx] == y[test_idx]
        fixed_vs_root = int(np.sum((~root_correct) & full_correct))
        broken_vs_root = int(np.sum(root_correct & (~full_correct)))
        redirect_vs_root = int(np.sum((~root_correct) & (~full_correct) & (root_pred[test_idx] != pred[test_idx])))

        split_row = {
            'split': int(split),
            'class_dims_json': json.dumps({str(k): int(v) for k, v in meta['class_dims'].items()}, ensure_ascii=False),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'root_val_acc': root_val_acc,
            'root_test_acc': root_test_acc,
            'fixed_vs_root': fixed_vs_root,
            'broken_vs_root': broken_vs_root,
            'redirect_vs_root': redirect_vs_root,
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
        }
        for c in classes:
            c = int(c)
            split_row[f'class_{c}_dim'] = int(meta['class_dims'][c])
            split_row[f'class_{c}_branch_count'] = int(len(branch_models[c]))
            split_row[f'class_{c}_extra_branch_count'] = int(sum(1 for b in branch_models[c] if b['kind'] == 'extra'))
        split_rows.append(split_row)

        print(
            f"split={split:2d} dims={split_row['class_dims_json']} "
            f"val={val_acc:.4f} test={test_acc:.4f} root_test={root_test_acc:.4f} "
            f"branches={total_branches} (extra={extra_branches})"
        )

        # Dimension trace
        for r in meta.get('dim_trace', []):
            rr = dict(r)
            rr['split'] = int(split)
            dim_trace_rows.append(rr)

        # OOF confusion and branch debug
        conf = meta.get('confusion_counts', {})
        for true_c, mp in conf.items():
            for pred_c, cnt in mp.items():
                oof_rows.append({
                    'split': int(split),
                    'kind': 'oof_root_confusion',
                    'true_class': int(true_c),
                    'other_class': int(pred_c),
                    'count': int(cnt),
                })
        for c, debug_list in meta.get('branch_debug', {}).items():
            for idx, d in enumerate(debug_list):
                row = {'split': int(split), 'kind': 'branch_debug', 'class_label': int(c), 'debug_index': int(idx)}
                row.update(d)
                oof_rows.append(row)

        # Branch summaries and members
        for c in classes:
            c = int(c)
            for b_idx, b in enumerate(branch_models[c]):
                member_idx = np.asarray(b.get('member_idx', []), dtype=np.int64)
                confuser = b.get('confuser_class', '')
                branch_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'class_dim': int(meta['class_dims'][c]),
                    'branch_index': int(b_idx),
                    'kind': b.get('kind', ''),
                    'confuser_class': confuser,
                    'basis_dim': int(np.asarray(b['basis']).shape[1]),
                    'feature_dim': int(np.asarray(b['basis']).shape[0]),
                    'member_size': int(member_idx.size),
                    'seed_size': b.get('seed_size', ''),
                    'median_gain': b.get('median_gain', ''),
                    'max_dim_used': b.get('max_dim_used', ''),
                    'radius': float(b.get('radius', np.nan)),
                    'class_scale': float(b.get('class_scale', np.nan)),
                    'extra_bias_scale': float(b.get('extra_bias_scale', np.nan)),
                    'gate_strength': float(b.get('gate_strength', np.nan)),
                    'member_idx_json': json.dumps([int(x) for x in member_idx.tolist()], ensure_ascii=False),
                    'member_original_id_json': json.dumps([int(original_node_ids[x]) for x in member_idx.tolist()], ensure_ascii=False),
                })
                for node in member_idx:
                    member_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'class_dim': int(meta['class_dims'][c]),
                        'branch_index': int(b_idx),
                        'kind': b.get('kind', ''),
                        'confuser_class': confuser,
                        'node_idx_sorted': int(node),
                        'original_node_id': int(original_node_ids[node]),
                    })

        # Pairwise branch geometry
        all_branch_refs = []
        for c in classes:
            c = int(c)
            for b_idx, b in enumerate(branch_models[c]):
                all_branch_refs.append((c, b_idx, b))
        for i in range(len(all_branch_refs)):
            ci, bi, bri = all_branch_refs[i]
            for j in range(i + 1, len(all_branch_refs)):
                cj, bj, brj = all_branch_refs[j]
                center_l2, center_sq, overlap, mean_cos2, min_angle = pairwise_branch_geometry(bri, brj)
                branch_pair_rows.append({
                    'split': int(split),
                    'class_i': int(ci),
                    'branch_i': int(bi),
                    'kind_i': bri.get('kind', ''),
                    'class_j': int(cj),
                    'branch_j': int(bj),
                    'kind_j': brj.get('kind', ''),
                    'center_l2_distance': center_l2,
                    'center_sq_distance': center_sq,
                    'subspace_overlap_fro': overlap,
                    'mean_cos2_principal': mean_cos2,
                    'min_principal_angle_deg': min_angle,
                })

        true_exp_1, true_exp_2, train_exp_1, train_exp_2 = compute_exposures(mod, A_sym, y, train_idx, classes)

        # Node-level rows
        for node in test_idx:
            true_label = int(y[node])
            true_pos = class_to_pos[true_label]
            pred_label = int(pred[node])
            pred_pos = class_to_pos[pred_label]
            root_pred_label = int(root_pred[node])

            true_branch_i = int(best_branch_idx[node, true_pos])
            pred_branch_i = int(best_branch_idx[node, pred_pos])
            true_branch = branch_models[true_label][true_branch_i]
            pred_branch = branch_models[pred_label][pred_branch_i]

            comp_true = mod.branch_score_components(F[[node]], true_branch)
            comp_pred = mod.branch_score_components(F[[node]], pred_branch)

            st = {k: float(v[0]) for k, v in comp_true.items()}
            sp = {k: float(v[0]) for k, v in comp_pred.items()}

            center_l2, center_sq, overlap, mean_cos2, min_angle = pairwise_branch_geometry(true_branch, pred_branch)

            # top2 gap
            row_scores = scores[node].copy()
            order = np.argsort(row_scores)
            top2_gap = float(row_scores[order[1]] - row_scores[order[0]]) if len(order) >= 2 else np.nan

            node_row = {
                'split': int(split),
                'node_idx_sorted': int(node),
                'original_node_id': int(original_node_ids[node]),
                'true_label': true_label,
                'root_pred_label': root_pred_label,
                'adaptive_pred_label': pred_label,
                'is_correct_root': int(root_pred_label == true_label),
                'is_correct_full': int(pred_label == true_label),
                'class_dims_json': json.dumps({str(k): int(v) for k, v in meta['class_dims'].items()}, ensure_ascii=False),
                'true_class_dim': int(meta['class_dims'][true_label]),
                'pred_class_dim': int(meta['class_dims'][pred_label]),
                'pred_branch_kind': pred_branch.get('kind', ''),
                'pred_branch_index': int(pred_branch_i),
                'pred_branch_confuser_class': pred_branch.get('confuser_class', ''),
                'true_branch_kind': true_branch.get('kind', ''),
                'true_branch_index': int(true_branch_i),
                'true_branch_confuser_class': true_branch.get('confuser_class', ''),
                'adaptive_top2_gap': top2_gap,
                'adaptive_true_score': st['score'],
                'adaptive_pred_score': sp['score'],
                'adaptive_true_total_sq': st['total'],
                'adaptive_pred_total_sq': sp['total'],
                'adaptive_true_evidence_sq': st['evidence'],
                'adaptive_pred_evidence_sq': sp['evidence'],
                'adaptive_true_residual': st['residual'],
                'adaptive_pred_residual': sp['residual'],
                'true_explained_ratio': st['evidence'] / max(st['total'], 1e-12),
                'pred_explained_ratio': sp['evidence'] / max(sp['total'], 1e-12),
                'delta_total_pred_minus_true': sp['total'] - st['total'],
                'delta_evidence_pred_minus_true': sp['evidence'] - st['evidence'],
                'delta_residual_pred_minus_true': sp['residual'] - st['residual'],
                'delta_explained_ratio_pred_minus_true': (sp['evidence'] / max(sp['total'], 1e-12)) - (st['evidence'] / max(st['total'], 1e-12)),
                'pred_bias_penalty': sp['bias_penalty'],
                'pred_gate_penalty': sp['gate_penalty'],
                'pred_normalized_center_dist': sp['normalized_center_dist'],
                'true_pred_branch_center_l2_distance': center_l2,
                'true_pred_branch_center_sq_distance': center_sq,
                'true_pred_branch_subspace_overlap_fro': overlap,
                'true_pred_branch_mean_cos2_principal': mean_cos2,
                'true_pred_branch_min_principal_angle_deg': min_angle,
                'train_exposure_true_1hop': float(train_exp_1[node, true_pos]),
                'train_exposure_pred_1hop': float(train_exp_1[node, pred_pos]),
                'train_exposure_true_2hop': float(train_exp_2[node, true_pos]),
                'train_exposure_pred_2hop': float(train_exp_2[node, pred_pos]),
                'true_exposure_true_1hop': float(true_exp_1[node, true_pos]),
                'true_exposure_pred_1hop': float(true_exp_1[node, pred_pos]),
                'true_exposure_true_2hop': float(true_exp_2[node, true_pos]),
                'true_exposure_pred_2hop': float(true_exp_2[node, pred_pos]),
            }
            node_rows.append(node_row)
            if pred_label != true_label:
                mis_rows.append(node_row)

        # Extra branch effect by removing one extra branch at a time.
        full_pred_test = pred[test_idx]
        full_correct_test = full_pred_test == y[test_idx]
        full_pred_val = pred[val_idx]
        full_correct_val = full_pred_val == y[val_idx]

        for c in classes:
            c = int(c)
            for b_idx, b in enumerate(branch_models[c]):
                if b.get('kind') != 'extra':
                    continue
                scores_rm, _, _ = scores_with_removed_branch(mod, F, branch_models, classes, c, b_idx)
                pred_rm = classes[np.argmin(scores_rm, axis=1)]

                rm_test = pred_rm[test_idx]
                rm_correct_test = rm_test == y[test_idx]
                rm_val = pred_rm[val_idx]
                rm_correct_val = rm_val == y[val_idx]

                test_fixed = int(np.sum(full_correct_test & (~rm_correct_test)))
                test_broken = int(np.sum((~full_correct_test) & rm_correct_test))
                test_redirect = int(np.sum((~full_correct_test) & (~rm_correct_test) & (full_pred_test != rm_test)))

                val_fixed = int(np.sum(full_correct_val & (~rm_correct_val)))
                val_broken = int(np.sum((~full_correct_val) & rm_correct_val))
                val_redirect = int(np.sum((~full_correct_val) & (~rm_correct_val) & (full_pred_val != rm_val)))

                extra_effect_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'confuser_class': b.get('confuser_class', ''),
                    'class_dim': int(meta['class_dims'][c]),
                    'basis_dim': int(np.asarray(b['basis']).shape[1]),
                    'member_size': int(len(np.asarray(b.get('member_idx', [])))),
                    'seed_size': b.get('seed_size', ''),
                    'median_gain': b.get('median_gain', ''),
                    'val_fixed_by_this_branch': val_fixed,
                    'val_broken_by_this_branch': val_broken,
                    'val_wrong_redirect_by_this_branch': val_redirect,
                    'test_fixed_by_this_branch': test_fixed,
                    'test_broken_by_this_branch': test_broken,
                    'test_wrong_redirect_by_this_branch': test_redirect,
                    'net_fix_test': int(test_fixed - test_broken),
                    'net_fix_val': int(val_fixed - val_broken),
                })

                changed = test_idx[full_pred_test != rm_test]
                for node in changed:
                    extra_rewrite_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'confuser_class': b.get('confuser_class', ''),
                        'node_idx_sorted': int(node),
                        'original_node_id': int(original_node_ids[node]),
                        'true_label': int(y[node]),
                        'full_pred_label': int(pred[node]),
                        'removed_branch_pred_label': int(pred_rm[node]),
                        'full_correct': int(pred[node] == y[node]),
                        'removed_correct': int(pred_rm[node] == y[node]),
                        'effect_type': (
                            'fixed_by_branch' if (pred[node] == y[node] and pred_rm[node] != y[node])
                            else 'broken_by_branch' if (pred[node] != y[node] and pred_rm[node] == y[node])
                            else 'wrong_redirect_by_branch' if (pred[node] != y[node] and pred_rm[node] != y[node])
                            else 'changed_between_correct_classes'
                        ),
                    })

    vals = np.asarray([r['val_acc'] for r in split_rows], dtype=np.float64)
    tests = np.asarray([r['test_acc'] for r in split_rows], dtype=np.float64)
    summary = {
        'dataset': dataset,
        'data_base': str(data_base),
        'raw_root': str(raw_root),
        'out_dir': str(out_dir),
        'dim_candidates': json.dumps([int(x) for x in dim_candidates], ensure_ascii=False),
        'num_splits': int(num_splits),
        'val_mean': float(np.mean(vals)),
        'val_std': float(np.std(vals)),
        'test_mean': float(np.mean(tests)),
        'test_std': float(np.std(tests)),
    }

    write_csv(out_dir / 'split_summary_src_v13_perclass_dim_coverage_audit.csv', split_rows)
    write_csv(out_dir / 'test_node_predictions_all_src_v13.csv', node_rows)
    write_csv(out_dir / 'misclassified_test_nodes_src_v13.csv', mis_rows)
    write_csv(out_dir / 'branch_subspace_summary_src_v13.csv', branch_rows)
    write_csv(out_dir / 'branch_member_rows_src_v13.csv', member_rows)
    write_csv(out_dir / 'dim_search_trace_src_v13.csv', dim_trace_rows)
    write_csv(out_dir / 'oof_confusion_and_branch_debug_src_v13.csv', oof_rows)
    write_csv(out_dir / 'extra_branch_effect_summary_src_v13.csv', extra_effect_rows)
    write_csv(out_dir / 'extra_branch_rewrite_nodes_src_v13.csv', extra_rewrite_rows)
    write_csv(out_dir / 'pairwise_branch_geometry_src_v13.csv', branch_pair_rows)

    (out_dir / 'run_summary_src_v13_perclass_dim_coverage_audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print()
    print(f"val_mean  = {summary['val_mean']:.4f} +- {summary['val_std']:.4f}")
    print(f"test_mean = {summary['test_mean']:.4f} +- {summary['test_std']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run src_v13 per-class dimension coverage audit.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--src-v13-file', type=str, default=None)
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    project_root, drive_root, default_data_base, default_out_dir, src_path = resolve_default_paths(args.dataset, args.src_v13_file)
    data_base = Path(args.data_base) if args.data_base else default_data_base
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    mod = import_module_from_path(src_path, 'src_v13_algo')
    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
    print(f'Src_v13 file: {src_path}')

    run_audit(
        mod,
        dataset=args.dataset,
        data_base=data_base,
        out_dir=out_dir,
        dim_candidates=[int(x) for x in args.dims],
        num_splits=int(args.num_splits),
    )


if __name__ == '__main__':
    main()


