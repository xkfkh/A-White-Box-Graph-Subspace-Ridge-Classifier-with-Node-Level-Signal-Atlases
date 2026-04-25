
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run src_v11 geometry-coupled safe adaptive-branch PCA and record:
1) node-level test predictions / errors with related features
2) root-initial -> root-calibrated -> full branch effect
3) per-extra-branch marginal impact (fix / break counts)
4) branch metadata, members, and calibration diagnostics

Place under project/scripts and run:
    python run_src_v11_geometry_safe_branch_with_audit.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from copy import deepcopy
from typing import Dict, List

import numpy as np


# ============================================================
# Generic utilities
# ============================================================

def write_csv(path: Path, rows: List[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            pass
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    normalized_rows = [{k: row.get(k, '') for k in keys} for row in rows]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(normalized_rows)


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
        if (p / 'scripts').exists() and (p / 'src_v11').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts and src_v11')


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


def resolve_default_paths(dataset: str, src_v11_file: str | None):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v11_geometry_safe_branch_audit_{dataset}'

    if src_v11_file is not None:
        src_v11_path = Path(src_v11_file)
    else:
        src_v11_path = find_first_existing([
            project_root / 'src_v11' / 'algo1_multihop_pca_geometry_coupled_safe_branch_src_v11.py',
            project_root / 'src_v11' / 'algo1_multihop_pca_geometry_coupled_safe_branch.py',
        ])
    if src_v11_path is None or not src_v11_path.exists():
        raise FileNotFoundError('Cannot find src_v11 algorithm file in src_v11. Please pass --src-v11-file explicitly.')

    return project_root, drive_root, data_base, out_dir, src_v11_path


def safe_mean(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.mean(a))


# ============================================================
# Feature helpers
# ============================================================

def build_features(mod, data_base: Path):
    raw_root = mod.find_raw_root(data_base)
    X_raw, y, A_sym, original_ids = mod.load_chameleon_raw(raw_root)
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
    return raw_root, F, y, A_sym, original_ids


def compute_exposure_features(mod, A_sym, y, train_idx, classes):
    P = mod.row_normalize(A_sym)
    n = A_sym.shape[0]
    cnum = len(classes)
    true_exp_1 = np.zeros((n, cnum), dtype=np.float64)
    true_exp_2 = np.zeros((n, cnum), dtype=np.float64)
    train_exp_1 = np.zeros((n, cnum), dtype=np.float64)
    train_exp_2 = np.zeros((n, cnum), dtype=np.float64)
    for pos, c in enumerate(classes):
        mask_true = (y == c).astype(np.float64)
        mask_train = np.zeros(n, dtype=np.float64)
        cls_train = train_idx[y[train_idx] == c]
        mask_train[cls_train] = 1.0
        true_exp_1[:, pos] = np.asarray(P @ mask_true).ravel()
        true_exp_2[:, pos] = np.asarray(P @ true_exp_1[:, pos]).ravel()
        train_exp_1[:, pos] = np.asarray(P @ mask_train).ravel()
        train_exp_2[:, pos] = np.asarray(P @ train_exp_1[:, pos]).ravel()
    return true_exp_1, true_exp_2, train_exp_1, train_exp_2


# ============================================================
# Model evaluation helpers
# ============================================================

def root_prediction_from_subspaces(mod, F, root_subspaces, classes):
    R = mod.residuals_to_root_subspaces(F, root_subspaces, classes)
    pred = classes[np.argmin(R, axis=1)]
    return pred, R


def get_branch_component_tensor(mod, F, branch_models, classes):
    classes = np.asarray(classes, dtype=np.int64)
    tensor = {}
    best_score = np.full((F.shape[0], len(classes)), np.inf, dtype=np.float64)
    best_branch_idx = np.full((F.shape[0], len(classes)), -1, dtype=np.int64)
    best_kind = np.empty((F.shape[0], len(classes)), dtype=object)

    per_branch_components = {}
    for pos, c in enumerate(classes):
        per_branch_components[int(c)] = []
        for b_idx, b in enumerate(branch_models[int(c)]):
            comp = mod.branch_score_components(F, b)
            per_branch_components[int(c)].append(comp)
            s = comp['score']
            better = s < best_score[:, pos]
            best_score[better, pos] = s[better]
            best_branch_idx[better, pos] = int(b_idx)
            best_kind[better, pos] = b['kind']
    tensor['best_score'] = best_score
    tensor['best_branch_idx'] = best_branch_idx
    tensor['best_kind'] = best_kind
    tensor['per_branch_components'] = per_branch_components
    return tensor


def summarize_rewrite(old_pred, new_pred, y, idx_subset):
    old_pred = np.asarray(old_pred, dtype=np.int64)
    new_pred = np.asarray(new_pred, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    idx_subset = np.asarray(idx_subset, dtype=np.int64)

    changed = idx_subset[old_pred[idx_subset] != new_pred[idx_subset]]
    fixed = idx_subset[(old_pred[idx_subset] != y[idx_subset]) & (new_pred[idx_subset] == y[idx_subset])]
    broken = idx_subset[(old_pred[idx_subset] == y[idx_subset]) & (new_pred[idx_subset] != y[idx_subset])]
    redirect = idx_subset[
        (old_pred[idx_subset] != y[idx_subset]) &
        (new_pred[idx_subset] != y[idx_subset]) &
        (old_pred[idx_subset] != new_pred[idx_subset])
    ]
    return {
        'changed': changed,
        'fixed': fixed,
        'broken': broken,
        'redirect': redirect,
    }


# ============================================================
# Main audit
# ============================================================

def run_audit(mod, dataset: str, data_base: Path, out_dir: Path, dim_candidates: List[int], num_splits: int):
    raw_root, F, y, A_sym, original_ids = build_features(mod, data_base)

    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

    branch_kwargs = dict(
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
        anchor_top_k=2,
        pair_top_k=2,
        pair_confusion_min=12,
        base_eta_basis=0.10,
        eta_center_ratio=0.10,
        anchor_move_scale=0.25,
        max_correct_harm_rate=0.03,
        max_correct_residual_increase_ratio=0.08,
        pair_score_min=0.60,
    )

    split_rows = []
    node_rows = []
    misclf_rows = []
    branch_summary_rows = []
    branch_member_rows = []
    oof_rows = []
    calib_class_rows = []
    calib_pair_rows = []
    calibration_rewrite_rows = []
    extra_branch_effect_rows = []
    extra_branch_rewrite_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        class_to_pos = {int(c): pos for pos, c in enumerate(classes)}

        best_val = -1.0
        best_dim = -1
        best_test = -1.0
        best_models = None
        best_meta = None
        best_tensor = None

        for dim in dim_candidates:
            branch_models, meta = mod.fit_geometry_coupled_safe_branches(F, y, train_idx, classes, max_dim=dim, **branch_kwargs)
            tensor = get_branch_component_tensor(mod, F, branch_models, classes)
            pred = classes[np.argmin(tensor['best_score'], axis=1)]

            v = float(np.mean(pred[val_idx] == y[val_idx]))
            t = float(np.mean(pred[test_idx] == y[test_idx]))
            if v > best_val:
                best_val = v
                best_dim = int(dim)
                best_test = t
                best_models = branch_models
                best_meta = meta
                best_tensor = tensor

        full_pred = classes[np.argmin(best_tensor['best_score'], axis=1)]

        # Root-only predictions: initial vs calibrated
        initial_roots = best_meta['root_subspaces_initial']
        calibrated_roots = best_meta['root_subspaces_calibrated']
        init_root_pred, init_root_R = root_prediction_from_subspaces(mod, F, initial_roots, classes)
        calib_root_pred, calib_root_R = root_prediction_from_subspaces(mod, F, calibrated_roots, classes)

        total_branches = sum(len(best_models[int(c)]) for c in classes)
        extra_branches = sum(sum(1 for b in best_models[int(c)] if b.get('kind') == 'extra') for c in classes)

        rewrite_init_to_calib_val = summarize_rewrite(init_root_pred, calib_root_pred, y, val_idx)
        rewrite_init_to_calib_test = summarize_rewrite(init_root_pred, calib_root_pred, y, test_idx)
        rewrite_calib_to_full_val = summarize_rewrite(calib_root_pred, full_pred, y, val_idx)
        rewrite_calib_to_full_test = summarize_rewrite(calib_root_pred, full_pred, y, test_idx)

        split_rows.append({
            'split': int(split),
            'best_dim': int(best_dim),
            'val_acc_full': float(np.mean(full_pred[val_idx] == y[val_idx])),
            'test_acc_full': float(np.mean(full_pred[test_idx] == y[test_idx])),
            'val_acc_root_initial': float(np.mean(init_root_pred[val_idx] == y[val_idx])),
            'test_acc_root_initial': float(np.mean(init_root_pred[test_idx] == y[test_idx])),
            'val_acc_root_calibrated': float(np.mean(calib_root_pred[val_idx] == y[val_idx])),
            'test_acc_root_calibrated': float(np.mean(calib_root_pred[test_idx] == y[test_idx])),
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
            'anchor_classes_json': json.dumps([int(x) for x in best_meta['calibration_meta']['anchor_classes']], ensure_ascii=False),
            'val_fixed_by_calibration': int(len(rewrite_init_to_calib_val['fixed'])),
            'val_broken_by_calibration': int(len(rewrite_init_to_calib_val['broken'])),
            'test_fixed_by_calibration': int(len(rewrite_init_to_calib_test['fixed'])),
            'test_broken_by_calibration': int(len(rewrite_init_to_calib_test['broken'])),
            'val_fixed_by_extra': int(len(rewrite_calib_to_full_val['fixed'])),
            'val_broken_by_extra': int(len(rewrite_calib_to_full_val['broken'])),
            'test_fixed_by_extra': int(len(rewrite_calib_to_full_test['fixed'])),
            'test_broken_by_extra': int(len(rewrite_calib_to_full_test['broken'])),
        })

        print(
            f"split={split:2d}  best_dim={best_dim:3d}  "
            f"val={np.mean(full_pred[val_idx] == y[val_idx]):.4f}  "
            f"test={np.mean(full_pred[test_idx] == y[test_idx]):.4f}  "
            f"branches={total_branches} (extra={extra_branches})"
        )

        true_exp_1, true_exp_2, train_exp_1, train_exp_2 = compute_exposure_features(mod, A_sym, y, train_idx, classes)

        # calibration debug rows
        for row in best_meta['calibration_meta']['class_debug']:
            calib_class_rows.append({'split': int(split), **row})
        for row in best_meta['calibration_meta']['pair_debug']:
            calib_pair_rows.append({'split': int(split), **row})
        for true_c, conf_map in best_meta['confusion_counts'].items():
            for pred_c, cnt in conf_map.items():
                oof_rows.append({
                    'split': int(split),
                    'kind': 'oof_confusion',
                    'true_class': int(true_c),
                    'other_class': int(pred_c),
                    'count': int(cnt),
                })
        for c, rows_c in best_meta['branch_debug'].items():
            for idx_dbg, dbg in enumerate(rows_c):
                oof_rows.append({
                    'split': int(split),
                    'kind': 'branch_debug',
                    'class_label': int(c),
                    'branch_debug_index': int(idx_dbg),
                    'confuser': int(dbg.get('confuser', -1)),
                    'seed_size': int(dbg.get('seed_size', -1)),
                    'group_size': int(dbg.get('group_size', -1)),
                    'median_gain': float(dbg.get('median_gain', np.nan)),
                })

        # calibration rewrite node rows (test only)
        for node_idx in rewrite_init_to_calib_test['changed']:
            calibration_rewrite_rows.append({
                'split': int(split),
                'node_idx_sorted': int(node_idx),
                'original_node_id': int(original_ids[node_idx]),
                'true_label': int(y[node_idx]),
                'initial_root_pred': int(init_root_pred[node_idx]),
                'calibrated_root_pred': int(calib_root_pred[node_idx]),
                'change_kind': (
                    'fixed' if node_idx in set(rewrite_init_to_calib_test['fixed'].tolist())
                    else 'broken' if node_idx in set(rewrite_init_to_calib_test['broken'].tolist())
                    else 'redirect'
                ),
            })

        # branch summary rows + members
        for c in classes:
            for b_idx, b in enumerate(best_models[int(c)]):
                member_idx = np.asarray(b.get('member_idx', []), dtype=np.int64)
                confuser = b.get('confuser_class', None)
                seed_size = b.get('seed_size', None)
                median_gain = b.get('median_gain', None)
                branch_summary_rows.append({
                    'split': int(split),
                    'best_dim': int(best_dim),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': str(b.get('kind', '')),
                    'confuser_class': '' if confuser is None else int(confuser),
                    'basis_dim': int(np.asarray(b.get('basis')).shape[1]),
                    'member_size': int(member_idx.size),
                    'radius': float(b.get('radius', np.nan)),
                    'class_scale': float(b.get('class_scale', np.nan)),
                    'extra_bias_scale': float(b.get('extra_bias_scale', np.nan)),
                    'gate_strength': float(b.get('gate_strength', np.nan)),
                    'seed_size': '' if seed_size is None else int(seed_size),
                    'median_gain': '' if median_gain is None else float(median_gain),
                    'pair_center_l2_distance': '' if 'pair_center_l2_distance' not in b else float(b['pair_center_l2_distance']),
                    'pair_min_principal_angle_deg': '' if 'pair_min_principal_angle_deg' not in b else float(b['pair_min_principal_angle_deg']),
                    'pair_coupled_confusion_score': '' if 'pair_coupled_confusion_score' not in b else float(b['pair_coupled_confusion_score']),
                })
                for midx in member_idx:
                    branch_member_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'kind': str(b.get('kind', '')),
                        'confuser_class': '' if confuser is None else int(confuser),
                        'node_idx_sorted': int(midx),
                        'original_node_id': int(original_ids[midx]),
                    })

        # node-level rows on test set
        for node_idx in test_idx:
            true_label = int(y[node_idx])
            pred_label = int(full_pred[node_idx])
            true_pos = class_to_pos[true_label]
            pred_pos = class_to_pos[pred_label]

            pred_branch_idx = int(best_tensor['best_branch_idx'][node_idx, pred_pos])
            true_branch_idx = int(best_tensor['best_branch_idx'][node_idx, true_pos])
            pred_branch = best_models[pred_label][pred_branch_idx]
            true_branch = best_models[true_label][true_branch_idx]

            pred_comp = best_tensor['per_branch_components'][pred_label][pred_branch_idx]
            true_comp = best_tensor['per_branch_components'][true_label][true_branch_idx]

            pred_score = float(pred_comp['score'][node_idx])
            true_score = float(true_comp['score'][node_idx])

            score_row = best_tensor['best_score'][node_idx].copy()
            order = np.argsort(score_row)
            top1_score = float(score_row[order[0]])
            top2_score = float(score_row[order[1]]) if len(order) > 1 else np.nan
            top2_gap = float(top2_score - top1_score) if np.isfinite(top2_score) else np.nan

            # selected branch pair geometry
            center_l2, overlap_fro, min_angle_rad, singular_vals = mod.principal_geometry(
                true_branch['mu'], true_branch['basis'], pred_branch['mu'], pred_branch['basis']
            )

            delta_total = float(pred_comp['total'][node_idx] - true_comp['total'][node_idx])
            delta_evidence = float(pred_comp['evidence'][node_idx] - true_comp['evidence'][node_idx])
            delta_residual = float(pred_comp['residual'][node_idx] - true_comp['residual'][node_idx])
            pred_ratio = float(pred_comp['evidence'][node_idx] / max(pred_comp['total'][node_idx], 1e-12))
            true_ratio = float(true_comp['evidence'][node_idx] / max(true_comp['total'][node_idx], 1e-12))
            delta_ratio = float(pred_ratio - true_ratio)

            node_row = {
                'split': int(split),
                'best_dim': int(best_dim),
                'node_idx_sorted': int(node_idx),
                'original_node_id': int(original_ids[node_idx]),
                'true_label': true_label,
                'adaptive_pred_label': pred_label,
                'root_initial_pred_label': int(init_root_pred[node_idx]),
                'root_calibrated_pred_label': int(calib_root_pred[node_idx]),
                'is_correct_full': int(pred_label == true_label),
                'adaptive_top1_score': top1_score,
                'adaptive_top2_score': top2_score,
                'adaptive_top2_gap': top2_gap,
                'adaptive_pred_score': pred_score,
                'adaptive_true_score': true_score,
                'adaptive_pred_total_sq': float(pred_comp['total'][node_idx]),
                'adaptive_true_total_sq': float(true_comp['total'][node_idx]),
                'adaptive_pred_evidence_sq': float(pred_comp['evidence'][node_idx]),
                'adaptive_true_evidence_sq': float(true_comp['evidence'][node_idx]),
                'adaptive_pred_residual': float(pred_comp['residual'][node_idx]),
                'adaptive_true_residual': float(true_comp['residual'][node_idx]),
                'delta_total_pred_minus_true': delta_total,
                'delta_evidence_pred_minus_true': delta_evidence,
                'delta_residual_pred_minus_true': delta_residual,
                'pred_explained_ratio': pred_ratio,
                'true_explained_ratio': true_ratio,
                'delta_explained_ratio_pred_minus_true': delta_ratio,
                'pred_branch_index': pred_branch_idx,
                'pred_branch_kind': str(pred_branch.get('kind', '')),
                'pred_branch_confuser_class': '' if pred_branch.get('confuser_class', None) is None else int(pred_branch['confuser_class']),
                'pred_branch_member_size': int(np.asarray(pred_branch.get('member_idx', [])).size),
                'pred_branch_basis_dim': int(np.asarray(pred_branch.get('basis')).shape[1]),
                'pred_branch_radius': float(pred_branch.get('radius', np.nan)),
                'pred_branch_class_scale': float(pred_branch.get('class_scale', np.nan)),
                'pred_branch_extra_bias_scale': float(pred_branch.get('extra_bias_scale', np.nan)),
                'pred_branch_gate_strength': float(pred_branch.get('gate_strength', np.nan)),
                'pred_branch_bias_penalty': float(pred_comp['bias_penalty'][node_idx]),
                'pred_branch_gate_penalty': float(pred_comp['gate_penalty'][node_idx]),
                'pred_branch_sqdist_to_center': float(pred_comp['sqdist_to_center'][node_idx]),
                'pred_branch_normalized_center_dist': float(pred_comp['normalized_center_dist'][node_idx]),
                'true_branch_index': true_branch_idx,
                'true_branch_kind': str(true_branch.get('kind', '')),
                'true_branch_confuser_class': '' if true_branch.get('confuser_class', None) is None else int(true_branch['confuser_class']),
                'true_branch_member_size': int(np.asarray(true_branch.get('member_idx', [])).size),
                'true_branch_basis_dim': int(np.asarray(true_branch.get('basis')).shape[1]),
                'true_branch_radius': float(true_branch.get('radius', np.nan)),
                'true_branch_class_scale': float(true_branch.get('class_scale', np.nan)),
                'true_branch_bias_penalty': float(true_comp['bias_penalty'][node_idx]),
                'true_branch_gate_penalty': float(true_comp['gate_penalty'][node_idx]),
                'true_branch_sqdist_to_center': float(true_comp['sqdist_to_center'][node_idx]),
                'true_branch_normalized_center_dist': float(true_comp['normalized_center_dist'][node_idx]),
                'true_pred_branch_center_l2_distance': center_l2,
                'true_pred_branch_subspace_overlap_fro': overlap_fro,
                'true_pred_branch_min_principal_angle_deg': float(np.degrees(min_angle_rad)),
                'true_pred_branch_principal_cosines_json': json.dumps([float(x) for x in singular_vals.tolist()], ensure_ascii=False),
                'train_exposure_true_1hop': float(train_exp_1[node_idx, true_pos]),
                'train_exposure_true_2hop': float(train_exp_2[node_idx, true_pos]),
                'train_exposure_pred_1hop': float(train_exp_1[node_idx, pred_pos]),
                'train_exposure_pred_2hop': float(train_exp_2[node_idx, pred_pos]),
                'true_exposure_true_1hop': float(true_exp_1[node_idx, true_pos]),
                'true_exposure_true_2hop': float(true_exp_2[node_idx, true_pos]),
                'true_exposure_pred_1hop': float(true_exp_1[node_idx, pred_pos]),
                'true_exposure_pred_2hop': float(true_exp_2[node_idx, pred_pos]),
            }
            node_rows.append(node_row)
            if true_label != pred_label:
                misclf_rows.append(node_row)

        # Per-extra-branch marginal effect
        for c in classes:
            branches_c = best_models[int(c)]
            for b_idx, b in enumerate(branches_c):
                if str(b.get('kind', '')) != 'extra':
                    continue

                branch_models_removed = deepcopy(best_models)
                branch_models_removed[int(c)] = [
                    branch for idx_branch, branch in enumerate(branch_models_removed[int(c)])
                    if idx_branch != b_idx
                ]
                tensor_removed = get_branch_component_tensor(mod, F, branch_models_removed, classes)
                pred_removed = classes[np.argmin(tensor_removed['best_score'], axis=1)]

                rewrite_val = summarize_rewrite(pred_removed, full_pred, y, val_idx)
                rewrite_test = summarize_rewrite(pred_removed, full_pred, y, test_idx)

                extra_branch_effect_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'confuser_class': '' if b.get('confuser_class', None) is None else int(b['confuser_class']),
                    'seed_size': '' if b.get('seed_size', None) is None else int(b['seed_size']),
                    'member_size': int(np.asarray(b.get('member_idx', [])).size),
                    'basis_dim': int(np.asarray(b.get('basis')).shape[1]),
                    'radius': float(b.get('radius', np.nan)),
                    'median_gain': '' if b.get('median_gain', None) is None else float(b['median_gain']),
                    'pair_center_l2_distance': '' if 'pair_center_l2_distance' not in b else float(b['pair_center_l2_distance']),
                    'pair_min_principal_angle_deg': '' if 'pair_min_principal_angle_deg' not in b else float(b['pair_min_principal_angle_deg']),
                    'pair_coupled_confusion_score': '' if 'pair_coupled_confusion_score' not in b else float(b['pair_coupled_confusion_score']),
                    'val_fixed_by_this_branch': int(len(rewrite_val['fixed'])),
                    'val_broken_by_this_branch': int(len(rewrite_val['broken'])),
                    'val_wrong_redirect_by_this_branch': int(len(rewrite_val['redirect'])),
                    'test_fixed_by_this_branch': int(len(rewrite_test['fixed'])),
                    'test_broken_by_this_branch': int(len(rewrite_test['broken'])),
                    'test_wrong_redirect_by_this_branch': int(len(rewrite_test['redirect'])),
                    'changed_when_removed_test': int(len(rewrite_test['changed'])),
                    'net_fix_test': int(len(rewrite_test['fixed']) - len(rewrite_test['broken'])),
                })

                fixed_set = set(rewrite_test['fixed'].tolist())
                broken_set = set(rewrite_test['broken'].tolist())
                redirect_set = set(rewrite_test['redirect'].tolist())
                for node_idx in rewrite_test['changed']:
                    extra_branch_rewrite_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'confuser_class': '' if b.get('confuser_class', None) is None else int(b['confuser_class']),
                        'node_idx_sorted': int(node_idx),
                        'original_node_id': int(original_ids[node_idx]),
                        'true_label': int(y[node_idx]),
                        'pred_without_branch': int(pred_removed[node_idx]),
                        'pred_with_full_model': int(full_pred[node_idx]),
                        'change_kind': (
                            'fixed' if int(node_idx) in fixed_set
                            else 'broken' if int(node_idx) in broken_set
                            else 'redirect'
                        ),
                    })

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'split_summary_src_v11_geometry_safe_branch_audit.csv', split_rows)
    write_csv(out_dir / 'test_node_predictions_all_src_v11.csv', node_rows)
    write_csv(out_dir / 'misclassified_test_nodes_src_v11.csv', misclf_rows)
    write_csv(out_dir / 'branch_subspace_summary_src_v11.csv', branch_summary_rows)
    write_csv(out_dir / 'branch_member_rows_src_v11.csv', branch_member_rows)
    write_csv(out_dir / 'oof_confusion_and_branch_debug_src_v11.csv', oof_rows)
    write_csv(out_dir / 'calibration_class_debug_src_v11.csv', calib_class_rows)
    write_csv(out_dir / 'calibration_pair_debug_src_v11.csv', calib_pair_rows)
    write_csv(out_dir / 'root_calibration_rewrite_nodes_src_v11.csv', calibration_rewrite_rows)
    write_csv(out_dir / 'extra_branch_effect_summary_src_v11.csv', extra_branch_effect_rows)
    write_csv(out_dir / 'extra_branch_rewrite_nodes_src_v11.csv', extra_branch_rewrite_rows)

    vals = [r['val_acc_full'] for r in split_rows]
    tests = [r['test_acc_full'] for r in split_rows]
    summary = {
        'dataset': dataset,
        'data_base': str(data_base),
        'raw_root': str(raw_root),
        'out_dir': str(out_dir),
        'dim_candidates': [int(x) for x in dim_candidates],
        'num_splits': int(num_splits),
        'val_mean': float(np.mean(vals)),
        'val_std': float(np.std(vals)),
        'test_mean': float(np.mean(tests)),
        'test_std': float(np.std(tests)),
    }
    (out_dir / 'run_summary_src_v11_geometry_safe_branch_audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print()
    print(f"val_mean  = {summary['val_mean']:.4f} +- {summary['val_std']:.4f}")
    print(f"test_mean = {summary['test_mean']:.4f} +- {summary['test_std']:.4f}")
    print("\nWrote:")
    for name in [
        'misclassified_test_nodes_src_v11.csv',
        'branch_subspace_summary_src_v11.csv',
        'calibration_class_debug_src_v11.csv',
        'calibration_pair_debug_src_v11.csv',
        'extra_branch_effect_summary_src_v11.csv',
        'extra_branch_rewrite_nodes_src_v11.csv',
    ]:
        print(f"  {out_dir / name}")


def main():
    parser = argparse.ArgumentParser(description='Run src_v11 geometry-safe branch model with full audit.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None,
                        help='Path like D:/planetoid/data/chameleon . Default: auto-discover.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory. Default: project/scripts/results_src_v11_geometry_safe_branch_audit_<dataset>')
    parser.add_argument('--src-v11-file', type=str, default=None,
                        help='Path to src_v11 algorithm file. Default: auto-discover under project/src_v11')
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    project_root, drive_root, default_data_base, default_out_dir, src_v11_path = resolve_default_paths(args.dataset, args.src_v11_file)
    data_base = Path(args.data_base) if args.data_base is not None else default_data_base
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir
    mod = import_module_from_path(src_v11_path, 'src_v11_algo_audit')
    run_audit(mod, dataset=args.dataset, data_base=data_base, out_dir=out_dir,
              dim_candidates=[int(x) for x in args.dims], num_splits=int(args.num_splits))


if __name__ == '__main__':
    main()


