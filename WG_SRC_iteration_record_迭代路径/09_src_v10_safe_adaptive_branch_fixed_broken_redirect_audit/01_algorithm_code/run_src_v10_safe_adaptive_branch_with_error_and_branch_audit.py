#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run src_v10 safe adaptive-branch PCA on Chameleon and record:
1) node-level test errors with related features
2) split-level branch-vs-root impact
3) per-extra-branch marginal impact (fix / break counts)
4) branch metadata, members, and OOF diagnostics used to create branches

Place under project/scripts and run:
    python run_src_v10_safe_adaptive_branch_with_error_and_branch_audit.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse


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
        if (p / 'scripts').exists() and (p / 'src_v10').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts and src_v10')


def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError('Cannot locate drive root containing planetoid/data')


def find_first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def json_int_list(arr) -> str:
    return json.dumps([int(x) for x in np.asarray(arr, dtype=np.int64).tolist()], ensure_ascii=False)


def load_original_node_ids(raw_root: Path) -> np.ndarray:
    node_file = Path(raw_root) / 'out1_node_feature_label.txt'
    ids = []
    with open(node_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            node_id, feat, label = line.rstrip('\n').split('\t')
            ids.append(int(node_id))
    order = np.argsort(np.asarray(ids, dtype=np.int64))
    return np.asarray(ids, dtype=np.int64)[order]


# ============================================================
# Shared feature / exposure helpers
# ============================================================

def build_features(mod, data_base: Path):
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
    original_ids = load_original_node_ids(raw_root)
    return raw_root, F, y, A_sym, original_ids


def compute_exposure_features(mod, A_sym, y, train_idx, classes):
    P = mod.row_normalize(A_sym)
    n = A_sym.shape[0]
    cnum = len(classes)
    train_exp_1 = np.zeros((n, cnum), dtype=np.float64)
    train_exp_2 = np.zeros((n, cnum), dtype=np.float64)
    true_exp_1 = np.zeros((n, cnum), dtype=np.float64)
    true_exp_2 = np.zeros((n, cnum), dtype=np.float64)
    for pos, c in enumerate(classes):
        mask_true = (y == c).astype(np.float64)
        mask_train = np.zeros(n, dtype=np.float64)
        idx_c = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        mask_train[idx_c] = 1.0
        true_exp_1[:, pos] = np.asarray(P @ mask_true).ravel()
        true_exp_2[:, pos] = np.asarray(P @ true_exp_1[:, pos]).ravel()
        train_exp_1[:, pos] = np.asarray(P @ mask_train).ravel()
        train_exp_2[:, pos] = np.asarray(P @ train_exp_1[:, pos]).ravel()
    return true_exp_1, true_exp_2, train_exp_1, train_exp_2


# ============================================================
# Score decomposition helpers
# ============================================================

def top2_stats(score_mat: np.ndarray):
    order = np.argsort(score_mat, axis=1)
    top1_pos = order[:, 0]
    top2_pos = order[:, 1]
    top1 = score_mat[np.arange(score_mat.shape[0]), top1_pos]
    top2 = score_mat[np.arange(score_mat.shape[0]), top2_pos]
    gap = top2 - top1
    return top1_pos, top2_pos, top1, top2, gap


def branch_score_components(mod, F: np.ndarray, branch: dict):
    total, evidence, residual = mod.subspace_stats(F, branch['mu'], branch['basis'])
    if branch['kind'] == 'root':
        sqdist = np.sum((F - branch['center'][None, :]) ** 2, axis=1)
        zero = np.zeros(F.shape[0], dtype=np.float64)
        return {
            'total_sq': total,
            'evidence_sq': evidence,
            'residual': residual,
            'sqdist_to_center': sqdist,
            'normalized_center_dist': sqdist / np.maximum(branch.get('radius', 1.0), 1e-8),
            'bias_penalty': zero,
            'gate_penalty': zero,
            'final_score': residual,
        }

    sqdist = np.sum((F - branch['center'][None, :]) ** 2, axis=1)
    normalized = sqdist / max(branch['radius'], 1e-8)
    gate_penalty = branch['gate_strength'] * branch['class_scale'] * np.maximum(normalized - 1.0, 0.0) ** 2
    bias_penalty = np.full(F.shape[0], branch['extra_bias_scale'] * branch['class_scale'], dtype=np.float64)
    final_score = residual + bias_penalty + gate_penalty
    return {
        'total_sq': total,
        'evidence_sq': evidence,
        'residual': residual,
        'sqdist_to_center': sqdist,
        'normalized_center_dist': normalized,
        'bias_penalty': bias_penalty,
        'gate_penalty': gate_penalty,
        'final_score': final_score,
    }


def adaptive_score_details(mod, F: np.ndarray, branch_models: Dict[int, List[dict]], classes: np.ndarray):
    n = F.shape[0]
    cnum = len(classes)
    best_score = np.full((n, cnum), np.inf, dtype=np.float64)
    best_branch_index = np.full((n, cnum), -1, dtype=np.int64)
    best_kind = np.empty((n, cnum), dtype=object)
    best_confuser = np.full((n, cnum), -1, dtype=np.int64)
    best_residual = np.full((n, cnum), np.nan, dtype=np.float64)
    best_total_sq = np.full((n, cnum), np.nan, dtype=np.float64)
    best_evidence_sq = np.full((n, cnum), np.nan, dtype=np.float64)
    best_bias_penalty = np.full((n, cnum), np.nan, dtype=np.float64)
    best_gate_penalty = np.full((n, cnum), np.nan, dtype=np.float64)
    best_sqdist = np.full((n, cnum), np.nan, dtype=np.float64)
    best_norm_center = np.full((n, cnum), np.nan, dtype=np.float64)

    for pos, c in enumerate(classes):
        for b_idx, branch in enumerate(branch_models[int(c)]):
            comp = branch_score_components(mod, F, branch)
            s = comp['final_score']
            better = s < best_score[:, pos]
            best_score[better, pos] = s[better]
            best_branch_index[better, pos] = int(b_idx)
            best_kind[better, pos] = branch['kind']
            best_confuser[better, pos] = int(branch.get('confuser_class', -1))
            best_residual[better, pos] = comp['residual'][better]
            best_total_sq[better, pos] = comp['total_sq'][better]
            best_evidence_sq[better, pos] = comp['evidence_sq'][better]
            best_bias_penalty[better, pos] = comp['bias_penalty'][better]
            best_gate_penalty[better, pos] = comp['gate_penalty'][better]
            best_sqdist[better, pos] = comp['sqdist_to_center'][better]
            best_norm_center[better, pos] = comp['normalized_center_dist'][better]

    return {
        'score': best_score,
        'branch_index': best_branch_index,
        'kind': best_kind,
        'confuser': best_confuser,
        'residual': best_residual,
        'total_sq': best_total_sq,
        'evidence_sq': best_evidence_sq,
        'bias_penalty': best_bias_penalty,
        'gate_penalty': best_gate_penalty,
        'sqdist_to_center': best_sqdist,
        'normalized_center_dist': best_norm_center,
    }


def root_score_details(mod, F: np.ndarray, root_subspaces: Dict[int, dict], classes: np.ndarray):
    n = F.shape[0]
    cnum = len(classes)
    residual = np.full((n, cnum), np.nan, dtype=np.float64)
    total_sq = np.full((n, cnum), np.nan, dtype=np.float64)
    evidence_sq = np.full((n, cnum), np.nan, dtype=np.float64)
    for pos, c in enumerate(classes):
        total, evidence, disc = mod.subspace_stats(F, root_subspaces[int(c)]['mu'], root_subspaces[int(c)]['basis'])
        residual[:, pos] = disc
        total_sq[:, pos] = total
        evidence_sq[:, pos] = evidence
    return {
        'score': residual,
        'residual': residual,
        'total_sq': total_sq,
        'evidence_sq': evidence_sq,
    }


# ============================================================
# Impact accounting
# ============================================================

def compare_predictions(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, mask: np.ndarray):
    y = y_true[mask]
    a = pred_a[mask]
    b = pred_b[mask]
    a_ok = (a == y)
    b_ok = (b == y)
    changed = (a != b)
    fixed = (~a_ok) & b_ok
    broken = a_ok & (~b_ok)
    wrong_redirect = (~a_ok) & (~b_ok) & changed
    return {
        'n': int(mask.sum()),
        'changed': int(np.sum(changed)),
        'fixed': int(np.sum(fixed)),
        'broken': int(np.sum(broken)),
        'wrong_redirect': int(np.sum(wrong_redirect)),
    }


# ============================================================
# Main audit routine
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
    )

    split_rows = []
    node_rows = []
    misclf_rows = []
    branch_summary_rows = []
    branch_member_rows = []
    oof_rows = []
    branch_effect_rows = []
    branch_rewrite_rows = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        class_to_pos = {int(c): pos for pos, c in enumerate(classes)}
        true_exp_1, true_exp_2, train_exp_1, train_exp_2 = compute_exposure_features(mod, A_sym, y, train_idx, classes)

        best_val = -1.0
        best_dim = None
        best_test = None
        best_branch_models = None
        best_meta = None
        best_adaptive_detail = None
        best_adaptive_pred = None
        best_adaptive_top = None

        # Choose dim exactly by full adaptive val accuracy.
        for dim in dim_candidates:
            branch_models, meta = mod.fit_conservative_adaptive_branches(
                F, y, train_idx, classes, max_dim=int(dim), **branch_kwargs
            )
            adaptive_detail = adaptive_score_details(mod, F, branch_models, classes)
            pred = classes[np.argmin(adaptive_detail['score'], axis=1)]
            _, _, top1, top2, gap = top2_stats(adaptive_detail['score'])
            val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
            test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
            if val_acc > best_val:
                best_val = val_acc
                best_dim = int(dim)
                best_test = test_acc
                best_branch_models = branch_models
                best_meta = meta
                best_adaptive_detail = adaptive_detail
                best_adaptive_pred = pred
                best_adaptive_top = (top1, top2, gap)

        # Root-only model with the same chosen dim.
        root_subspaces = mod.fit_root_subspaces(F, y, train_idx, classes, max_dim=int(best_dim))
        root_detail = root_score_details(mod, F, root_subspaces, classes)
        root_pred = classes[np.argmin(root_detail['score'], axis=1)]
        _, _, root_top1, root_top2, root_gap = top2_stats(root_detail['score'])
        _, _, adaptive_top1, adaptive_top2, adaptive_gap = top2_stats(best_adaptive_detail['score'])

        total_branches = 0
        extra_branches = 0
        for c in classes:
            branches = best_branch_models[int(c)]
            total_branches += len(branches)
            extra_branches += sum(1 for b in branches if b['kind'] == 'extra')

        val_root_vs_full = compare_predictions(y, root_pred, best_adaptive_pred, np.isin(np.arange(F.shape[0]), val_idx))
        test_root_vs_full = compare_predictions(y, root_pred, best_adaptive_pred, np.isin(np.arange(F.shape[0]), test_idx))

        split_rows.append({
            'split': int(split),
            'best_dim': int(best_dim),
            'val_acc_full': float(best_val),
            'test_acc_full': float(best_test),
            'val_acc_root_same_dim': float(np.mean(root_pred[val_idx] == y[val_idx])),
            'test_acc_root_same_dim': float(np.mean(root_pred[test_idx] == y[test_idx])),
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
            'val_changed_vs_root': int(val_root_vs_full['changed']),
            'val_fixed_vs_root': int(val_root_vs_full['fixed']),
            'val_broken_vs_root': int(val_root_vs_full['broken']),
            'val_wrong_redirect_vs_root': int(val_root_vs_full['wrong_redirect']),
            'test_changed_vs_root': int(test_root_vs_full['changed']),
            'test_fixed_vs_root': int(test_root_vs_full['fixed']),
            'test_broken_vs_root': int(test_root_vs_full['broken']),
            'test_wrong_redirect_vs_root': int(test_root_vs_full['wrong_redirect']),
        })

        print(
            f'split={split:2d}  best_dim={best_dim:3d}  '
            f'val={best_val:.4f}  test={best_test:.4f}  '
            f'branches={total_branches} (extra={extra_branches})'
        )

        # Record OOF diagnostics used to drive branches.
        conf_counts = best_meta.get('confusion_counts', {}) if best_meta else {}
        branch_debug = best_meta.get('branch_debug', {}) if best_meta else {}
        class_scales = best_meta.get('class_scales', {}) if best_meta else {}
        for true_c, conf_map in conf_counts.items():
            for pred_c, cnt in conf_map.items():
                oof_rows.append({
                    'split': int(split),
                    'record_kind': 'oof_confusion',
                    'class_label': int(true_c),
                    'other_class': int(pred_c),
                    'count': int(cnt),
                    'class_scale': float(class_scales.get(int(true_c), np.nan)),
                })
        for c, dbg_list in branch_debug.items():
            for j, dbg in enumerate(dbg_list):
                oof_rows.append({
                    'split': int(split),
                    'record_kind': 'branch_debug',
                    'class_label': int(c),
                    'other_class': int(dbg.get('confuser', -1)),
                    'count': int(dbg.get('group_size', -1)),
                    'class_scale': float(class_scales.get(int(c), np.nan)),
                    'seed_size': int(dbg.get('seed_size', -1)),
                    'median_gain': float(dbg.get('median_gain', np.nan)),
                    'branch_debug_index': int(j),
                })

        # Branch summary / members.
        for c in classes:
            branches = best_branch_models[int(c)]
            for b_idx, branch in enumerate(branches):
                member_idx = np.asarray(branch.get('member_idx', []), dtype=np.int64)
                kind = str(branch.get('kind', ''))
                branch_summary_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': kind,
                    'confuser_class': int(branch.get('confuser_class', -1)),
                    'basis_dim': int(branch['basis'].shape[1]),
                    'member_size': int(member_idx.size),
                    'seed_size': int(branch.get('seed_size', member_idx.size if kind == 'root' else -1)),
                    'class_scale': float(branch.get('class_scale', np.nan)),
                    'radius': float(branch.get('radius', np.nan)),
                    'extra_bias_scale': float(branch.get('extra_bias_scale', np.nan)),
                    'gate_strength': float(branch.get('gate_strength', np.nan)),
                    'median_gain': float(branch.get('median_gain', np.nan)),
                    'member_idx_json': json_int_list(member_idx),
                    'member_original_id_json': json.dumps([int(original_ids[i]) for i in member_idx.tolist()], ensure_ascii=False),
                })
                for node_idx in member_idx.tolist():
                    branch_member_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'kind': kind,
                        'confuser_class': int(branch.get('confuser_class', -1)),
                        'node_idx_sorted': int(node_idx),
                        'original_node_id': int(original_ids[node_idx]),
                    })

        # Per-node test records.
        for node_idx in test_idx:
            true_label = int(y[node_idx])
            root_pred_label = int(root_pred[node_idx])
            adaptive_pred_label = int(best_adaptive_pred[node_idx])
            true_pos = class_to_pos[true_label]
            root_pred_pos = class_to_pos[root_pred_label]
            adaptive_pred_pos = class_to_pos[adaptive_pred_label]

            adaptive_pred_branch_index = int(best_adaptive_detail['branch_index'][node_idx, adaptive_pred_pos])
            adaptive_true_branch_index = int(best_adaptive_detail['branch_index'][node_idx, true_pos])
            adaptive_pred_branch = best_branch_models[adaptive_pred_label][adaptive_pred_branch_index]
            adaptive_true_branch = best_branch_models[true_label][adaptive_true_branch_index]

            row = {
                'split': int(split),
                'best_dim': int(best_dim),
                'node_idx_sorted': int(node_idx),
                'original_node_id': int(original_ids[node_idx]),
                'true_label': true_label,
                'root_pred_label': root_pred_label,
                'adaptive_pred_label': adaptive_pred_label,
                'root_correct': int(root_pred_label == true_label),
                'adaptive_correct': int(adaptive_pred_label == true_label),
                'changed_vs_root': int(root_pred_label != adaptive_pred_label),
                'fixed_vs_root': int((root_pred_label != true_label) and (adaptive_pred_label == true_label)),
                'broken_vs_root': int((root_pred_label == true_label) and (adaptive_pred_label != true_label)),
                'root_top1_score': float(root_top1[node_idx]),
                'root_top2_score': float(root_top2[node_idx]),
                'root_top2_gap': float(root_gap[node_idx]),
                'adaptive_top1_score': float(adaptive_top1[node_idx]),
                'adaptive_top2_score': float(adaptive_top2[node_idx]),
                'adaptive_top2_gap': float(adaptive_gap[node_idx]),
                'root_true_score': float(root_detail['score'][node_idx, true_pos]),
                'root_pred_score': float(root_detail['score'][node_idx, root_pred_pos]),
                'root_true_total_sq': float(root_detail['total_sq'][node_idx, true_pos]),
                'root_true_evidence_sq': float(root_detail['evidence_sq'][node_idx, true_pos]),
                'root_pred_total_sq': float(root_detail['total_sq'][node_idx, root_pred_pos]),
                'root_pred_evidence_sq': float(root_detail['evidence_sq'][node_idx, root_pred_pos]),
                'adaptive_true_score': float(best_adaptive_detail['score'][node_idx, true_pos]),
                'adaptive_pred_score': float(best_adaptive_detail['score'][node_idx, adaptive_pred_pos]),
                'adaptive_true_branch_index': int(adaptive_true_branch_index),
                'adaptive_true_branch_kind': str(best_adaptive_detail['kind'][node_idx, true_pos]),
                'adaptive_true_branch_confuser': int(best_adaptive_detail['confuser'][node_idx, true_pos]),
                'adaptive_true_residual': float(best_adaptive_detail['residual'][node_idx, true_pos]),
                'adaptive_true_total_sq': float(best_adaptive_detail['total_sq'][node_idx, true_pos]),
                'adaptive_true_evidence_sq': float(best_adaptive_detail['evidence_sq'][node_idx, true_pos]),
                'adaptive_true_bias_penalty': float(best_adaptive_detail['bias_penalty'][node_idx, true_pos]),
                'adaptive_true_gate_penalty': float(best_adaptive_detail['gate_penalty'][node_idx, true_pos]),
                'adaptive_true_sqdist_to_center': float(best_adaptive_detail['sqdist_to_center'][node_idx, true_pos]),
                'adaptive_true_normalized_center_dist': float(best_adaptive_detail['normalized_center_dist'][node_idx, true_pos]),
                'adaptive_pred_branch_index': int(adaptive_pred_branch_index),
                'adaptive_pred_branch_kind': str(best_adaptive_detail['kind'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_branch_confuser': int(best_adaptive_detail['confuser'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_residual': float(best_adaptive_detail['residual'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_total_sq': float(best_adaptive_detail['total_sq'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_evidence_sq': float(best_adaptive_detail['evidence_sq'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_bias_penalty': float(best_adaptive_detail['bias_penalty'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_gate_penalty': float(best_adaptive_detail['gate_penalty'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_sqdist_to_center': float(best_adaptive_detail['sqdist_to_center'][node_idx, adaptive_pred_pos]),
                'adaptive_pred_normalized_center_dist': float(best_adaptive_detail['normalized_center_dist'][node_idx, adaptive_pred_pos]),
                'train_exposure_true_1hop': float(train_exp_1[node_idx, true_pos]),
                'train_exposure_true_2hop': float(train_exp_2[node_idx, true_pos]),
                'train_exposure_root_pred_1hop': float(train_exp_1[node_idx, root_pred_pos]),
                'train_exposure_root_pred_2hop': float(train_exp_2[node_idx, root_pred_pos]),
                'train_exposure_adaptive_pred_1hop': float(train_exp_1[node_idx, adaptive_pred_pos]),
                'train_exposure_adaptive_pred_2hop': float(train_exp_2[node_idx, adaptive_pred_pos]),
                'true_exposure_true_1hop': float(true_exp_1[node_idx, true_pos]),
                'true_exposure_true_2hop': float(true_exp_2[node_idx, true_pos]),
                'true_exposure_adaptive_pred_1hop': float(true_exp_1[node_idx, adaptive_pred_pos]),
                'true_exposure_adaptive_pred_2hop': float(true_exp_2[node_idx, adaptive_pred_pos]),
                'pred_branch_member_size': int(np.asarray(adaptive_pred_branch.get('member_idx', []), dtype=np.int64).size),
                'pred_branch_basis_dim': int(adaptive_pred_branch['basis'].shape[1]),
                'true_branch_member_size': int(np.asarray(adaptive_true_branch.get('member_idx', []), dtype=np.int64).size),
                'true_branch_basis_dim': int(adaptive_true_branch['basis'].shape[1]),
            }
            node_rows.append(row)
            if adaptive_pred_label != true_label:
                misclf_rows.append(row)

        # Per-extra-branch marginal effect: remove one extra branch and compare to full adaptive.
        test_mask = np.isin(np.arange(F.shape[0]), test_idx)
        val_mask = np.isin(np.arange(F.shape[0]), val_idx)
        full_pred = best_adaptive_pred
        full_ok = (full_pred == y)
        full_global_pred_pos = np.argmin(best_adaptive_detail['score'], axis=1)
        full_global_branch_index = np.array([
            int(best_adaptive_detail['branch_index'][i, full_global_pred_pos[i]]) for i in range(F.shape[0])
        ], dtype=np.int64)

        for c in classes:
            for b_idx, branch in enumerate(best_branch_models[int(c)]):
                if branch['kind'] != 'extra':
                    continue
                ablated = {int(k): list(v) for k, v in best_branch_models.items()}
                ablated[int(c)] = [b for j, b in enumerate(ablated[int(c)]) if j != b_idx]
                ablated_detail = adaptive_score_details(mod, F, ablated, classes)
                ablated_pred = classes[np.argmin(ablated_detail['score'], axis=1)]
                ablated_ok = (ablated_pred == y)

                val_cmp = compare_predictions(y, ablated_pred, full_pred, val_mask)
                test_cmp = compare_predictions(y, ablated_pred, full_pred, test_mask)

                selected_global = (full_pred == int(c)) & (full_global_branch_index == int(b_idx))
                selected_global_val = selected_global & val_mask
                selected_global_test = selected_global & test_mask

                help_vs_root_selected_test = int(np.sum(selected_global_test & (root_pred != y) & (full_pred == y)))
                hurt_vs_root_selected_test = int(np.sum(selected_global_test & (root_pred == y) & (full_pred != y)))
                help_vs_root_selected_val = int(np.sum(selected_global_val & (root_pred != y) & (full_pred == y)))
                hurt_vs_root_selected_val = int(np.sum(selected_global_val & (root_pred == y) & (full_pred != y)))

                branch_effect_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': 'extra',
                    'confuser_class': int(branch.get('confuser_class', -1)),
                    'basis_dim': int(branch['basis'].shape[1]),
                    'member_size': int(np.asarray(branch.get('member_idx', []), dtype=np.int64).size),
                    'seed_size': int(branch.get('seed_size', -1)),
                    'median_gain': float(branch.get('median_gain', np.nan)),
                    'radius': float(branch.get('radius', np.nan)),
                    'extra_bias_scale': float(branch.get('extra_bias_scale', np.nan)),
                    'gate_strength': float(branch.get('gate_strength', np.nan)),
                    'val_selected_count': int(np.sum(selected_global_val)),
                    'val_selected_correct_count': int(np.sum(selected_global_val & full_ok)),
                    'val_selected_wrong_count': int(np.sum(selected_global_val & (~full_ok))),
                    'val_help_vs_root_selected_count': int(help_vs_root_selected_val),
                    'val_hurt_vs_root_selected_count': int(hurt_vs_root_selected_val),
                    'val_changed_when_removed': int(val_cmp['changed']),
                    'val_fixed_by_this_branch': int(val_cmp['fixed']),
                    'val_broken_by_this_branch': int(val_cmp['broken']),
                    'val_wrong_redirect_by_this_branch': int(val_cmp['wrong_redirect']),
                    'test_selected_count': int(np.sum(selected_global_test)),
                    'test_selected_correct_count': int(np.sum(selected_global_test & full_ok)),
                    'test_selected_wrong_count': int(np.sum(selected_global_test & (~full_ok))),
                    'test_help_vs_root_selected_count': int(help_vs_root_selected_test),
                    'test_hurt_vs_root_selected_count': int(hurt_vs_root_selected_test),
                    'test_changed_when_removed': int(test_cmp['changed']),
                    'test_fixed_by_this_branch': int(test_cmp['fixed']),
                    'test_broken_by_this_branch': int(test_cmp['broken']),
                    'test_wrong_redirect_by_this_branch': int(test_cmp['wrong_redirect']),
                })

                changed_nodes = np.where((full_pred != ablated_pred) & test_mask)[0]
                for node_idx in changed_nodes.tolist():
                    true_label = int(y[node_idx])
                    branch_rewrite_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'confuser_class': int(branch.get('confuser_class', -1)),
                        'node_idx_sorted': int(node_idx),
                        'original_node_id': int(original_ids[node_idx]),
                        'true_label': true_label,
                        'root_pred_label': int(root_pred[node_idx]),
                        'ablated_pred_label': int(ablated_pred[node_idx]),
                        'full_pred_label': int(full_pred[node_idx]),
                        'ablated_correct': int(ablated_ok[node_idx]),
                        'full_correct': int(full_ok[node_idx]),
                        'fixed_by_this_branch': int((not ablated_ok[node_idx]) and full_ok[node_idx]),
                        'broken_by_this_branch': int(ablated_ok[node_idx] and (not full_ok[node_idx])),
                        'wrong_redirect_by_this_branch': int((not ablated_ok[node_idx]) and (not full_ok[node_idx]) and (ablated_pred[node_idx] != full_pred[node_idx])),
                        'full_branch_selected_for_pred': int((full_pred[node_idx] == int(c)) and (full_global_branch_index[node_idx] == int(b_idx))),
                    })

    # Final summaries.
    vals = np.asarray([r['val_acc_full'] for r in split_rows], dtype=np.float64)
    tests = np.asarray([r['test_acc_full'] for r in split_rows], dtype=np.float64)
    run_summary = {
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

    write_csv(out_dir / 'split_summary_src_v10_branch_effect_audit.csv', split_rows)
    write_csv(out_dir / 'test_node_predictions_all.csv', node_rows)
    write_csv(out_dir / 'misclassified_test_nodes_src_v10.csv', misclf_rows)
    write_csv(out_dir / 'branch_subspace_summary_src_v10.csv', branch_summary_rows)
    write_csv(out_dir / 'branch_member_rows_src_v10.csv', branch_member_rows)
    write_csv(out_dir / 'oof_confusion_and_branch_debug_src_v10.csv', oof_rows)
    write_csv(out_dir / 'extra_branch_effect_summary_src_v10.csv', branch_effect_rows)
    write_csv(out_dir / 'extra_branch_rewrite_nodes_src_v10.csv', branch_rewrite_rows)
    (out_dir / 'run_summary_src_v10_branch_effect_audit.json').write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print()
    print(f"val_mean  = {run_summary['val_mean']:.4f} +- {run_summary['val_std']:.4f}")
    print(f"test_mean = {run_summary['test_mean']:.4f} +- {run_summary['test_std']:.4f}")
    print('Wrote:')
    print(f"  {out_dir / 'split_summary_src_v10_branch_effect_audit.csv'}")
    print(f"  {out_dir / 'misclassified_test_nodes_src_v10.csv'}")
    print(f"  {out_dir / 'extra_branch_effect_summary_src_v10.csv'}")
    print(f"  {out_dir / 'extra_branch_rewrite_nodes_src_v10.csv'}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run src_v10 safe adaptive-branch PCA and record test errors plus per-branch effects.'
    )
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--src-v10-file', type=str, default=None,
                        help='Optional explicit path to src_v10 safe adaptive branch python file.')
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)

    if args.src_v10_file is not None:
        src_v10_path = Path(args.src_v10_file)
    else:
        src_v10_path = find_first_existing([
            project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10.py',
            project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10(1).py',
        ])
    if src_v10_path is None or not src_v10_path.exists():
        raise FileNotFoundError('Cannot find src_v10 safe adaptive branch file. Use --src-v10-file explicitly.')

    mod = import_module_from_path(src_v10_path, 'algo1_safe_adaptive_branch_src_v10_audit')
    data_base = Path(args.data_base) if args.data_base else (drive_root / 'planetoid' / 'data' / args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else (project_root / 'scripts' / f'results_src_v10_branch_effect_audit_{args.dataset}')

    run_audit(mod, dataset=args.dataset, data_base=data_base, out_dir=out_dir,
              dim_candidates=[int(x) for x in args.dims], num_splits=int(args.num_splits))


if __name__ == '__main__':
    main()


