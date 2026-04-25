#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm 1.7: Per-Class Dimension + Coverage-Guided Safe Adaptive Branch
=========================================================================

Based on src_v10 conservative adaptive-branch PCA residual.

New ideas tested here
---------------------
A. Per-class dimension:
   Each class has its own root PCA dimension d_c instead of one global dim.

B. True-class coverage objective:
   Dimension search does not only maximize validation accuracy. It also rewards:
     - lower validation true-class residual
     - higher validation explained ratio E_true / T_true
   while penalizing:
     - false positives into a class
     - unnecessary dimension growth
     - changes that break too many previously correct validation nodes

The geometry-repulsion calibration from src_v11/src_v12 is NOT used here.
This file is meant to isolate whether per-class dimensions and coverage improve
the strong src_v10 behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse


# ============================================================
# Utilities
# ============================================================

def row_l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X / np.maximum(norm, eps)


def row_normalize(A):
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    return sparse.diags(inv) @ A


def find_raw_root(base):
    base = Path(base)
    hits = list(base.rglob('out1_node_feature_label.txt'))
    if not hits:
        raise FileNotFoundError('Cannot find out1_node_feature_label.txt')
    return hits[0].parent


def load_chameleon_raw(raw_root):
    raw_root = Path(raw_root)
    node_file = raw_root / 'out1_node_feature_label.txt'
    edge_file = raw_root / 'out1_graph_edges.txt'
    ids, feats, labels = [], [], []
    with open(node_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            node_id, feat, label = line.rstrip('\n').split('\t')
            ids.append(int(node_id))
            feats.append(np.array(feat.split(','), dtype=np.float64))
            labels.append(int(label))
    order = np.argsort(ids)
    X = np.vstack(feats)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    original_node_ids = np.asarray(ids, dtype=np.int64)[order]

    src, dst = [], []
    with open(edge_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            a, b = line.strip().split('\t')
            src.append(int(a))
            dst.append(int(b))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    n = X.shape[0]
    A_out = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n), dtype=np.float64)
    A_out.setdiag(0)
    A_out.eliminate_zeros()
    A_sym = ((A_out + A_out.T) > 0).astype(np.float64).tocsr()
    A_sym.setdiag(0)
    A_sym.eliminate_zeros()
    return X, y, A_sym, original_node_ids


def load_split(raw_root, repeat):
    path = Path(raw_root) / f'chameleon_split_0.6_0.2_{repeat}.npz'
    z = np.load(path)
    return (
        np.where(z['train_mask'].astype(bool))[0],
        np.where(z['val_mask'].astype(bool))[0],
        np.where(z['test_mask'].astype(bool))[0],
    )


def robust_median(arr, default=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def robust_quantile(arr, q, default=np.nan):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def write_csv(path, rows):
    path = Path(path)
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


# ============================================================
# PCA / subspace operations
# ============================================================

def orthonormalize_columns(B):
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0:
        return B.reshape(B.shape[0], 0)
    Q, _ = np.linalg.qr(B)
    keep = min(B.shape[1], Q.shape[1])
    return np.asarray(Q[:, :keep], dtype=np.float64)


def fit_pca_subspace(X, max_dim):
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0)
    C = X - mu
    d_eff = min(int(max_dim), max(0, X.shape[0] - 1), X.shape[1])
    if d_eff == 0:
        B = np.empty((X.shape[1], 0), dtype=np.float64)
    else:
        Gram = C @ C.T
        evals, evecs = np.linalg.eigh(Gram)
        order = np.argsort(evals)[::-1]
        keep = order[evals[order] > 1e-12][:d_eff]
        if keep.size == 0:
            B = np.empty((X.shape[1], 0), dtype=np.float64)
        else:
            B = C.T @ (evecs[:, keep] / np.sqrt(evals[keep])[None, :])
            B = orthonormalize_columns(np.asarray(B, dtype=np.float64))
    return mu, B


def subspace_stats(F, mu, B):
    C = F - mu
    total = np.sum(C * C, axis=1)
    if B.shape[1] == 0:
        evidence = np.zeros(F.shape[0], dtype=np.float64)
    else:
        coord = C @ B
        evidence = np.sum(coord * coord, axis=1)
    residual = np.maximum(total - evidence, 0.0)
    return total, evidence, residual


def fit_root_subspaces(F, y, train_idx, classes, class_dims):
    subspaces = {}
    for c in classes:
        c = int(c)
        idx = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        d = int(class_dims[c])
        mu, B = fit_pca_subspace(F[idx], max_dim=d)
        subspaces[c] = {
            'kind': 'root',
            'class_label': c,
            'mu': mu,
            'basis': B,
            'member_idx': idx,
            'root_dim_request': d,
            'center': np.mean(F[idx], axis=0),
            'radius': 1e8,
            'class_scale': 1.0,
            'extra_bias_scale': 0.0,
            'gate_strength': 0.0,
        }
    return subspaces


def root_stats_matrices(F, root_subspaces, classes):
    classes = np.asarray(classes, dtype=np.int64)
    n = F.shape[0]
    cnum = len(classes)
    total = np.zeros((n, cnum), dtype=np.float64)
    evidence = np.zeros((n, cnum), dtype=np.float64)
    residual = np.zeros((n, cnum), dtype=np.float64)
    for pos, c in enumerate(classes):
        b = root_subspaces[int(c)]
        t, e, r = subspace_stats(F, b['mu'], b['basis'])
        total[:, pos] = t
        evidence[:, pos] = e
        residual[:, pos] = r
    return total, evidence, residual


def residuals_to_root_subspaces(F, root_subspaces, classes):
    _, _, R = root_stats_matrices(F, root_subspaces, classes)
    return R


# ============================================================
# Coverage-guided per-class dimension search
# ============================================================

def evaluate_root_dim_setting(
    F, y, train_idx, val_idx, classes, class_dims,
    coverage_weight=0.010,
    explained_weight=0.004,
    fp_weight=0.020,
    dim_penalty=0.001,
):
    classes = np.asarray(classes, dtype=np.int64)
    class_to_pos = {int(c): pos for pos, c in enumerate(classes)}
    roots = fit_root_subspaces(F, y, train_idx, classes, class_dims)
    total, evidence, R = root_stats_matrices(F, roots, classes)
    pred = classes[np.argmin(R, axis=1)]

    val_true = y[val_idx]
    val_pred = pred[val_idx]
    val_acc = float(np.mean(val_pred == val_true))

    # Per-class scales from train true residual.
    scales = {}
    for c in classes:
        c = int(c)
        pos = class_to_pos[c]
        idx_c = train_idx[y[train_idx] == c]
        scales[c] = max(1e-8, robust_median(R[idx_c, pos], default=1.0))

    norm_true_res = []
    explained_ratios = []
    per_class_acc = {}
    fp_rates = []
    for c in classes:
        c = int(c)
        pos = class_to_pos[c]
        mask_c = (val_true == c)
        if np.any(mask_c):
            idx = val_idx[mask_c]
            norm_true_res.append(np.mean(R[idx, pos] / scales[c]))
            ratio = evidence[idx, pos] / np.maximum(total[idx, pos], 1e-12)
            explained_ratios.append(np.mean(ratio))
            per_class_acc[c] = float(np.mean(val_pred[mask_c] == c))
        else:
            per_class_acc[c] = np.nan

        non_c = (val_true != c)
        fp = np.sum((val_pred == c) & non_c)
        denom = max(1, int(np.sum(non_c)))
        fp_rates.append(fp / denom)

    balanced_norm_res = float(np.mean(norm_true_res)) if norm_true_res else 0.0
    balanced_explained = float(np.mean(explained_ratios)) if explained_ratios else 0.0
    mean_fp_rate = float(np.mean(fp_rates)) if fp_rates else 0.0
    mean_dim_frac = float(np.mean([class_dims[int(c)] for c in classes]) / max(class_dims.values()))

    objective = (
        val_acc
        - coverage_weight * balanced_norm_res
        + explained_weight * balanced_explained
        - fp_weight * mean_fp_rate
        - dim_penalty * mean_dim_frac
    )

    return {
        'objective': float(objective),
        'val_acc': val_acc,
        'balanced_norm_true_residual': balanced_norm_res,
        'balanced_true_explained_ratio': balanced_explained,
        'mean_fp_rate': mean_fp_rate,
        'mean_dim_frac': mean_dim_frac,
        'pred': pred,
        'roots': roots,
        'total': total,
        'evidence': evidence,
        'residual': R,
        'scales': scales,
        'per_class_acc': per_class_acc,
    }


def select_per_class_dims_by_coverage(
    F, y, train_idx, val_idx, classes, dim_candidates,
    max_passes=2,
    coverage_weight=0.010,
    explained_weight=0.004,
    fp_weight=0.020,
    dim_penalty=0.001,
    max_val_acc_drop=0.0,
    max_break_rate=0.015,
):
    classes = np.asarray(classes, dtype=np.int64)
    dim_candidates = [int(d) for d in dim_candidates]
    trace_rows = []

    def eval_dims(dims):
        return evaluate_root_dim_setting(
            F, y, train_idx, val_idx, classes, dims,
            coverage_weight=coverage_weight,
            explained_weight=explained_weight,
            fp_weight=fp_weight,
            dim_penalty=dim_penalty,
        )

    # Start from best global-dim setting under the same objective.
    best_global = None
    for d in dim_candidates:
        dims = {int(c): int(d) for c in classes}
        ev = eval_dims(dims)
        row = {
            'stage': 'global_init_scan',
            'changed_class': '',
            'trial_dim': int(d),
            'class_dims_json': json.dumps({str(k): int(v) for k, v in dims.items()}, ensure_ascii=False),
            **{k: ev[k] for k in ['objective', 'val_acc', 'balanced_norm_true_residual', 'balanced_true_explained_ratio', 'mean_fp_rate', 'mean_dim_frac']},
            'accepted': 0,
            'reason': '',
        }
        trace_rows.append(row)
        if best_global is None or ev['objective'] > best_global[0]['objective']:
            best_global = (ev, dims)

    current_eval, current_dims = best_global
    trace_rows.append({
        'stage': 'global_init_accept',
        'changed_class': '',
        'trial_dim': '',
        'class_dims_json': json.dumps({str(k): int(v) for k, v in current_dims.items()}, ensure_ascii=False),
        **{k: current_eval[k] for k in ['objective', 'val_acc', 'balanced_norm_true_residual', 'balanced_true_explained_ratio', 'mean_fp_rate', 'mean_dim_frac']},
        'accepted': 1,
        'reason': 'best_global_objective',
    })

    for pass_id in range(int(max_passes)):
        improved_any = False
        for c in classes:
            c = int(c)
            base_eval = current_eval
            base_pred_val = base_eval['pred'][val_idx]
            base_correct = base_pred_val == y[val_idx]
            best_candidate = None

            for d in dim_candidates:
                if int(d) == int(current_dims[c]):
                    continue
                trial_dims = dict(current_dims)
                trial_dims[c] = int(d)
                ev = eval_dims(trial_dims)
                trial_pred_val = ev['pred'][val_idx]
                trial_correct = trial_pred_val == y[val_idx]
                broken = int(np.sum(base_correct & (~trial_correct)))
                fixed = int(np.sum((~base_correct) & trial_correct))
                break_rate = broken / max(1, int(np.sum(base_correct)))

                accept_like = (
                    ev['objective'] > base_eval['objective'] + 1e-12
                    and ev['val_acc'] >= base_eval['val_acc'] - max_val_acc_drop - 1e-12
                    and break_rate <= max_break_rate + 1e-12
                )
                reason = []
                if ev['objective'] <= base_eval['objective'] + 1e-12:
                    reason.append('no_objective_gain')
                if ev['val_acc'] < base_eval['val_acc'] - max_val_acc_drop - 1e-12:
                    reason.append('val_acc_drop')
                if break_rate > max_break_rate + 1e-12:
                    reason.append('break_rate_high')
                if accept_like:
                    reason.append('candidate_ok')

                trace_rows.append({
                    'stage': f'greedy_pass_{pass_id}',
                    'changed_class': int(c),
                    'trial_dim': int(d),
                    'class_dims_json': json.dumps({str(k): int(v) for k, v in trial_dims.items()}, ensure_ascii=False),
                    **{k: ev[k] for k in ['objective', 'val_acc', 'balanced_norm_true_residual', 'balanced_true_explained_ratio', 'mean_fp_rate', 'mean_dim_frac']},
                    'base_objective': float(base_eval['objective']),
                    'base_val_acc': float(base_eval['val_acc']),
                    'fixed_vs_base': fixed,
                    'broken_vs_base': broken,
                    'break_rate': float(break_rate),
                    'accepted': 0,
                    'reason': '|'.join(reason),
                })

                if accept_like:
                    if best_candidate is None or ev['objective'] > best_candidate[0]['objective']:
                        best_candidate = (ev, trial_dims, fixed, broken, break_rate, d)

            if best_candidate is not None:
                current_eval, current_dims, fixed, broken, break_rate, d = best_candidate
                improved_any = True
                trace_rows.append({
                    'stage': f'greedy_pass_{pass_id}_accept',
                    'changed_class': int(c),
                    'trial_dim': int(d),
                    'class_dims_json': json.dumps({str(k): int(v) for k, v in current_dims.items()}, ensure_ascii=False),
                    **{k: current_eval[k] for k in ['objective', 'val_acc', 'balanced_norm_true_residual', 'balanced_true_explained_ratio', 'mean_fp_rate', 'mean_dim_frac']},
                    'fixed_vs_base': int(fixed),
                    'broken_vs_base': int(broken),
                    'break_rate': float(break_rate),
                    'accepted': 1,
                    'reason': 'best_class_coordinate_update',
                })

        if not improved_any:
            break

    return current_dims, current_eval, trace_rows


# ============================================================
# OOF diagnostics on root model with per-class dims
# ============================================================

def make_stratified_folds(train_idx, y, n_splits=4, seed=0):
    rng = np.random.RandomState(seed)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    fold_id = np.zeros(train_idx.shape[0], dtype=np.int64)
    unique_classes = np.unique(y[train_idx])
    for c in unique_classes:
        cls_local = np.where(y[train_idx] == c)[0].copy()
        rng.shuffle(cls_local)
        for k, local_pos in enumerate(cls_local):
            fold_id[local_pos] = k % n_splits
    return fold_id


def collect_oof_root_diagnostics(F, y, train_idx, classes, class_dims, n_folds=4, seed=0):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    class_to_pos = {int(c): pos for pos, c in enumerate(classes)}
    folds = make_stratified_folds(train_idx, y, n_splits=n_folds, seed=seed)

    n_all = F.shape[0]
    diag = {
        'pred_label': np.full(n_all, -1, dtype=np.int64),
        'true_residual': np.full(n_all, np.nan, dtype=np.float64),
        'best_other_label': np.full(n_all, -1, dtype=np.int64),
        'best_other_residual': np.full(n_all, np.nan, dtype=np.float64),
        'margin_to_other': np.full(n_all, np.nan, dtype=np.float64),
    }
    confusion_counts = {int(c): {} for c in classes}

    for fold in range(n_folds):
        inner_train = train_idx[folds != fold]
        inner_eval = train_idx[folds == fold]
        root_sub = fit_root_subspaces(F, y, inner_train, classes, class_dims)
        R = residuals_to_root_subspaces(F[inner_eval], root_sub, classes)
        pred = classes[np.argmin(R, axis=1)]

        for row_pos, node_idx in enumerate(inner_eval):
            true_label = int(y[node_idx])
            true_pos = class_to_pos[true_label]
            true_res = float(R[row_pos, true_pos])
            other_res = R[row_pos].copy()
            other_res[true_pos] = np.inf
            best_other_pos = int(np.argmin(other_res))
            best_other_label = int(classes[best_other_pos])
            best_other_res = float(other_res[best_other_pos])

            diag['pred_label'][node_idx] = int(pred[row_pos])
            diag['true_residual'][node_idx] = true_res
            diag['best_other_label'][node_idx] = best_other_label
            diag['best_other_residual'][node_idx] = best_other_res
            diag['margin_to_other'][node_idx] = best_other_res - true_res

            if int(pred[row_pos]) != true_label:
                confusion_counts[true_label][int(pred[row_pos])] = confusion_counts[true_label].get(int(pred[row_pos]), 0) + 1

    return diag, confusion_counts


# ============================================================
# Conservative adaptive branches, per-class dims
# ============================================================

def branch_budget(err_rate, n_class, max_extra_branches=2):
    if n_class < 30:
        return 0
    if err_rate < 0.10:
        return 0
    if err_rate < 0.20:
        return min(1, max_extra_branches)
    return min(2, max_extra_branches)


def mild_repel_basis(B, foreign_bases, repel_strength=0.03):
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0 or repel_strength <= 0:
        return B
    B_new = B.copy()
    for weight, Bref in foreign_bases:
        if Bref is None or Bref.size == 0 or weight <= 0:
            continue
        overlap_raw = np.linalg.norm(B_new.T @ Bref, ord='fro')
        denom = np.sqrt(max(1, B_new.shape[1]) * max(1, Bref.shape[1]))
        overlap = overlap_raw / max(denom, 1e-12)
        if overlap < 0.15:
            continue
        proj = Bref @ (Bref.T @ B_new)
        B_new = B_new - (repel_strength * float(weight) * overlap) * proj
    return orthonormalize_columns(B_new)


def build_local_group(F, pool_idx, seed_idx, min_branch_size=12, local_quantile=0.35, max_group_frac=0.45):
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    seed_idx = np.asarray(seed_idx, dtype=np.int64)
    if seed_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    center = np.mean(F[seed_idx], axis=0, keepdims=True)
    sqdist = np.sum((F[pool_idx] - center) ** 2, axis=1)
    order = np.argsort(sqdist)
    target_size = max(min_branch_size, int(np.ceil(local_quantile * pool_idx.size)))
    target_size = min(target_size, int(np.ceil(max_group_frac * pool_idx.size)))
    target_size = max(target_size, seed_idx.size)
    return np.unique(np.concatenate([seed_idx, pool_idx[order[:target_size]]])).astype(np.int64)


def group_overlap_ratio(a, b):
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    inter = len(sa & sb)
    union = max(1, len(sa | sb))
    return inter / union


def fit_branch_model(
    F,
    member_idx,
    max_dim,
    class_scale,
    confuser_bases=None,
    repel_strength=0.03,
    extra_bias_scale=0.10,
    gate_strength=2.5,
    gate_quantile=0.85,
    kind='root',
):
    member_idx = np.asarray(member_idx, dtype=np.int64)
    X = F[member_idx]
    mu, B = fit_pca_subspace(X, max_dim=max_dim)
    if kind == 'extra' and confuser_bases:
        B = mild_repel_basis(B, confuser_bases, repel_strength=repel_strength)
    center = np.mean(X, axis=0)
    member_sqdist = np.sum((X - center[None, :]) ** 2, axis=1)
    radius = robust_quantile(member_sqdist, gate_quantile, default=robust_median(member_sqdist, default=1.0))
    radius = max(radius, 1e-8)
    return {
        'kind': kind,
        'mu': mu,
        'basis': B,
        'member_idx': member_idx,
        'center': center,
        'radius': float(radius),
        'class_scale': float(max(class_scale, 1e-8)),
        'extra_bias_scale': float(extra_bias_scale),
        'gate_strength': float(gate_strength),
        'max_dim_used': int(max_dim),
    }


def branch_score_components(F, branch):
    total, evidence, residual = subspace_stats(F, branch['mu'], branch['basis'])
    if branch['kind'] == 'root':
        bias_penalty = np.zeros(F.shape[0], dtype=np.float64)
        gate_penalty = np.zeros(F.shape[0], dtype=np.float64)
        normalized = np.zeros(F.shape[0], dtype=np.float64)
        score = residual
    else:
        sqdist = np.sum((F - branch['center'][None, :]) ** 2, axis=1)
        normalized = sqdist / max(branch['radius'], 1e-8)
        gate_penalty = branch['gate_strength'] * branch['class_scale'] * np.maximum(normalized - 1.0, 0.0) ** 2
        bias_penalty = np.full(F.shape[0], branch['extra_bias_scale'] * branch['class_scale'], dtype=np.float64)
        score = residual + bias_penalty + gate_penalty
    return {
        'score': score,
        'total': total,
        'evidence': evidence,
        'residual': residual,
        'bias_penalty': bias_penalty,
        'gate_penalty': gate_penalty,
        'normalized_center_dist': normalized,
    }


def branch_score(F, branch):
    return branch_score_components(F, branch)['score']


def fit_perclass_coverage_adaptive_branches(
    F,
    y,
    train_idx,
    val_idx,
    classes,
    dim_candidates,
    dim_search_kwargs=None,
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
):
    if dim_search_kwargs is None:
        dim_search_kwargs = {}

    classes = np.asarray(classes, dtype=np.int64)
    class_dims, root_eval, dim_trace = select_per_class_dims_by_coverage(
        F, y, train_idx, val_idx, classes, dim_candidates, **dim_search_kwargs
    )
    root_sub = fit_root_subspaces(F, y, train_idx, classes, class_dims)
    diag, confusion_counts = collect_oof_root_diagnostics(
        F, y, train_idx, classes, class_dims, n_folds=internal_n_folds, seed=internal_seed
    )

    class_scales = {}
    branch_models = {int(c): [] for c in classes}
    branch_debug = {int(c): [] for c in classes}

    for c in classes:
        c = int(c)
        idx_c = train_idx[y[train_idx] == c]
        class_scales[c] = max(1e-8, robust_median(diag['true_residual'][idx_c], default=1.0))

    # Root branch uses selected per-class dim.
    for c in classes:
        c = int(c)
        idx_c = root_sub[c]['member_idx']
        root_model = fit_branch_model(
            F,
            member_idx=idx_c,
            max_dim=class_dims[c],
            class_scale=class_scales[c],
            confuser_bases=None,
            repel_strength=0.0,
            extra_bias_scale=0.0,
            gate_strength=0.0,
            kind='root',
        )
        root_model['root_dim_request'] = int(class_dims[c])
        branch_models[c].append(root_model)

    class_to_root_basis = {int(c): root_sub[int(c)]['basis'] for c in classes}

    for c in classes:
        c = int(c)
        idx_c = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        n_c = idx_c.size
        if n_c == 0:
            continue

        pred_c = diag['pred_label'][idx_c]
        true_res_c = diag['true_residual'][idx_c]
        margin_c = diag['margin_to_other'][idx_c]
        confuser_c = diag['best_other_label'][idx_c]

        err_rate = float(np.mean(pred_c != c))
        budget = branch_budget(err_rate, n_c, max_extra_branches=max_extra_branches)
        if budget == 0:
            continue

        res_thr = robust_quantile(true_res_c, high_res_quantile, default=np.inf)
        mar_thr = robust_quantile(margin_c, low_margin_quantile, default=-np.inf)
        mis_mask = pred_c != c
        hard_mask = mis_mask | ((np.isfinite(true_res_c) & (true_res_c >= res_thr)) & (np.isfinite(margin_c) & (margin_c <= mar_thr)))

        ranked_conf = sorted(confusion_counts.get(c, {}).items(), key=lambda kv: kv[1], reverse=True)
        selected_groups = []

        for conf, cnt in ranked_conf:
            if len(selected_groups) >= budget:
                break
            conf = int(conf)
            seed = np.asarray(idx_c[hard_mask & (confuser_c == conf)], dtype=np.int64)
            if seed.size < min_seed_size:
                continue
            if seed.size / max(1, n_c) < 0.06:
                continue

            group = build_local_group(
                F, pool_idx=idx_c, seed_idx=seed,
                min_branch_size=min_branch_size,
                local_quantile=local_quantile,
                max_group_frac=max_group_frac,
            )
            if group.size < min_branch_size:
                continue
            if any(group_overlap_ratio(group, prev) >= 0.55 for prev in selected_groups):
                continue

            root_mu = root_sub[c]['mu']
            root_B = root_sub[c]['basis']
            _, _, root_res_group = subspace_stats(F[group], root_mu, root_B)

            extra_dim = max(4, min(int(class_dims[c]), int(group.size - 1)))
            cand_mu, cand_B = fit_pca_subspace(F[group], max_dim=extra_dim)
            conf_basis = class_to_root_basis.get(conf, None)
            if conf_basis is not None and conf_basis.size > 0:
                cand_B = mild_repel_basis(cand_B, [(1.0, conf_basis)], repel_strength=repel_strength)
            _, _, cand_res_group = subspace_stats(F[group], cand_mu, cand_B)
            gain = robust_median(root_res_group - cand_res_group, default=0.0)
            if gain <= min_gain_ratio * class_scales[c]:
                continue

            selected_groups.append(group)

            confuser_bases = []
            if conf_basis is not None and conf_basis.size > 0:
                confuser_bases.append((1.0, conf_basis))

            extra_model = fit_branch_model(
                F,
                member_idx=group,
                max_dim=extra_dim,
                class_scale=class_scales[c],
                confuser_bases=confuser_bases,
                repel_strength=repel_strength,
                extra_bias_scale=extra_bias_scale,
                gate_strength=gate_strength,
                kind='extra',
            )
            extra_model['confuser_class'] = conf
            extra_model['seed_size'] = int(seed.size)
            extra_model['median_gain'] = float(gain)
            extra_model['root_dim_request'] = int(class_dims[c])
            branch_models[c].append(extra_model)
            branch_debug[c].append({
                'confuser': conf,
                'seed_size': int(seed.size),
                'group_size': int(group.size),
                'median_gain': float(gain),
                'extra_dim': int(extra_dim),
            })

    meta = {
        'class_dims': {int(k): int(v) for k, v in class_dims.items()},
        'root_eval': root_eval,
        'dim_trace': dim_trace,
        'class_scales': class_scales,
        'confusion_counts': confusion_counts,
        'branch_debug': branch_debug,
    }
    return branch_models, meta


def class_scores_from_branch_models(F, branch_models, classes):
    classes = np.asarray(classes, dtype=np.int64)
    n = F.shape[0]
    cnum = len(classes)
    scores = np.full((n, cnum), np.inf, dtype=np.float64)
    best_kind = np.empty((n, cnum), dtype=object)
    best_branch_idx = np.full((n, cnum), -1, dtype=np.int64)

    for pos, c in enumerate(classes):
        c = int(c)
        for b_idx, branch in enumerate(branch_models[c]):
            s = branch_score(F, branch)
            better = s < scores[:, pos]
            scores[better, pos] = s[better]
            best_kind[better, pos] = branch['kind']
            best_branch_idx[better, pos] = int(b_idx)
    return scores, best_kind, best_branch_idx


# ============================================================
# Path discovery / main experiment
# ============================================================

def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError('Cannot locate drive root containing planetoid/data. Please pass --data-base explicitly.')


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'src_v13').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing src_v13 and scripts.')


def resolve_default_paths(dataset: str):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v13_perclass_dim_coverage_{dataset}'
    return project_root, drive_root, data_base, out_dir


def run_experiment(dataset='chameleon', data_base=None, out_dir=None, dim_candidates=None, num_splits=10):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    project_root, drive_root, default_data_base, default_out_dir = resolve_default_paths(dataset)
    data_base = Path(data_base) if data_base is not None else default_data_base
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = find_raw_root(data_base)
    X_raw, y, A_sym, original_node_ids = load_chameleon_raw(raw_root)

    P = row_normalize(A_sym)
    PX = np.asarray(P @ X_raw)
    P2X = np.asarray(P @ PX)
    F = np.hstack([
        row_l2_normalize(X_raw),
        row_l2_normalize(PX),
        row_l2_normalize(P2X),
        row_l2_normalize(X_raw - PX),
        row_l2_normalize(PX - P2X),
    ])
    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
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

    rows = []
    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = load_split(raw_root, split)
        classes = np.unique(y[train_idx])

        branch_models, meta = fit_perclass_coverage_adaptive_branches(
            F, y, train_idx, val_idx, classes,
            dim_candidates=dim_candidates,
            **branch_kwargs,
        )
        scores, best_kind, best_branch_idx = class_scores_from_branch_models(F, branch_models, classes)
        pred = classes[np.argmin(scores, axis=1)]
        val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
        test_acc = float(np.mean(pred[test_idx] == y[test_idx]))

        total_branches = 0
        extra_branches = 0
        row = {
            'split': int(split),
            'class_dims_json': json.dumps({str(k): int(v) for k, v in meta['class_dims'].items()}, ensure_ascii=False),
            'val_acc': val_acc,
            'test_acc': test_acc,
        }
        for c in classes:
            c = int(c)
            bc = len(branch_models[c])
            eb = sum(1 for b in branch_models[c] if b['kind'] == 'extra')
            total_branches += bc
            extra_branches += eb
            row[f'class_{c}_dim'] = int(meta['class_dims'][c])
            row[f'class_{c}_branch_count'] = int(bc)
            row[f'class_{c}_extra_branch_count'] = int(eb)
        row['total_branches'] = int(total_branches)
        row['extra_branches'] = int(extra_branches)
        rows.append(row)

        print(
            f"split={split:2d}  dims={row['class_dims_json']}  "
            f"val={val_acc:.4f}  test={test_acc:.4f}  branches={total_branches} (extra={extra_branches})"
        )

    vals = np.asarray([r['val_acc'] for r in rows], dtype=np.float64)
    tests = np.asarray([r['test_acc'] for r in rows], dtype=np.float64)
    print()
    print(f'val_mean  = {np.mean(vals):.4f} +- {np.std(vals):.4f}')
    print(f'test_mean = {np.mean(tests):.4f} +- {np.std(tests):.4f}')

    write_csv(out_dir / 'split_summary_perclass_dim_coverage.csv', rows)
    (out_dir / 'run_summary.txt').write_text(
        '\n'.join([
            f'dataset={dataset}',
            f'data_base={data_base}',
            f'raw_root={raw_root}',
            f'out_dir={out_dir}',
            f'dim_candidates={dim_candidates}',
            f'val_mean={float(np.mean(vals)):.6f}',
            f'val_std={float(np.std(vals)):.6f}',
            f'test_mean={float(np.mean(tests)):.6f}',
            f'test_std={float(np.std(tests)):.6f}',
        ]),
        encoding='utf-8',
    )
    return {
        'val_mean': float(np.mean(vals)),
        'val_std': float(np.std(vals)),
        'test_mean': float(np.mean(tests)),
        'test_std': float(np.std(tests)),
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Run per-class dimension + coverage-guided adaptive branch.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        data_base=args.data_base,
        out_dir=args.out_dir,
        dim_candidates=[int(x) for x in args.dims],
        num_splits=int(args.num_splits),
    )


if __name__ == '__main__':
    main()


