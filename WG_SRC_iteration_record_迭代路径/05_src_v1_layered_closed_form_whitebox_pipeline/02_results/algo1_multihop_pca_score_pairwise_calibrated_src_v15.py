#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm 1.9 / src_v15
Per-Class Dimension Floor + Coverage Branch + Score Calibration + Pairwise Specialist
===================================================================================

This version keeps src_v14's subspace construction unchanged, then adds a safe
score-level calibration layer:

1. Class bias calibration:
   S'_c(x) = S_c(x) + b_c
   Accepted only if validation fixed - lambda * broken > 0.

2. Pairwise top-2 specialist:
   For dangerous top-2 pairs, fit a tiny ridge linear rule on validation features:
       score diff, evidence diff, total diff, residual diff, explained-ratio diff,
       1-hop/2-hop train exposure diff.
   It only triggers when the top-2 pair is the target pair and gap <= threshold.
   Accepted only if validation fixed - lambda * broken > 0 and broken rate is small.

No PCA basis / branch geometry is changed here.
The goal is to fix remaining systematic pairwise mistakes without damaging
previously correct points.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# General IO / import helpers
# ============================================================

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
        if (p / 'scripts').exists() and (p / 'src_v14').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts and src_v14')


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


def resolve_default_paths(dataset: str, v14_file=None):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v15_score_pairwise_calibrated_{dataset}'

    if v14_file is not None:
        v14_path = Path(v14_file)
    else:
        v14_path = find_first_existing([
            project_root / 'src_v14' / 'algo1_multihop_pca_perclass_dim_floor_coverage_branch_src_v14.py',
            project_root / 'src_v14' / 'algo1_multihop_pca_perclass_dim_floor_coverage_branch.py',
        ])
    if v14_path is None or not v14_path.exists():
        raise FileNotFoundError('Cannot find src_v14 algorithm file. Pass --src-v14-file explicitly.')

    return project_root, drive_root, data_base, out_dir, v14_path


def load_v14_module(v14_file=None):
    _, _, _, _, v14_path = resolve_default_paths('chameleon', v14_file)
    return import_module_from_path(v14_path, 'src_v14_base_for_v15')


# ============================================================
# Feature construction / exposure
# ============================================================

def build_multihop_features(mod, X_raw, A_sym):
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
    return F


def compute_train_exposures(mod, A_sym, y, train_idx, classes):
    P = mod.row_normalize(A_sym)
    n = A_sym.shape[0]
    cnum = len(classes)
    exp1 = np.zeros((n, cnum), dtype=np.float64)
    exp2 = np.zeros((n, cnum), dtype=np.float64)
    for pos, c in enumerate(classes):
        c = int(c)
        m = np.zeros(n, dtype=np.float64)
        cls_train = train_idx[y[train_idx] == c]
        m[cls_train] = 1.0
        exp1[:, pos] = np.asarray(P @ m).ravel()
        exp2[:, pos] = np.asarray(P @ exp1[:, pos]).ravel()
    return exp1, exp2


# ============================================================
# Score/stat matrix helpers
# ============================================================

def branch_score_components_safe(mod, F, branch):
    # src_v14 already exposes this.
    return mod.branch_score_components(F, branch)


def class_stat_matrices(mod, F, branch_models, classes):
    classes = np.asarray(classes, dtype=np.int64)
    n = F.shape[0]
    cnum = len(classes)

    scores = np.full((n, cnum), np.inf, dtype=np.float64)
    total = np.full((n, cnum), np.nan, dtype=np.float64)
    evidence = np.full((n, cnum), np.nan, dtype=np.float64)
    residual = np.full((n, cnum), np.nan, dtype=np.float64)
    bias_penalty = np.zeros((n, cnum), dtype=np.float64)
    gate_penalty = np.zeros((n, cnum), dtype=np.float64)
    norm_center = np.zeros((n, cnum), dtype=np.float64)
    best_kind = np.empty((n, cnum), dtype=object)
    best_branch_idx = np.full((n, cnum), -1, dtype=np.int64)

    for pos, c in enumerate(classes):
        c = int(c)
        for b_idx, branch in enumerate(branch_models[c]):
            comp = branch_score_components_safe(mod, F, branch)
            s = comp['score']
            better = s < scores[:, pos]
            scores[better, pos] = s[better]
            total[better, pos] = comp['total'][better]
            evidence[better, pos] = comp['evidence'][better]
            residual[better, pos] = comp['residual'][better]
            bias_penalty[better, pos] = comp['bias_penalty'][better]
            gate_penalty[better, pos] = comp['gate_penalty'][better]
            norm_center[better, pos] = comp['normalized_center_dist'][better]
            best_kind[better, pos] = branch.get('kind', '')
            best_branch_idx[better, pos] = int(b_idx)

    ratio = evidence / np.maximum(total, 1e-12)
    return {
        'score': scores,
        'total': total,
        'evidence': evidence,
        'residual': residual,
        'explained_ratio': ratio,
        'bias_penalty': bias_penalty,
        'gate_penalty': gate_penalty,
        'normalized_center_dist': norm_center,
        'best_kind': best_kind,
        'best_branch_idx': best_branch_idx,
    }


def pred_from_scores(scores, classes):
    classes = np.asarray(classes, dtype=np.int64)
    return classes[np.argmin(scores, axis=1)]


def top2_info(scores, classes):
    order = np.argsort(scores, axis=1)
    top1_pos = order[:, 0]
    top2_pos = order[:, 1]
    top1 = classes[top1_pos]
    top2 = classes[top2_pos]
    gap = scores[np.arange(scores.shape[0]), top2_pos] - scores[np.arange(scores.shape[0]), top1_pos]
    return top1, top2, top1_pos, top2_pos, gap


# ============================================================
# Calibration: class bias
# ============================================================

def evaluate_pred_change(y, idx, old_pred, new_pred):
    old_correct = old_pred[idx] == y[idx]
    new_correct = new_pred[idx] == y[idx]
    fixed = int(np.sum((~old_correct) & new_correct))
    broken = int(np.sum(old_correct & (~new_correct)))
    redirect = int(np.sum((~old_correct) & (~new_correct) & (old_pred[idx] != new_pred[idx])))
    old_num_correct = max(1, int(np.sum(old_correct)))
    broken_rate = broken / old_num_correct
    return fixed, broken, redirect, broken_rate


def fit_class_bias_calibration(
    y,
    val_idx,
    classes,
    base_scores,
    bias_grid=None,
    lambda_broken=1.5,
    max_broken_rate=0.012,
    max_passes=2,
):
    classes = np.asarray(classes, dtype=np.int64)
    cnum = len(classes)
    if bias_grid is None:
        bias_grid = [-0.16, -0.12, -0.08, -0.04, 0.04, 0.08, 0.12, 0.16]

    bias = np.zeros(cnum, dtype=np.float64)
    trace = []

    current_scores = base_scores + bias[None, :]
    current_pred = pred_from_scores(current_scores, classes)

    for pass_id in range(int(max_passes)):
        improved = False
        for pos, c in enumerate(classes):
            best = None
            for delta in bias_grid:
                trial_bias = bias.copy()
                trial_bias[pos] += float(delta)
                trial_scores = base_scores + trial_bias[None, :]
                trial_pred = pred_from_scores(trial_scores, classes)
                fixed, broken, redirect, broken_rate = evaluate_pred_change(y, val_idx, current_pred, trial_pred)
                gain = fixed - lambda_broken * broken
                accepted_like = gain > 0 and broken_rate <= max_broken_rate

                trace.append({
                    'stage': 'class_bias_scan',
                    'pass_id': int(pass_id),
                    'class_label': int(c),
                    'delta': float(delta),
                    'candidate_bias': float(trial_bias[pos]),
                    'fixed': fixed,
                    'broken': broken,
                    'redirect': redirect,
                    'broken_rate': float(broken_rate),
                    'gain': float(gain),
                    'accepted': 0,
                    'reason': 'candidate_ok' if accepted_like else 'reject',
                })

                if accepted_like:
                    # Prefer larger gain, then fewer broken, then smaller absolute bias.
                    key = (gain, -broken, -abs(trial_bias[pos]))
                    if best is None or key > best[0]:
                        best = (key, trial_bias, trial_pred, fixed, broken, redirect, broken_rate, gain, delta)

            if best is not None:
                _, bias, current_pred, fixed, broken, redirect, broken_rate, gain, delta = best
                current_scores = base_scores + bias[None, :]
                improved = True
                trace.append({
                    'stage': 'class_bias_accept',
                    'pass_id': int(pass_id),
                    'class_label': int(c),
                    'delta': float(delta),
                    'candidate_bias': float(bias[pos]),
                    'fixed': int(fixed),
                    'broken': int(broken),
                    'redirect': int(redirect),
                    'broken_rate': float(broken_rate),
                    'gain': float(gain),
                    'accepted': 1,
                    'reason': 'accepted_best_for_class',
                })

        if not improved:
            break

    return bias, current_pred, trace


# ============================================================
# Calibration: pairwise top-2 specialist
# ============================================================

def make_pair_features(stats, scores_cal, exp1, exp2, classes, a, b, nodes):
    class_to_pos = {int(c): pos for pos, c in enumerate(classes)}
    pa, pb = class_to_pos[int(a)], class_to_pos[int(b)]
    nodes = np.asarray(nodes, dtype=np.int64)

    Sa = scores_cal[nodes, pa]
    Sb = scores_cal[nodes, pb]
    Ea = stats['evidence'][nodes, pa]
    Eb = stats['evidence'][nodes, pb]
    Ta = stats['total'][nodes, pa]
    Tb = stats['total'][nodes, pb]
    Ra = stats['residual'][nodes, pa]
    Rb = stats['residual'][nodes, pb]
    rhoa = stats['explained_ratio'][nodes, pa]
    rhob = stats['explained_ratio'][nodes, pb]
    e1a = exp1[nodes, pa]
    e1b = exp1[nodes, pb]
    e2a = exp2[nodes, pa]
    e2b = exp2[nodes, pb]

    # Positive h -> class a, negative h -> class b after ridge fit.
    X = np.vstack([
        Sa - Sb,
        Ea - Eb,
        Ta - Tb,
        Ra - Rb,
        rhoa - rhob,
        e1a - e1b,
        e2a - e2b,
    ]).T
    return np.asarray(X, dtype=np.float64)


def fit_ridge_binary(X, y_pm, ridge=1.0):
    X = np.asarray(X, dtype=np.float64)
    y_pm = np.asarray(y_pm, dtype=np.float64)
    if X.shape[0] < 4:
        return None
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean[None, :]) / std[None, :]
    Xd = np.hstack([Xs, np.ones((Xs.shape[0], 1), dtype=np.float64)])
    A = Xd.T @ Xd + float(ridge) * np.eye(Xd.shape[1])
    A[-1, -1] -= float(ridge)  # do not regularize intercept
    b = Xd.T @ y_pm
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return {
        'coef': w[:-1],
        'intercept': float(w[-1]),
        'mean': mean,
        'std': std,
    }


def apply_ridge(model, X):
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model['mean'][None, :]) / model['std'][None, :]
    return Xs @ model['coef'] + model['intercept']


def pair_key(a, b):
    a, b = int(a), int(b)
    return tuple(sorted((a, b)))


def default_danger_pairs():
    return [(2, 3), (0, 1), (1, 2), (0, 2), (3, 4), (2, 4)]


def candidate_pairs_from_val(y, val_idx, pred, classes, base_scores, min_confusion=4):
    # Include default dangerous pairs and pairs that appear in val confusion.
    pairs = set(pair_key(a, b) for a, b in default_danger_pairs())
    for node in val_idx:
        yt = int(y[node])
        yp = int(pred[node])
        if yt != yp:
            pairs.add(pair_key(yt, yp))
    return sorted(pairs)


def apply_pairwise_specialists(
    y,
    nodes,
    classes,
    stats,
    scores_cal,
    pred_in,
    specialists,
    exp1,
    exp2,
):
    classes = np.asarray(classes, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)
    pred = pred_in.copy()
    _, _, top1_pos, top2_pos, gap = top2_info(scores_cal, classes)
    top1 = classes[top1_pos]
    top2 = classes[top2_pos]

    for spec in specialists:
        a = int(spec['class_a'])
        b = int(spec['class_b'])
        max_gap = float(spec['max_gap'])
        mask = np.zeros(nodes.shape[0], dtype=bool)
        for ii, node in enumerate(nodes):
            if pair_key(top1[node], top2[node]) == pair_key(a, b) and gap[node] <= max_gap:
                mask[ii] = True
        if not np.any(mask):
            continue
        selected_nodes = nodes[mask]
        X = make_pair_features(stats, scores_cal, exp1, exp2, classes, a, b, selected_nodes)
        h = apply_ridge(spec, X)
        pair_pred = np.where(h >= 0, a, b).astype(np.int64)
        pred[selected_nodes] = pair_pred
    return pred


def fit_pairwise_specialists(
    y,
    val_idx,
    classes,
    stats,
    scores_cal,
    pred_after_bias,
    exp1,
    exp2,
    lambda_broken=1.5,
    max_broken_rate=0.012,
    gap_candidates=(0.15, 0.25, 0.35, 0.50),
    min_pair_train=10,
    ridge=1.0,
):
    classes = np.asarray(classes, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    specialists = []
    trace = []

    current_pred = pred_after_bias.copy()
    pairs = candidate_pairs_from_val(y, val_idx, current_pred, classes, scores_cal)

    top1, top2, _, _, gap_all = top2_info(scores_cal, classes)

    for a, b in pairs:
        a, b = int(a), int(b)

        best_spec = None
        for max_gap in gap_candidates:
            # Train only on val nodes whose true label is in pair and top2 pair is this pair.
            train_nodes = []
            for node in val_idx:
                if int(y[node]) not in (a, b):
                    continue
                if pair_key(top1[node], top2[node]) != pair_key(a, b):
                    continue
                if gap_all[node] > float(max_gap):
                    continue
                train_nodes.append(int(node))
            if len(train_nodes) < min_pair_train:
                trace.append({
                    'pair': f'{a}-{b}',
                    'class_a': a,
                    'class_b': b,
                    'max_gap': float(max_gap),
                    'train_nodes': int(len(train_nodes)),
                    'fixed': '',
                    'broken': '',
                    'gain': '',
                    'accepted': 0,
                    'reason': 'too_few_pair_nodes',
                })
                continue

            X = make_pair_features(stats, scores_cal, exp1, exp2, classes, a, b, train_nodes)
            y_pm = np.where(y[np.asarray(train_nodes)] == a, 1.0, -1.0)
            model = fit_ridge_binary(X, y_pm, ridge=ridge)
            if model is None:
                continue

            spec = {
                'class_a': a,
                'class_b': b,
                'max_gap': float(max_gap),
                **model,
            }

            trial_pred = apply_pairwise_specialists(
                y, val_idx, classes, stats, scores_cal, current_pred, [spec], exp1, exp2
            )
            fixed, broken, redirect, broken_rate = evaluate_pred_change(y, val_idx, current_pred, trial_pred)
            gain = fixed - lambda_broken * broken
            accepted_like = gain > 0 and broken_rate <= max_broken_rate

            trace.append({
                'pair': f'{a}-{b}',
                'class_a': a,
                'class_b': b,
                'max_gap': float(max_gap),
                'train_nodes': int(len(train_nodes)),
                'fixed': fixed,
                'broken': broken,
                'redirect': redirect,
                'broken_rate': float(broken_rate),
                'gain': float(gain),
                'accepted': 0,
                'reason': 'candidate_ok' if accepted_like else 'reject',
                'coef_json': json.dumps([float(x) for x in model['coef'].tolist()], ensure_ascii=False),
                'intercept': float(model['intercept']),
            })

            if accepted_like:
                key = (gain, -broken, fixed, -float(max_gap))
                if best_spec is None or key > best_spec[0]:
                    best_spec = (key, spec, trial_pred, fixed, broken, redirect, broken_rate, gain, len(train_nodes))

        if best_spec is not None:
            _, spec, current_pred, fixed, broken, redirect, broken_rate, gain, ntrain = best_spec
            specialists.append(spec)
            trace.append({
                'pair': f'{a}-{b}',
                'class_a': a,
                'class_b': b,
                'max_gap': float(spec['max_gap']),
                'train_nodes': int(ntrain),
                'fixed': int(fixed),
                'broken': int(broken),
                'redirect': int(redirect),
                'broken_rate': float(broken_rate),
                'gain': float(gain),
                'accepted': 1,
                'reason': 'accepted_pairwise_specialist',
                'coef_json': json.dumps([float(x) for x in spec['coef'].tolist()], ensure_ascii=False),
                'intercept': float(spec['intercept']),
            })

    return specialists, current_pred, trace


def apply_full_calibration(stats, classes, class_bias, specialists, exp1, exp2, nodes=None):
    classes = np.asarray(classes, dtype=np.int64)
    scores_cal = stats['score'] + np.asarray(class_bias, dtype=np.float64)[None, :]
    pred_bias = pred_from_scores(scores_cal, classes)
    if nodes is None:
        nodes = np.arange(scores_cal.shape[0], dtype=np.int64)
    pred_final = pred_bias.copy()
    if specialists:
        pred_final = apply_pairwise_specialists(
            None, np.asarray(nodes, dtype=np.int64), classes, stats, scores_cal, pred_final,
            specialists, exp1, exp2
        )
    return scores_cal, pred_bias, pred_final


# ============================================================
# Model fit
# ============================================================

def fit_src_v15_model(
    mod,
    F,
    y,
    train_idx,
    val_idx,
    classes,
    dim_candidates,
    A_sym,
    class_bias_kwargs=None,
    pairwise_kwargs=None,
    branch_kwargs=None,
):
    if class_bias_kwargs is None:
        class_bias_kwargs = {}
    if pairwise_kwargs is None:
        pairwise_kwargs = {}
    if branch_kwargs is None:
        branch_kwargs = {}

    default_branch_kwargs = dict(
        dim_search_kwargs=dict(
            max_passes=2,
            coverage_weight=0.010,
            explained_weight=0.004,
            fp_weight=0.020,
            dim_penalty=0.001,
            max_val_acc_drop=0.0,
            max_break_rate=0.015,
            high_global_dim_threshold=48,
            low_dim_floor_if_global_high=32,
            max_low_dim_classes_if_global_high=1,
            protected_classes=(0, 1),
            protected_min_dim=32,
            fallback_require_val_gain=True,
            fallback_allow_equal_if_coverage_gain=True,
            min_coverage_gain_to_keep=0.005,
            max_global_break_rate=0.020,
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
    default_branch_kwargs.update(branch_kwargs)

    branch_models, meta = mod.fit_perclass_coverage_adaptive_branches(
        F, y, train_idx, val_idx, classes,
        dim_candidates=dim_candidates,
        **default_branch_kwargs,
    )

    stats = class_stat_matrices(mod, F, branch_models, classes)
    base_pred = pred_from_scores(stats['score'], classes)
    exp1, exp2 = compute_train_exposures(mod, A_sym, y, train_idx, classes)

    default_class_bias_kwargs = dict(
        lambda_broken=1.5,
        max_broken_rate=0.012,
        max_passes=2,
        bias_grid=[-0.16, -0.12, -0.08, -0.04, 0.04, 0.08, 0.12, 0.16],
    )
    default_class_bias_kwargs.update(class_bias_kwargs)

    class_bias, pred_after_bias, class_bias_trace = fit_class_bias_calibration(
        y, val_idx, classes, stats['score'], **default_class_bias_kwargs
    )

    scores_cal = stats['score'] + class_bias[None, :]

    default_pairwise_kwargs = dict(
        lambda_broken=1.5,
        max_broken_rate=0.012,
        gap_candidates=(0.15, 0.25, 0.35, 0.50),
        min_pair_train=10,
        ridge=1.0,
    )
    default_pairwise_kwargs.update(pairwise_kwargs)

    specialists, pred_after_pairwise, pairwise_trace = fit_pairwise_specialists(
        y, val_idx, classes, stats, scores_cal, pred_after_bias, exp1, exp2,
        **default_pairwise_kwargs,
    )

    # Final prediction for all nodes.
    pred_final = pred_after_bias.copy()
    if specialists:
        pred_final = apply_pairwise_specialists(
            y, np.arange(F.shape[0], dtype=np.int64), classes, stats, scores_cal,
            pred_final, specialists, exp1, exp2
        )

    meta_v15 = {
        **meta,
        'class_bias': {int(c): float(class_bias[pos]) for pos, c in enumerate(classes)},
        'class_bias_trace': class_bias_trace,
        'pairwise_specialists': specialists,
        'pairwise_trace': pairwise_trace,
        'base_pred': base_pred,
        'pred_after_bias': pred_after_bias,
        'pred_final': pred_final,
        'stats': stats,
        'scores_calibrated': scores_cal,
        'train_exposure_1hop': exp1,
        'train_exposure_2hop': exp2,
    }
    return branch_models, meta_v15


# ============================================================
# Main experiment
# ============================================================

def run_experiment(dataset='chameleon', data_base=None, out_dir=None, dim_candidates=None, num_splits=10, src_v14_file=None):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    project_root, drive_root, default_data_base, default_out_dir, v14_path = resolve_default_paths(dataset, src_v14_file)
    data_base = Path(data_base) if data_base is not None else default_data_base
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = import_module_from_path(v14_path, 'src_v14_base_for_v15_run')
    raw_root = mod.find_raw_root(data_base)
    X_raw, y, A_sym, original_node_ids = mod.load_chameleon_raw(raw_root)
    F = build_multihop_features(mod, X_raw, A_sym)

    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Src_v14 file: {v14_path}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

    rows = []
    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        branch_models, meta = fit_src_v15_model(
            mod, F, y, train_idx, val_idx, classes,
            dim_candidates=dim_candidates,
            A_sym=A_sym,
        )
        pred = meta['pred_final']
        base_pred = meta['base_pred']
        bias_pred = meta['pred_after_bias']

        val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
        test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
        base_val_acc = float(np.mean(base_pred[val_idx] == y[val_idx]))
        base_test_acc = float(np.mean(base_pred[test_idx] == y[test_idx]))
        bias_val_acc = float(np.mean(bias_pred[val_idx] == y[val_idx]))
        bias_test_acc = float(np.mean(bias_pred[test_idx] == y[test_idx]))

        total_branches = sum(len(branch_models[int(c)]) for c in classes)
        extra_branches = sum(1 for c in classes for b in branch_models[int(c)] if b.get('kind') == 'extra')

        row = {
            'split': int(split),
            'class_dims_json': json.dumps({str(k): int(v) for k, v in meta['class_dims'].items()}, ensure_ascii=False),
            'class_bias_json': json.dumps({str(k): float(v) for k, v in meta['class_bias'].items()}, ensure_ascii=False),
            'accepted_pairwise_count': int(len(meta['pairwise_specialists'])),
            'accepted_pairwise_pairs_json': json.dumps([f"{int(s['class_a'])}-{int(s['class_b'])}" for s in meta['pairwise_specialists']], ensure_ascii=False),
            'base_val_acc': base_val_acc,
            'base_test_acc': base_test_acc,
            'bias_val_acc': bias_val_acc,
            'bias_test_acc': bias_test_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
        }
        rows.append(row)
        print(
            f"split={split:2d} val={val_acc:.4f} test={test_acc:.4f} "
            f"base_test={base_test_acc:.4f} bias_test={bias_test_acc:.4f} "
            f"pairs={len(meta['pairwise_specialists'])} branches={total_branches} (extra={extra_branches})"
        )

    vals = np.asarray([r['val_acc'] for r in rows], dtype=np.float64)
    tests = np.asarray([r['test_acc'] for r in rows], dtype=np.float64)
    print()
    print(f'val_mean  = {np.mean(vals):.4f} +- {np.std(vals):.4f}')
    print(f'test_mean = {np.mean(tests):.4f} +- {np.std(tests):.4f}')

    write_csv(out_dir / 'split_summary_src_v15_score_pairwise.csv', rows)
    (out_dir / 'run_summary_src_v15_score_pairwise.json').write_text(
        json.dumps({
            'dataset': dataset,
            'data_base': str(data_base),
            'raw_root': str(raw_root),
            'src_v14_file': str(v14_path),
            'out_dir': str(out_dir),
            'dim_candidates': json.dumps([int(x) for x in dim_candidates], ensure_ascii=False),
            'num_splits': int(num_splits),
            'val_mean': float(np.mean(vals)),
            'val_std': float(np.std(vals)),
            'test_mean': float(np.mean(tests)),
            'test_std': float(np.std(tests)),
        }, ensure_ascii=False, indent=2),
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
    parser = argparse.ArgumentParser(description='Run src_v15 score/pairwise calibrated model.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--src-v14-file', type=str, default=None)
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        data_base=args.data_base,
        out_dir=args.out_dir,
        dim_candidates=[int(x) for x in args.dims],
        num_splits=int(args.num_splits),
        src_v14_file=args.src_v14_file,
    )


if __name__ == '__main__':
    main()


