
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm 1.6: Multihop-Full + Geometry-Coupled Safe Adaptive Branch PCA Residual
=================================================================================
Main idea
---------
Build on src_v10, but before adding local extra branches we add a *safe root calibration*
stage driven by pairwise class geometry and train OOF diagnostics.

For each split:
  1) Fit root per-class PCA subspaces as before.
  2) Run OOF diagnosis on the train set.
  3) Compute, for each class pair (i,j):
       - center distance
       - minimum principal angle
       - confusion counts
     and couple them into a pair closeness score.
  4) Choose top-accuracy classes as anchors.
  5) For non-anchor classes, apply a small repulsion / center-separation update away from
     the most coupled confuser classes, but only if:
       - geometry separation improves
       - OOF-correct train nodes for that class remain stable
     The amplitude is backtracked and tunable.
  6) On top of the calibrated roots, add conservative local extra branches (same spirit as v10).

Root calibration is intentionally conservative:
  - root score remains raw residual
  - only a few top coupled pairs influence each class
  - anchor classes move less / or not at all
  - the step is accepted only under explicit stability constraints
"""

from __future__ import annotations

import csv
import json
import math
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
    original_ids = np.asarray(ids, dtype=np.int64)[order]

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
    return X, y, A_sym, original_ids


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
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            full = {k: row.get(k, '') for k in keys}
            w.writerow(full)


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
        evals_keep = np.empty((0,), dtype=np.float64)
    else:
        Gram = C @ C.T
        evals, evecs = np.linalg.eigh(Gram)
        order = np.argsort(evals)[::-1]
        keep = order[evals[order] > 1e-12][:d_eff]
        if keep.size == 0:
            B = np.empty((X.shape[1], 0), dtype=np.float64)
            evals_keep = np.empty((0,), dtype=np.float64)
        else:
            evals_keep = evals[keep].copy()
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


def fit_root_subspaces(F, y, train_idx, classes, max_dim):
    subspaces = {}
    for c in classes:
        idx = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        mu, B = fit_pca_subspace(F[idx], max_dim=max_dim)
        subspaces[int(c)] = {
            'class_label': int(c),
            'mu': mu,
            'basis': B,
            'member_idx': idx,
        }
    return subspaces


def residuals_to_root_subspaces(F, root_subspaces, classes):
    classes = np.asarray(classes, dtype=np.int64)
    R = np.zeros((F.shape[0], len(classes)), dtype=np.float64)
    for pos, c in enumerate(classes):
        mu = root_subspaces[int(c)]['mu']
        B = root_subspaces[int(c)]['basis']
        _, _, residual = subspace_stats(F, mu, B)
        R[:, pos] = residual
    return R


def principal_geometry(mu_i, B_i, mu_j, B_j):
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    B_i = np.asarray(B_i, dtype=np.float64)
    B_j = np.asarray(B_j, dtype=np.float64)
    center_l2 = float(np.linalg.norm(mu_i - mu_j))
    if B_i.size == 0 or B_j.size == 0:
        overlap_fro = 0.0
        singular_vals = np.empty((0,), dtype=np.float64)
        min_angle_rad = np.pi / 2.0
    else:
        cross = B_i.T @ B_j
        singular_vals = np.linalg.svd(cross, compute_uv=False)
        overlap_fro = float(np.linalg.norm(cross, ord='fro'))
        clipped = np.clip(singular_vals, -1.0, 1.0)
        angles = np.arccos(clipped)
        min_angle_rad = float(np.min(angles)) if angles.size > 0 else np.pi / 2.0
    return center_l2, overlap_fro, min_angle_rad, singular_vals


# ============================================================
# OOF diagnostics on root model
# ============================================================

def make_stratified_folds(train_idx, y, n_splits=4, seed=0):
    rng = np.random.RandomState(seed)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    fold_id = np.zeros(train_idx.shape[0], dtype=np.int64)
    unique_classes = np.unique(y[train_idx])
    for c in unique_classes:
        cls_local = np.where(y[train_idx] == c)[0]
        cls_local = cls_local.copy()
        rng.shuffle(cls_local)
        for k, local_pos in enumerate(cls_local):
            fold_id[local_pos] = k % n_splits
    return fold_id


def collect_oof_root_diagnostics(F, y, train_idx, classes, max_dim, n_folds=4, seed=0):
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
        'is_correct': np.zeros(n_all, dtype=np.int64),
    }
    confusion_counts = {int(c): {} for c in classes}

    for fold in range(n_folds):
        inner_train = train_idx[folds != fold]
        inner_eval = train_idx[folds == fold]
        root_sub = fit_root_subspaces(F, y, inner_train, classes, max_dim=max_dim)
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
            diag['is_correct'][node_idx] = int(pred[row_pos] == true_label)

            if int(pred[row_pos]) != true_label:
                confusion_counts[true_label][int(pred[row_pos])] = confusion_counts[true_label].get(int(pred[row_pos]), 0) + 1

    return diag, confusion_counts


# ============================================================
# Root calibration driven by geometry + OOF
# ============================================================

def select_anchor_classes(train_idx, y, classes, diag, anchor_top_k=2):
    classes = np.asarray(classes, dtype=np.int64)
    rows = []
    for c in classes:
        idx_c = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        acc = float(np.mean(diag['is_correct'][idx_c])) if idx_c.size > 0 else 0.0
        rows.append((acc, int(c), int(idx_c.size)))
    rows = sorted(rows, key=lambda x: (-x[0], -x[2], x[1]))
    anchors = [c for _, c, _ in rows[:min(anchor_top_k, len(rows))]]
    return anchors, rows


def branch_budget(err_rate, n_class, max_extra_branches=2):
    if n_class < 30:
        return 0
    if err_rate < 0.10:
        return 0
    if err_rate < 0.20:
        return min(1, max_extra_branches)
    return min(2, max_extra_branches)


def build_pair_coupling_table(classes, root_subspaces, confusion_counts, distance_eps=1e-8, angle_eps=1e-6):
    classes = np.asarray(classes, dtype=np.int64)
    pair_rows = []
    d_vals = []
    a_vals = []
    c_vals = []
    for i in range(len(classes)):
        ci = int(classes[i])
        for j in range(i + 1, len(classes)):
            cj = int(classes[j])
            d, overlap, min_angle, singular_vals = principal_geometry(
                root_subspaces[ci]['mu'], root_subspaces[ci]['basis'],
                root_subspaces[cj]['mu'], root_subspaces[cj]['basis'],
            )
            conf_ij = int(confusion_counts.get(ci, {}).get(cj, 0))
            conf_ji = int(confusion_counts.get(cj, {}).get(ci, 0))
            pair_conf = conf_ij + conf_ji
            d_vals.append(d)
            a_vals.append(min_angle)
            c_vals.append(pair_conf)
            pair_rows.append({
                'class_i': ci,
                'class_j': cj,
                'center_l2_distance': float(d),
                'subspace_overlap_fro': float(overlap),
                'min_principal_angle_rad': float(min_angle),
                'min_principal_angle_deg': float(np.degrees(min_angle)),
                'pair_confusion': int(pair_conf),
                'conf_i_to_j': conf_ij,
                'conf_j_to_i': conf_ji,
                'principal_cosines_json': json.dumps([float(x) for x in singular_vals.tolist()], ensure_ascii=False),
            })

    dist_ref = robust_median(d_vals, default=1.0)
    angle_ref = robust_median(a_vals, default=1.0)
    conf_ref = max(1.0, robust_median(c_vals, default=1.0))

    pair_map = {}
    for row in pair_rows:
        d = float(row['center_l2_distance'])
        a = float(row['min_principal_angle_rad'])
        c = float(row['pair_confusion'])
        distance_close = dist_ref / max(d, distance_eps)
        angle_close = angle_ref / max(a, angle_eps)
        confusion_scale = c / conf_ref
        coupling = distance_close * angle_close
        coupled_confusion = coupling * confusion_scale
        row['distance_close_score'] = float(distance_close)
        row['angle_close_score'] = float(angle_close)
        row['coupling_distance_angle'] = float(coupling)
        row['coupled_confusion_score'] = float(coupled_confusion)
        pair_map[(int(row['class_i']), int(row['class_j']))] = row
        pair_map[(int(row['class_j']), int(row['class_i']))] = row
    return pair_rows, pair_map, dist_ref, angle_ref, conf_ref


def propose_root_adjustment(root_subspaces, class_c, target_infos, eta_basis, eta_center):
    """Create a proposal by repelling basis away from foreign bases and nudging center away from foreign centers."""
    mu = np.asarray(root_subspaces[class_c]['mu'], dtype=np.float64)
    B = np.asarray(root_subspaces[class_c]['basis'], dtype=np.float64)
    B_new = B.copy()
    center_push = np.zeros_like(mu)

    for info in target_infos:
        weight = float(info['weight'])
        mu_j = np.asarray(info['mu'], dtype=np.float64)
        B_j = np.asarray(info['basis'], dtype=np.float64)
        if B_j.size > 0 and B_new.size > 0 and eta_basis > 0:
            proj = B_j @ (B_j.T @ B_new)
            B_new = B_new - eta_basis * weight * proj
        if eta_center > 0:
            diff = mu - mu_j
            norm = float(np.linalg.norm(diff))
            if norm > 1e-12:
                center_push += weight * (diff / norm)

    B_new = orthonormalize_columns(B_new)
    mu_new = mu + eta_center * center_push
    return mu_new, B_new


def evaluate_root_proposal(F, y, train_idx, class_c, old_mu, old_B, new_mu, new_B, diag, target_infos,
                           class_scale, max_correct_harm_rate=0.03, max_correct_residual_increase_ratio=0.08):
    idx_c = np.asarray(train_idx[y[train_idx] == class_c], dtype=np.int64)
    if idx_c.size == 0:
        return None

    correct_idx = idx_c[diag['is_correct'][idx_c] == 1]
    hard_idx = idx_c[diag['best_other_label'][idx_c] >= 0]

    _, _, old_true_res_correct = subspace_stats(F[correct_idx], old_mu, old_B) if correct_idx.size > 0 else (None, None, np.empty((0,), dtype=np.float64))
    _, _, new_true_res_correct = subspace_stats(F[correct_idx], new_mu, new_B) if correct_idx.size > 0 else (None, None, np.empty((0,), dtype=np.float64))

    if correct_idx.size > 0:
        best_other = np.asarray(diag['best_other_residual'][correct_idx], dtype=np.float64)
        old_margin = best_other - old_true_res_correct
        new_margin = best_other - new_true_res_correct
        harmed = np.sum(new_margin <= 0.0)
        correct_harm_rate = float(harmed / max(1, correct_idx.size))
        residual_increase = np.maximum(new_true_res_correct - old_true_res_correct, 0.0)
        correct_residual_increase = float(np.mean(residual_increase))
    else:
        correct_harm_rate = 0.0
        correct_residual_increase = 0.0

    # Geometry gain on targeted confusing classes
    geom_gain = 0.0
    sep_gain_terms = []
    for info in target_infos:
        d_old, _, a_old, _ = principal_geometry(old_mu, old_B, info['mu'], info['basis'])
        d_new, _, a_new, _ = principal_geometry(new_mu, new_B, info['mu'], info['basis'])
        dist_gain = (d_new - d_old) / max(info['dist_ref'], 1e-8)
        angle_gain = (a_new - a_old) / max(info['angle_ref'], 1e-8)
        combined_gain = info['weight'] * (dist_gain + angle_gain)
        geom_gain += combined_gain
        sep_gain_terms.append(combined_gain)

    accepted = (
        (correct_harm_rate <= max_correct_harm_rate) and
        (correct_residual_increase <= max_correct_residual_increase_ratio * max(class_scale, 1e-8)) and
        (geom_gain > 0.0)
    )
    return {
        'accepted': bool(accepted),
        'correct_harm_rate': float(correct_harm_rate),
        'correct_residual_increase': float(correct_residual_increase),
        'geom_gain': float(geom_gain),
        'sep_gain_terms_json': json.dumps([float(x) for x in sep_gain_terms], ensure_ascii=False),
    }


def calibrate_root_subspaces(
    F,
    y,
    train_idx,
    classes,
    root_subspaces,
    diag,
    confusion_counts,
    anchor_top_k=2,
    pair_top_k=2,
    pair_confusion_min=12,
    base_eta_basis=0.10,
    eta_center_ratio=0.10,
    anchor_move_scale=0.25,
    max_correct_harm_rate=0.03,
    max_correct_residual_increase_ratio=0.08,
    pair_score_min=0.60,
):
    """
    Safe root calibration:
      - anchors are top OOF-accuracy classes
      - for each non-anchor class, repel from a few most coupled confuser classes
      - accept only if geometry improves and OOF-correct nodes stay stable
    """
    classes = np.asarray(classes, dtype=np.int64)
    anchor_classes, accuracy_rows = select_anchor_classes(train_idx, y, classes, diag, anchor_top_k=anchor_top_k)
    pair_rows, pair_map, dist_ref, angle_ref, conf_ref = build_pair_coupling_table(classes, root_subspaces, confusion_counts)

    calibrated = {
        int(c): {
            'class_label': int(c),
            'mu': np.asarray(root_subspaces[int(c)]['mu'], dtype=np.float64).copy(),
            'basis': np.asarray(root_subspaces[int(c)]['basis'], dtype=np.float64).copy(),
            'member_idx': np.asarray(root_subspaces[int(c)]['member_idx'], dtype=np.int64).copy(),
        }
        for c in classes
    }

    class_scales = {}
    for c in classes:
        idx_c = np.asarray(train_idx[y[train_idx] == c], dtype=np.int64)
        class_scales[int(c)] = max(1e-8, robust_median(diag['true_residual'][idx_c], default=1.0))

    class_debug = []
    pair_debug = []

    # Keep anchor classes more stable: by default we do not actively move them unless very compelling.
    ordered_classes = [int(c) for c in classes if int(c) not in anchor_classes] + [int(c) for c in anchor_classes]

    for c in ordered_classes:
        row = next(r for r in accuracy_rows if int(r[1]) == int(c))
        is_anchor = int(c) in anchor_classes

        # Collect candidate confusing pairs
        candidate_infos = []
        for other in classes:
            other = int(other)
            if other == int(c):
                continue
            prow = pair_map[(int(c), other)]
            pair_conf = int(prow['pair_confusion'])
            coupled_score = float(prow['coupled_confusion_score'])
            pair_debug.append({
                'class_label': int(c),
                'other_class': int(other),
                'is_anchor_class': int(is_anchor),
                'other_is_anchor': int(other in anchor_classes),
                'center_l2_distance': float(prow['center_l2_distance']),
                'min_principal_angle_deg': float(prow['min_principal_angle_deg']),
                'subspace_overlap_fro': float(prow['subspace_overlap_fro']),
                'pair_confusion': pair_conf,
                'coupling_distance_angle': float(prow['coupling_distance_angle']),
                'coupled_confusion_score': coupled_score,
            })
            if pair_conf < pair_confusion_min:
                continue
            if coupled_score < pair_score_min:
                continue
            candidate_infos.append({
                'other': other,
                'pair_confusion': pair_conf,
                'score': coupled_score,
                'mu': np.asarray(calibrated[other]['mu'], dtype=np.float64),
                'basis': np.asarray(calibrated[other]['basis'], dtype=np.float64),
                'dist_ref': float(dist_ref),
                'angle_ref': float(angle_ref),
            })

        if not candidate_infos:
            class_debug.append({
                'class_label': int(c),
                'is_anchor': int(is_anchor),
                'oof_accuracy': float(row[0]),
                'selected_pairs_json': json.dumps([], ensure_ascii=False),
                'accepted': 0,
                'accepted_eta_basis': 0.0,
                'accepted_eta_center': 0.0,
                'geom_gain': 0.0,
                'correct_harm_rate': 0.0,
                'correct_residual_increase': 0.0,
                'reason': 'no_candidate_pairs',
            })
            continue

        candidate_infos = sorted(candidate_infos, key=lambda x: (-x['score'], -x['pair_confusion'], x['other']))[:pair_top_k]
        total_score = sum(max(0.0, float(x['score'])) for x in candidate_infos)
        if total_score <= 0:
            weights = [1.0 / len(candidate_infos)] * len(candidate_infos)
        else:
            weights = [float(x['score']) / total_score for x in candidate_infos]
        for info, w in zip(candidate_infos, weights):
            info['weight'] = float(w)
        selected_pairs_json = json.dumps([int(x['other']) for x in candidate_infos], ensure_ascii=False)

        # Anchors move less. Non-anchors move more.
        class_eta_basis = float(base_eta_basis * (anchor_move_scale if is_anchor else 1.0))
        class_eta_center = float(eta_center_ratio * class_eta_basis)

        old_mu = np.asarray(calibrated[int(c)]['mu'], dtype=np.float64)
        old_B = np.asarray(calibrated[int(c)]['basis'], dtype=np.float64)
        class_scale = float(class_scales[int(c)])

        accepted = False
        accepted_info = None
        accepted_mu, accepted_B = old_mu, old_B
        eta_try = class_eta_basis
        eta_center_try = class_eta_center
        for _ in range(6):
            prop_mu, prop_B = propose_root_adjustment(
                calibrated, int(c), candidate_infos, eta_basis=eta_try, eta_center=eta_center_try
            )
            info = evaluate_root_proposal(
                F, y, train_idx, int(c), old_mu, old_B, prop_mu, prop_B, diag, candidate_infos,
                class_scale=class_scale,
                max_correct_harm_rate=max_correct_harm_rate,
                max_correct_residual_increase_ratio=max_correct_residual_increase_ratio,
            )
            if info is not None and info['accepted']:
                accepted = True
                accepted_info = info
                accepted_mu, accepted_B = prop_mu, prop_B
                break
            eta_try *= 0.5
            eta_center_try *= 0.5

        if accepted:
            calibrated[int(c)]['mu'] = np.asarray(accepted_mu, dtype=np.float64)
            calibrated[int(c)]['basis'] = np.asarray(accepted_B, dtype=np.float64)
            class_debug.append({
                'class_label': int(c),
                'is_anchor': int(is_anchor),
                'oof_accuracy': float(row[0]),
                'selected_pairs_json': selected_pairs_json,
                'accepted': 1,
                'accepted_eta_basis': float(eta_try),
                'accepted_eta_center': float(eta_center_try),
                'geom_gain': float(accepted_info['geom_gain']),
                'correct_harm_rate': float(accepted_info['correct_harm_rate']),
                'correct_residual_increase': float(accepted_info['correct_residual_increase']),
                'reason': 'accepted',
                'sep_gain_terms_json': accepted_info['sep_gain_terms_json'],
            })
        else:
            class_debug.append({
                'class_label': int(c),
                'is_anchor': int(is_anchor),
                'oof_accuracy': float(row[0]),
                'selected_pairs_json': selected_pairs_json,
                'accepted': 0,
                'accepted_eta_basis': 0.0,
                'accepted_eta_center': 0.0,
                'geom_gain': 0.0 if accepted_info is None else float(accepted_info['geom_gain']),
                'correct_harm_rate': 1.0 if accepted_info is None else float(accepted_info['correct_harm_rate']),
                'correct_residual_increase': 1.0 if accepted_info is None else float(accepted_info['correct_residual_increase']),
                'reason': 'rejected_by_stability_or_no_gain',
            })

    meta = {
        'anchor_classes': [int(x) for x in anchor_classes],
        'class_accuracy_rows': [(float(a), int(c), int(n)) for a, c, n in accuracy_rows],
        'class_debug': class_debug,
        'pair_debug': pair_debug,
        'class_scales': class_scales,
        'pair_refs': {'dist_ref': float(dist_ref), 'angle_ref': float(angle_ref), 'conf_ref': float(conf_ref)},
    }
    return calibrated, meta


# ============================================================
# Conservative adaptive branches on calibrated roots
# ============================================================

def mild_repel_basis(B, foreign_bases, repel_strength=0.03):
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0 or repel_strength <= 0:
        return B
    B_new = B.copy()
    for weight, Bref in foreign_bases:
        if Bref is None or Bref.size == 0 or weight <= 0:
            continue
        overlap_raw = np.linalg.norm(B_new.T @ Bref, ord='fro')
        denom = math.sqrt(max(1, B_new.shape[1]) * max(1, Bref.shape[1]))
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
    chosen = np.unique(np.concatenate([seed_idx, pool_idx[order[:target_size]]])).astype(np.int64)
    return chosen


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
    root_mu=None,
    root_B=None,
):
    member_idx = np.asarray(member_idx, dtype=np.int64)
    X = F[member_idx]
    if kind == 'root' and root_mu is not None and root_B is not None:
        mu = np.asarray(root_mu, dtype=np.float64)
        B = np.asarray(root_B, dtype=np.float64)
    else:
        mu, B = fit_pca_subspace(X, max_dim=max_dim)
        if kind == 'extra' and confuser_bases:
            B = mild_repel_basis(B, confuser_bases, repel_strength=repel_strength)

    center = np.mean(X, axis=0)
    member_sqdist = np.sum((X - center[None, :]) ** 2, axis=1)
    radius = robust_quantile(member_sqdist, gate_quantile, default=robust_median(member_sqdist, default=1.0))
    radius = max(radius, 1e-8)

    model = {
        'kind': kind,
        'mu': mu,
        'basis': B,
        'member_idx': member_idx,
        'center': center,
        'radius': float(radius),
        'class_scale': float(max(class_scale, 1e-8)),
        'extra_bias_scale': float(extra_bias_scale),
        'gate_strength': float(gate_strength),
    }
    return model


def branch_score_components(F, branch):
    total, evidence, residual = subspace_stats(F, branch['mu'], branch['basis'])
    sqdist = np.sum((F - branch['center'][None, :]) ** 2, axis=1)
    normalized = sqdist / max(branch['radius'], 1e-8)
    if branch['kind'] == 'root':
        bias_penalty = np.zeros(F.shape[0], dtype=np.float64)
        gate_penalty = np.zeros(F.shape[0], dtype=np.float64)
        score = residual
    else:
        gate_penalty = branch['gate_strength'] * branch['class_scale'] * np.maximum(normalized - 1.0, 0.0) ** 2
        bias_penalty = np.full(F.shape[0], branch['extra_bias_scale'] * branch['class_scale'], dtype=np.float64)
        score = residual + bias_penalty + gate_penalty
    return {
        'score': score,
        'total': total,
        'evidence': evidence,
        'residual': residual,
        'sqdist_to_center': sqdist,
        'normalized_center_dist': normalized,
        'bias_penalty': bias_penalty,
        'gate_penalty': gate_penalty,
    }


def class_scores_from_branch_models(F, branch_models, classes):
    classes = np.asarray(classes, dtype=np.int64)
    n = F.shape[0]
    cnum = len(classes)
    scores = np.full((n, cnum), np.inf, dtype=np.float64)
    best_kind = np.empty((n, cnum), dtype=object)
    best_branch_index = np.full((n, cnum), -1, dtype=np.int64)

    for pos, c in enumerate(classes):
        for branch_idx, branch in enumerate(branch_models[int(c)]):
            s = branch_score_components(F, branch)['score']
            better = s < scores[:, pos]
            scores[better, pos] = s[better]
            best_kind[better, pos] = branch['kind']
            best_branch_index[better, pos] = int(branch_idx)
    return scores, best_kind, best_branch_index


def fit_geometry_coupled_safe_branches(
    F,
    y,
    train_idx,
    classes,
    max_dim,
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
    # new root calibration params
    anchor_top_k=2,
    pair_top_k=2,
    pair_confusion_min=12,
    base_eta_basis=0.10,
    eta_center_ratio=0.10,
    anchor_move_scale=0.25,
    max_correct_harm_rate=0.03,
    max_correct_residual_increase_ratio=0.08,
    pair_score_min=0.60,
):
    classes = np.asarray(classes, dtype=np.int64)

    # Step A: initial roots + OOF
    root_sub = fit_root_subspaces(F, y, train_idx, classes, max_dim=max_dim)
    diag, confusion_counts = collect_oof_root_diagnostics(
        F, y, train_idx, classes, max_dim=max_dim, n_folds=internal_n_folds, seed=internal_seed
    )

    # Step B: safe geometry-coupled root calibration
    calibrated_roots, calibration_meta = calibrate_root_subspaces(
        F, y, train_idx, classes, root_sub, diag, confusion_counts,
        anchor_top_k=anchor_top_k,
        pair_top_k=pair_top_k,
        pair_confusion_min=pair_confusion_min,
        base_eta_basis=base_eta_basis,
        eta_center_ratio=eta_center_ratio,
        anchor_move_scale=anchor_move_scale,
        max_correct_harm_rate=max_correct_harm_rate,
        max_correct_residual_increase_ratio=max_correct_residual_increase_ratio,
        pair_score_min=pair_score_min,
    )

    class_scales = calibration_meta['class_scales']
    branch_models = {int(c): [] for c in classes}
    branch_debug = {int(c): [] for c in classes}

    # Root branches use calibrated mu/B but still keep the original root residual score.
    for c in classes:
        idx_c = calibrated_roots[int(c)]['member_idx']
        root_model = fit_branch_model(
            F,
            member_idx=idx_c,
            max_dim=max_dim,
            class_scale=class_scales[int(c)],
            confuser_bases=None,
            repel_strength=0.0,
            extra_bias_scale=0.0,
            gate_strength=0.0,
            kind='root',
            root_mu=calibrated_roots[int(c)]['mu'],
            root_B=calibrated_roots[int(c)]['basis'],
        )
        branch_models[int(c)].append(root_model)

    class_to_root_basis = {int(c): calibrated_roots[int(c)]['basis'] for c in classes}

    # Extra branches as in v10, but compare against calibrated roots and record direction stats.
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
        high_res_mask = np.isfinite(true_res_c) & (true_res_c >= res_thr)
        low_margin_mask = np.isfinite(margin_c) & (margin_c <= mar_thr)
        hard_mask = mis_mask | (high_res_mask & low_margin_mask)

        conf_count = confusion_counts.get(c, {})
        ranked_conf = sorted(conf_count.items(), key=lambda kv: kv[1], reverse=True)

        selected_groups = []
        for conf, cnt in ranked_conf:
            if len(selected_groups) >= budget:
                break
            conf = int(conf)

            seed = idx_c[hard_mask & (confuser_c == conf)]
            seed = np.asarray(seed, dtype=np.int64)
            if seed.size < min_seed_size:
                continue
            if seed.size / max(1, n_c) < 0.06:
                continue

            group = build_local_group(
                F,
                pool_idx=idx_c,
                seed_idx=seed,
                min_branch_size=min_branch_size,
                local_quantile=local_quantile,
                max_group_frac=max_group_frac,
            )
            if group.size < min_branch_size:
                continue

            duplicate = False
            for prev_group in selected_groups:
                if group_overlap_ratio(group, prev_group) >= 0.55:
                    duplicate = True
                    break
            if duplicate:
                continue

            root_mu = calibrated_roots[c]['mu']
            root_B = calibrated_roots[c]['basis']
            _, _, root_res_group = subspace_stats(F[group], root_mu, root_B)

            cand_mu, cand_B = fit_pca_subspace(F[group], max_dim=max(4, min(max_dim, group.size - 1)))
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
                max_dim=max(4, min(max_dim, group.size - 1)),
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

            # record how coupled this pair was during calibration
            pair_meta = next((r for r in calibration_meta['pair_debug'] if int(r['class_label']) == c and int(r['other_class']) == conf), None)
            if pair_meta is not None:
                extra_model['pair_center_l2_distance'] = float(pair_meta['center_l2_distance'])
                extra_model['pair_min_principal_angle_deg'] = float(pair_meta['min_principal_angle_deg'])
                extra_model['pair_coupled_confusion_score'] = float(pair_meta['coupled_confusion_score'])
            branch_models[c].append(extra_model)
            branch_debug[c].append({
                'confuser': conf,
                'seed_size': int(seed.size),
                'group_size': int(group.size),
                'median_gain': float(gain),
            })

    meta = {
        'class_scales': class_scales,
        'confusion_counts': confusion_counts,
        'branch_debug': branch_debug,
        'calibration_meta': calibration_meta,
        'root_subspaces_initial': root_sub,
        'root_subspaces_calibrated': calibrated_roots,
        'oof_diag': diag,
    }
    return branch_models, meta


# ============================================================
# Main experiment
# ============================================================

def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError(
        'Cannot locate drive root containing planetoid/data. '
        'Please pass --data-base explicitly.'
    )


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'src_v11').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError(
        'Cannot locate project root containing src_v11 and scripts. '
        'Please place this file inside src_v11 or pass --out-dir explicitly.'
    )


def resolve_default_paths(dataset: str):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v11_geometry_safe_branch_{dataset}'
    return project_root, drive_root, data_base, out_dir


def run_experiment(
    dataset='chameleon',
    data_base=None,
    out_dir=None,
    dim_candidates=None,
    num_splits=10,
):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    project_root, drive_root, default_data_base, default_out_dir = resolve_default_paths(dataset)
    data_base = Path(data_base) if data_base is not None else default_data_base
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir

    raw_root = find_raw_root(data_base)
    X_raw, y, A_sym, _ = load_chameleon_raw(raw_root)

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = load_split(raw_root, split)
        classes = np.unique(y[train_idx])

        best_val = -1.0
        best_dim = -1
        best_test = -1.0
        best_meta = None

        for dim in dim_candidates:
            branch_models, meta = fit_geometry_coupled_safe_branches(
                F, y, train_idx, classes, max_dim=dim, **branch_kwargs
            )
            scores, best_kind, best_branch_idx = class_scores_from_branch_models(F, branch_models, classes)
            pred = classes[np.argmin(scores, axis=1)]

            v = float(np.mean(pred[val_idx] == y[val_idx]))
            t = float(np.mean(pred[test_idx] == y[test_idx]))

            if v > best_val:
                best_val = v
                best_dim = dim
                best_test = t
                best_meta = {
                    'branch_models': branch_models,
                    'meta': meta,
                    'pred': pred,
                    'best_kind': best_kind,
                    'best_branch_idx': best_branch_idx,
                }

        total_branches = 0
        extra_branches = 0
        row = {
            'split': split,
            'best_dim': best_dim,
            'val_acc': best_val,
            'test_acc': best_test,
        }
        if best_meta is not None:
            anchors = best_meta['meta']['calibration_meta']['anchor_classes']
            row['anchor_classes_json'] = json.dumps([int(x) for x in anchors], ensure_ascii=False)
            for c in classes:
                bc = len(best_meta['branch_models'][int(c)])
                eb = sum(1 for b in best_meta['branch_models'][int(c)] if b['kind'] == 'extra')
                total_branches += bc
                extra_branches += eb
                row[f'class_{int(c)}_branch_count'] = bc
                row[f'class_{int(c)}_extra_branch_count'] = eb
        row['total_branches'] = total_branches
        row['extra_branches'] = extra_branches
        all_results.append(row)

        print(
            f'split={split:2d}  best_dim={best_dim:3d}  '
            f'val={best_val:.4f}  test={best_test:.4f}  '
            f'branches={total_branches} (extra={extra_branches})'
        )

    vals = [r['val_acc'] for r in all_results]
    tests = [r['test_acc'] for r in all_results]

    val_mean = float(np.mean(vals))
    val_std = float(np.std(vals))
    test_mean = float(np.mean(tests))
    test_std = float(np.std(tests))

    print()
    print(f'val_mean  = {val_mean:.4f} +- {val_std:.4f}')
    print(f'test_mean = {test_mean:.4f} +- {test_std:.4f}')

    write_csv(out_dir / 'split_summary_geometry_safe_branch.csv', all_results)

    summary_lines = [
        f'dataset={dataset}',
        f'data_base={data_base}',
        f'raw_root={raw_root}',
        f'out_dir={out_dir}',
        f'dim_candidates={dim_candidates}',
        f'val_mean={val_mean:.6f}',
        f'val_std={val_std:.6f}',
        f'test_mean={test_mean:.6f}',
        f'test_std={test_std:.6f}',
    ]
    (out_dir / 'run_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    return {
        'val_mean': val_mean,
        'val_std': val_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'rows': all_results,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run geometry-coupled safe adaptive-branch PCA residual classification on Chameleon.'
    )
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None,
                        help='Path like D:/planetoid/data/chameleon . Default: auto-discover from current file.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory. Default: project/scripts/results_src_v11_geometry_safe_branch_<dataset>')
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        data_base=args.data_base,
        out_dir=args.out_dir,
        dim_candidates=args.dims,
        num_splits=args.num_splits,
    )


