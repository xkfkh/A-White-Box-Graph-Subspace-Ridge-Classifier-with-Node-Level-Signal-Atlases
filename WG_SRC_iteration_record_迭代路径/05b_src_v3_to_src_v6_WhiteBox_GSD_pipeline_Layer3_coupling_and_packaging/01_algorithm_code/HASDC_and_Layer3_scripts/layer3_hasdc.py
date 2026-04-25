# -*- coding: utf-8 -*-
"""
Layer 3 HA-SDC: Homophily-Adaptive Signed Dual-Subspace Coupling
================================================================

Three scoring strategies:

1. channel_select: Pick smooth or raw channel based on overall val_acc.
   Simple but effective baseline for adaptive homophily handling.

2. merged_subspace: For each class c, merge smooth and raw subspace bases
   into a single enriched orthonormal basis via SVD on [B_c_s | B_c_r].
   Then compute plain residual on the merged subspace.
   This captures both local (raw) and neighborhood (smooth) structure.

3. dual_residual_rank: For each node, independently rank it by smooth-channel
   residual and raw-channel residual, then combine ranks (Borda count).
   Rank-based fusion is scale-invariant.

All methods are white-box: no MLP, no gradient, no learnable parameters.
"""

import os
import math
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _compute_channel_scores(
    Z, subspace_result, num_classes, mode="plain_residual",
    eta=0.10, activation_mode="mahalanobis", activation_rho=1e-6,
    max_activation_scale=None, aggregation_mode="suppression",
):
    """
    Compute per-class residual scores for a single channel.
    Returns: R [n, C], lower is better.
    """
    if mode == "plain_residual":
        from layer2_subspace import classify_by_residual
        _pred, neg_resid = classify_by_residual(Z, subspace_result, num_classes)
        return -neg_resid
    elif mode in ("suppression", "dynamic"):
        from layer3_discriminative import compute_discriminative_rperp
        R, _disc = compute_discriminative_rperp(
            Z, subspace_result, num_classes,
            eta_react=eta, activation_mode=activation_mode,
            activation_rho=activation_rho,
            max_activation_scale=max_activation_scale,
            aggregation_mode=aggregation_mode,
        )
        return R
    else:
        raise ValueError(f"Unknown scoring mode: {mode}")


def _compute_per_class_val_acc(R, Y, val_idx, num_classes):
    pred = R.argmin(dim=1)
    val_t = torch.tensor(val_idx, dtype=torch.long) if not torch.is_tensor(val_idx) else val_idx.long()
    Y_val = Y[val_t]
    pred_val = pred[val_t]
    per_class_acc = {}
    for c in range(num_classes):
        mask = (Y_val == c)
        n_c = mask.sum().item()
        if n_c == 0:
            per_class_acc[c] = 0.5
        else:
            per_class_acc[c] = float((pred_val[mask] == c).float().mean().item())
    overall = float((pred_val == Y_val).float().mean().item()) if val_t.numel() > 0 else 0.0
    return per_class_acc, overall


def build_merged_subspaces(sub_s, sub_r, num_classes, merged_dim=None):
    """
    For each class c, merge smooth and raw subspace bases into a single
    enriched orthonormal basis.

    Method: Concatenate B_c_s and B_c_r column-wise, then re-orthogonalize
    via SVD to get the top-d directions of the union subspace.

    Math:
      M_c = [B_c_s | B_c_r]   [D, d_s + d_r]
      SVD: M_c = U S V^T
      B_c_merged = U[:, :d_merged]

    The merged subspace spans the union of smooth and raw principal directions.
    On homophily graphs, smooth and raw bases are similar -> merged ~ smooth.
    On heterophily graphs, they differ -> merged captures complementary info.

    Returns: dict with mu_dict, B_dict, sub_dim_actual (compatible with
    classify_by_residual interface).
    """
    mu_s = sub_s['mu_dict']
    mu_r = sub_r['mu_dict']
    B_s = sub_s['B_dict']
    B_r = sub_r['B_dict']

    merged = {
        'mu_dict': {},
        'B_dict': {},
        'Sigma_dict': {},
        'S_dict': {},
        'var_ratio_dict': {},
        'sub_dim_actual': {},
        'n_c_dict': sub_s.get('n_c_dict', {}),
    }

    for c in range(num_classes):
        if c not in B_s or c not in B_r:
            continue

        # Merged mean: average of smooth and raw class means
        mu_merged = 0.5 * (mu_s[c] + mu_r[c])

        # Concatenate bases and re-orthogonalize
        M = torch.cat([B_s[c], B_r[c]], dim=1)  # [D, d_s + d_r]
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

        d_max = merged_dim or B_s[c].shape[1]
        d = min(d_max, U.shape[1])

        # Keep only directions with significant singular values
        # (singular values of concatenated orthonormal bases are in [0, sqrt(2)])
        B_merged = U[:, :d].clone()
        S_merged = S[:d].clone()

        # Construct compatible Sigma (use singular values as proxy for variance)
        var_proxy = S_merged ** 2 / max(merged['n_c_dict'].get(c, 2) - 1, 1)
        Sigma_merged = torch.diag(var_proxy)

        merged['mu_dict'][c] = mu_merged
        merged['B_dict'][c] = B_merged
        merged['Sigma_dict'][c] = Sigma_merged
        merged['S_dict'][c] = S_merged
        merged['sub_dim_actual'][c] = d
        merged['var_ratio_dict'][c] = 1.0  # not meaningful for merged

    return merged


def compute_hasdc_scores(
    Z, X, Y, train_idx, val_idx,
    subspace_result_s, subspace_result_r, num_classes,
    scoring_mode="plain_residual", eta=0.10, tau=5.0,
    activation_mode="mahalanobis", activation_rho=1e-6,
    max_activation_scale=None, aggregation_mode="suppression",
    fusion_strategy="channel_select", merged_dim=None,
):
    """
    HA-SDC scoring with multiple fusion strategies.

    fusion_strategy:
      "channel_select"   - pick better channel by overall val_acc
      "merged_subspace"  - merge smooth+raw bases, score on merged subspace
      "dual_rank"        - rank-based fusion (Borda count)
    """
    score_kw = dict(
        eta=eta, activation_mode=activation_mode,
        activation_rho=activation_rho,
        max_activation_scale=max_activation_scale,
        aggregation_mode=aggregation_mode,
    )

    # Always compute both channels for diagnostics
    R_s = _compute_channel_scores(Z, subspace_result_s, num_classes, mode=scoring_mode, **score_kw)
    R_r = _compute_channel_scores(X, subspace_result_r, num_classes, mode=scoring_mode, **score_kw)
    acc_s, overall_s = _compute_per_class_val_acc(R_s, Y, val_idx, num_classes)
    acc_r, overall_r = _compute_per_class_val_acc(R_r, Y, val_idx, num_classes)

    if fusion_strategy == "channel_select":
        if overall_s >= overall_r:
            R_final = R_s.clone()
            selected = "smooth"
        else:
            R_final = R_r.clone()
            selected = "raw"
        diag = {"selected_channel": selected}

    elif fusion_strategy == "merged_subspace":
        merged_sub = build_merged_subspaces(
            subspace_result_s, subspace_result_r, num_classes,
            merged_dim=merged_dim,
        )
        # Score using merged subspace on BOTH Z and X, pick better
        R_ms = _compute_channel_scores(Z, merged_sub, num_classes, mode=scoring_mode, **score_kw)
        R_mr = _compute_channel_scores(X, merged_sub, num_classes, mode=scoring_mode, **score_kw)
        _, ov_ms = _compute_per_class_val_acc(R_ms, Y, val_idx, num_classes)
        _, ov_mr = _compute_per_class_val_acc(R_mr, Y, val_idx, num_classes)
        if ov_ms >= ov_mr:
            R_final = R_ms
            selected = "merged_smooth"
        else:
            R_final = R_mr
            selected = "merged_raw"
        diag = {"selected_channel": selected, "merged_val_smooth": ov_ms, "merged_val_raw": ov_mr}

    elif fusion_strategy == "dual_rank":
        # Borda count: for each class c, rank nodes by R_s[:,c] and R_r[:,c],
        # then sum ranks. Lower combined rank = better.
        n = R_s.shape[0]
        R_final = torch.zeros_like(R_s)
        for c in range(num_classes):
            rank_s = torch.zeros(n)
            rank_r = torch.zeros(n)
            rank_s[R_s[:, c].argsort()] = torch.arange(n, dtype=torch.float)
            rank_r[R_r[:, c].argsort()] = torch.arange(n, dtype=torch.float)
            R_final[:, c] = rank_s + rank_r
        selected = "dual_rank"
        diag = {"selected_channel": selected}

    else:
        raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")

    g_c = {}
    for c in range(num_classes):
        g_c[c] = _sigmoid(tau * (acc_s.get(c, 0.5) - acc_r.get(c, 0.5)))

    diagnostics = {
        "g_c": g_c,
        "acc_smooth": acc_s,
        "acc_raw": acc_r,
        "overall_val_smooth": overall_s,
        "overall_val_raw": overall_r,
        "fusion_strategy": fusion_strategy,
        "scoring_mode": scoring_mode,
        **diag,
    }
    return R_final, diagnostics


def compute_freq_decomposed_scores(
    Z, X, Y, train_idx, val_idx, num_classes, sub_dim=12,
    scoring_mode="plain_residual", eta=0.10,
    activation_mode="mahalanobis", activation_rho=1e-6,
    max_activation_scale=None, aggregation_mode="suppression",
):
    """
    Frequency-decomposed subspace discriminant.

    Decomposes features into:
      Low-freq:  Z = (I + lam*L)^{-1} X  (smooth component)
      High-freq: H = X - Z               (residual = what smoothing removes)

    On homophily graphs: low-freq Z is discriminative, H is noise.
    On heterophily graphs: H contains the discriminative signal that
    smoothing destroys.

    Strategy: build subspaces on Z, X, and H independently.
    Score on all three, pick the best by val_acc.

    Math:
      H = X - Z = X - (I + lam*L)^{-1} X = [I - (I + lam*L)^{-1}] X
        = lam*L * (I + lam*L)^{-1} X
      This is the graph high-pass filtered version of X.

    White-box: all subspaces are PCA, scoring is residual-based,
    channel selection is by val_acc.
    """
    from layer2_subspace import build_class_subspaces

    H = X - Z  # high-frequency component

    train_list = list(train_idx) if not isinstance(train_idx, list) else train_idx

    # Build subspaces for each frequency band
    sub_z = build_class_subspaces(Z, Y, train_list, num_classes, sub_dim=sub_dim)
    sub_x = build_class_subspaces(X, Y, train_list, num_classes, sub_dim=sub_dim)
    sub_h = build_class_subspaces(H, Y, train_list, num_classes, sub_dim=sub_dim)

    score_kw = dict(
        eta=eta, activation_mode=activation_mode,
        activation_rho=activation_rho,
        max_activation_scale=max_activation_scale,
        aggregation_mode=aggregation_mode,
    )

    # Score each channel
    R_z = _compute_channel_scores(Z, sub_z, num_classes, mode=scoring_mode, **score_kw)
    R_x = _compute_channel_scores(X, sub_x, num_classes, mode=scoring_mode, **score_kw)
    R_h = _compute_channel_scores(H, sub_h, num_classes, mode=scoring_mode, **score_kw)

    _, ov_z = _compute_per_class_val_acc(R_z, Y, val_idx, num_classes)
    _, ov_x = _compute_per_class_val_acc(R_x, Y, val_idx, num_classes)
    _, ov_h = _compute_per_class_val_acc(R_h, Y, val_idx, num_classes)

    # Pick best channel
    candidates = [("low_freq", R_z, ov_z), ("raw", R_x, ov_x), ("high_freq", R_h, ov_h)]
    best_name, R_final, best_val = max(candidates, key=lambda t: t[2])

    diagnostics = {
        "selected_channel": best_name,
        "val_low_freq": ov_z,
        "val_raw": ov_x,
        "val_high_freq": ov_h,
        "scoring_mode": scoring_mode,
    }
    return R_final, diagnostics
