# -*- coding: utf-8 -*-
"""
Diagnostic analysis of WHY suppression fails + 4 new Layer3 candidate algorithms.
Run on cornell (183 nodes, fastest iteration) with Geom-GCN split 0.

New Layer3 candidates:
  1. Fisher-weighted residual
  2. PPCA scoring (orthogonal residual + within-subspace Mahalanobis + noise variance)
  3. Promotion (amplify unique directions instead of suppressing shared ones)
  4. Adaptive sub_dim per class based on spectral gap

All are white-box, no MLP, no gradient, no learnable parameters.
"""

import os, sys, time, json
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
SRC_V5 = os.path.join(BASE_DIR, "src_v5")

sys.path.insert(0, SCRIPT_DIR)
from whitebox_src_v5_bridge import setup_src_v5, build_subspaces_src, tikhonov_smooth_src
from fair_utils import load_extended_dataset, build_dense_laplacian, accuracy_from_pred

setup_src_v5(SRC_V5)
from layer2_subspace import classify_by_residual, build_class_subspaces
from layer3_discriminative import (
    compute_discriminative_rperp,
    compute_geometric_overlap_matrices,
    compute_base_reaction_matrices,
)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "layer3_diagnosis")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load cornell, split 0
# ============================================================
DATA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "..", "paper_data")
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "..", "planetoid", "data")

print("=== Loading cornell split=0 ===")
g = load_extended_dataset("cornell", DATA_ROOT, repeat=0, feature_norm="row_l1")
X = g.x.float()
Y = g.y.long()
train_idx = g.train_idx.numpy().tolist()
val_idx = g.val_idx.numpy().tolist()
test_idx = g.test_idx.numpy().tolist()
num_classes = g.num_classes
print(f"  n={g.num_nodes}, D={g.num_features}, C={num_classes}")
print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

L = build_dense_laplacian(g.edge_index, g.num_nodes, torch.device('cpu'))

# ============================================================
# Part 1: Diagnostic - why does suppression fail?
# ============================================================
print("\n" + "="*60)
print("PART 1: DIAGNOSTIC ANALYSIS")
print("="*60)

lam = 1.0
sub_dim = 12
Z = tikhonov_smooth_src(X, L, lam)
sub = build_subspaces_src(Z, Y, train_idx, num_classes, sub_dim)

# 1a. Overlap matrix spectral analysis
print("\n--- 1a. Geometric overlap G_{c<-c'} spectral analysis ---")
G_dict = compute_geometric_overlap_matrices(sub['B_dict'], num_classes)
for c in range(num_classes):
    for c2 in range(num_classes):
        if c == c2 or (c, c2) not in G_dict:
            continue
        G = G_dict[(c, c2)]
        evals = torch.linalg.eigvalsh(G)
        tr = evals.sum().item()
        max_e = evals.max().item()
        print(f"  G[{c}<-{c2}]: trace={tr:.4f}, max_eig={max_e:.4f}, "
              f"eigs=[{', '.join(f'{e:.3f}' for e in evals[-5:].tolist())}]")

# 1b. What suppression actually does to scores
print("\n--- 1b. Suppression vs plain residual comparison ---")
_pred_plain, neg_resid_plain = classify_by_residual(Z, sub, num_classes)
R_plain = -neg_resid_plain  # lower is better

R_supp, disc = compute_discriminative_rperp(Z, sub, num_classes, eta_react=0.10)

# Compare on test set
pred_plain = R_plain.argmin(dim=1)
pred_supp = R_supp.argmin(dim=1)
test_t = torch.tensor(test_idx, dtype=torch.long)
acc_plain = (pred_plain[test_t] == Y[test_t]).float().mean().item()
acc_supp = (pred_supp[test_t] == Y[test_t]).float().mean().item()
print(f"  Plain residual test_acc: {acc_plain:.4f}")
print(f"  Suppression test_acc:    {acc_supp:.4f}")
print(f"  Delta:                   {acc_supp - acc_plain:+.4f}")

# 1c. Analyze WHERE suppression changes predictions
changed = (pred_plain != pred_supp)
n_changed = changed.sum().item()
n_changed_test = changed[test_t].sum().item()
# Of the changed predictions, how many went from correct to wrong?
plain_correct = (pred_plain[test_t] == Y[test_t])
supp_correct = (pred_supp[test_t] == Y[test_t])
got_worse = (plain_correct & ~supp_correct).sum().item()
got_better = (~plain_correct & supp_correct).sum().item()
print(f"\n  Predictions changed: {n_changed}/{Z.shape[0]} total, {n_changed_test}/{len(test_idx)} test")
print(f"  Test: got_worse={got_worse}, got_better={got_better}")

# 1d. Activation distribution analysis
print("\n--- 1d. Activation distribution ---")
activations = disc['activations']
for c in range(num_classes):
    if c not in activations:
        continue
    a = activations[c]
    print(f"  Class {c}: mean={a.mean():.4f}, std={a.std():.4f}, "
          f"min={a.min():.4f}, max={a.max():.4f}, median={a.median():.4f}")

# 1e. Key insight: how much does W_c shrink the projection reward?
print("\n--- 1e. Projection reward shrinkage by W_c ---")
for c in range(min(num_classes, 5)):
    B_c = sub['B_dict'][c]
    mu_c = sub['mu_dict'][c]
    u_c = (Z - mu_c) @ B_c  # [n, d]

    # Plain reward: u^T u
    plain_reward = (u_c ** 2).sum(dim=1)

    # Suppressed reward: u^T W_c u
    W_dyn = disc['dynamic_matrices'][c]  # [n, d, d]
    u_col = u_c.unsqueeze(2)
    supp_reward = (u_col.transpose(1,2) @ W_dyn @ u_col).squeeze(-1).squeeze(-1)

    ratio = (supp_reward / plain_reward.clamp(min=1e-10))
    print(f"  Class {c}: reward_ratio mean={ratio.mean():.4f}, "
          f"min={ratio.min():.4f}, max={ratio.max():.4f}")
    # If ratio < 1 everywhere, suppression is REDUCING the projection reward
    # for the correct class too, which hurts discrimination


# ============================================================
# Part 2: NEW LAYER3 ALGORITHMS
# ============================================================
print("\n" + "="*60)
print("PART 2: NEW LAYER3 CANDIDATE ALGORITHMS")
print("="*60)

def eval_scores(R, Y, train_idx, val_idx, test_idx, name):
    """Evaluate R [n, C] (lower is better) on train/val/test."""
    pred = R.argmin(dim=1)
    tr_t = torch.tensor(train_idx, dtype=torch.long)
    va_t = torch.tensor(val_idx, dtype=torch.long)
    te_t = torch.tensor(test_idx, dtype=torch.long)
    tr_acc = (pred[tr_t] == Y[tr_t]).float().mean().item()
    va_acc = (pred[va_t] == Y[va_t]).float().mean().item()
    te_acc = (pred[te_t] == Y[te_t]).float().mean().item()
    print(f"  [{name}] train={tr_acc:.4f} val={va_acc:.4f} test={te_acc:.4f}")
    return {"name": name, "train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc}


def fisher_weighted_residual(Z, sub, num_classes, train_idx, alpha=0.0):
    """
    Fisher-weighted residual scoring.
    Directions with high between/within variance ratio get higher projection reward.
    alpha=0 -> plain residual; alpha=1 -> fully Fisher-weighted.
    """
    n = Z.shape[0]
    R = torch.zeros(n, num_classes)
    tr_t = torch.tensor(train_idx, dtype=torch.long)
    mu_global = Z[tr_t].mean(dim=0)

    for c in range(num_classes):
        B_c = sub['B_dict'][c]
        mu_c = sub['mu_dict'][c]
        S_c = sub['S_dict'][c]
        n_c = sub['n_c_dict'][c]
        d_c = B_c.shape[1]

        within_var = (S_c ** 2) / max(n_c - 1, 1)
        between_var = torch.zeros(d_c)
        for c2 in range(num_classes):
            if c2 == c or c2 not in sub['mu_dict']:
                continue
            n_c2 = sub['n_c_dict'][c2]
            proj = B_c.t() @ (sub['mu_dict'][c2] - mu_global)
            between_var += n_c2 * proj ** 2

        fisher = between_var / (within_var + 1e-8)
        f_max = fisher.max().clamp(min=1e-8)
        w = (1 - alpha) + alpha * (fisher / f_max)

        diff = Z - mu_c
        total_sq = (diff ** 2).sum(dim=1)
        proj = diff @ B_c
        weighted_proj_sq = (proj ** 2 * w.unsqueeze(0)).sum(dim=1)
        R[:, c] = total_sq - weighted_proj_sq

    return R


def ppca_scoring(Z, sub, num_classes, train_idx, noise_floor=1e-4):
    """
    PPCA-based scoring (negative log-likelihood proxy).
    Balances within-subspace Mahalanobis + orthogonal residual/noise + log-det.
    """
    n, D = Z.shape
    R = torch.zeros(n, num_classes)
    tr_t = torch.tensor(train_idx, dtype=torch.long)

    for c in range(num_classes):
        B_c = sub['B_dict'][c]
        mu_c = sub['mu_dict'][c]
        S_c = sub['S_dict'][c]
        n_c = sub['n_c_dict'][c]
        d_c = B_c.shape[1]

        sigma_sq = (S_c ** 2) / max(n_c - 1, 1)
        sigma_sq = sigma_sq.clamp(min=noise_floor)

        mask_c = (Y[tr_t] == c)
        diff_train = Z[tr_t][mask_c] - mu_c
        total_var = (diff_train ** 2).sum().item() / max(n_c - 1, 1)
        explained_var = sigma_sq.sum().item()
        remaining_dims = max(D - d_c, 1)
        sigma_noise_sq = max((total_var - explained_var) / remaining_dims, noise_floor)

        diff = Z - mu_c
        proj = diff @ B_c
        maha = (proj ** 2 / sigma_sq.unsqueeze(0)).sum(dim=1)
        proj_back = proj @ B_c.t()
        ortho_resid = diff - proj_back
        ortho_sq = (ortho_resid ** 2).sum(dim=1) / sigma_noise_sq
        log_det = sigma_sq.log().sum() + remaining_dims * np.log(sigma_noise_sq)

        R[:, c] = maha + ortho_sq + log_det

    return R


def promotion_scoring(Z, sub, num_classes, beta=1.0):
    """
    Promotion: amplify unique directions instead of suppressing shared ones.
    uniqueness_k = 1 - max_{c'!=c} ||P_{c'} B_c[:,k]||^2
    w_k = 1 + beta * uniqueness_k
    """
    n = Z.shape[0]
    R = torch.zeros(n, num_classes)

    for c in range(num_classes):
        B_c = sub['B_dict'][c]
        mu_c = sub['mu_dict'][c]
        d_c = B_c.shape[1]

        uniqueness = torch.ones(d_c)
        for k in range(d_c):
            v_k = B_c[:, k]
            max_overlap = 0.0
            for c2 in range(num_classes):
                if c2 == c or c2 not in sub['B_dict']:
                    continue
                B_c2 = sub['B_dict'][c2]
                overlap = (B_c2.t() @ v_k).pow(2).sum().item()
                max_overlap = max(max_overlap, overlap)
            uniqueness[k] = 1.0 - max_overlap

        w = 1.0 + beta * uniqueness.clamp(min=0)
        diff = Z - mu_c
        total_sq = (diff ** 2).sum(dim=1)
        proj = diff @ B_c
        weighted_proj_sq = (proj ** 2 * w.unsqueeze(0)).sum(dim=1)
        R[:, c] = total_sq - weighted_proj_sq

    return R


def adaptive_subdim_residual(Z, Y, train_idx, num_classes, max_dim=32, gap_ratio=0.5):
    """
    Adaptive sub_dim per class based on spectral gap.
    For each class, find the largest gap in singular values and cut there.
    gap_ratio: threshold for relative gap (S[k]/S[k+1] > 1/gap_ratio triggers cut).
    """
    n = Z.shape[0]
    tr_t = torch.tensor(train_idx, dtype=torch.long)
    Z_tr = Z[tr_t]
    Y_tr = Y[tr_t]

    sub_adaptive = {'mu_dict': {}, 'B_dict': {}, 'Sigma_dict': {}, 'S_dict': {},
                    'var_ratio_dict': {}, 'sub_dim_actual': {}, 'n_c_dict': {}}

    for c in range(num_classes):
        mask = (Y_tr == c)
        n_c = mask.sum().item()
        if n_c < 2:
            continue
        X_c = Z_tr[mask]
        mu_c = X_c.mean(dim=0)
        Xc = X_c - mu_c
        _, S_full, Vt = torch.linalg.svd(Xc, full_matrices=False)
        V = Vt.t()

        # Find spectral gap
        k = min(max_dim, S_full.shape[0])
        S_top = S_full[:k]
        best_d = k
        if k > 2:
            ratios = S_top[:-1] / S_top[1:].clamp(min=1e-10)
            # Find first big gap
            for i in range(len(ratios)):
                if ratios[i] > 1.0 / gap_ratio and i >= 1:
                    best_d = i + 1
                    break

        d = max(best_d, 2)  # at least 2 dims
        B_c = V[:, :d].clone()
        S_c = S_full[:d].clone()
        var_c = S_c ** 2 / max(n_c - 1, 1)

        sub_adaptive['mu_dict'][c] = mu_c
        sub_adaptive['B_dict'][c] = B_c
        sub_adaptive['Sigma_dict'][c] = torch.diag(var_c)
        sub_adaptive['S_dict'][c] = S_c
        sub_adaptive['sub_dim_actual'][c] = d
        sub_adaptive['n_c_dict'][c] = n_c
        total_var = (S_full ** 2).sum().item()
        sub_adaptive['var_ratio_dict'][c] = (S_c ** 2).sum().item() / max(total_var, 1e-12)

    # Print adaptive dims
    for c in range(num_classes):
        if c in sub_adaptive['sub_dim_actual']:
            print(f"  Class {c}: adaptive_dim={sub_adaptive['sub_dim_actual'][c]}")

    _pred, neg_resid = classify_by_residual(Z, sub_adaptive, num_classes)
    return -neg_resid, sub_adaptive



# ============================================================
# Part 3: RUN ALL CANDIDATES on cornell (smooth + raw)
# ============================================================
print("\n" + "="*60)
print("PART 3: EVALUATE ALL CANDIDATES")
print("="*60)

all_results = []

# --- Baselines ---
print("\n--- Baselines (smooth, lam=1.0, sub_dim=12) ---")
all_results.append(eval_scores(R_plain, Y, train_idx, val_idx, test_idx, "plain_residual_smooth"))
all_results.append(eval_scores(R_supp, Y, train_idx, val_idx, test_idx, "suppression_smooth"))

# --- Also test on raw features ---
print("\n--- Baselines (raw features) ---")
sub_raw = build_subspaces_src(X, Y, train_idx, num_classes, sub_dim)
_pred_raw, neg_resid_raw = classify_by_residual(X, sub_raw, num_classes)
R_plain_raw = -neg_resid_raw
all_results.append(eval_scores(R_plain_raw, Y, train_idx, val_idx, test_idx, "plain_residual_raw"))

# --- New algorithms on SMOOTH features ---
print("\n--- New algorithms (smooth, lam=1.0) ---")
for alpha in [0.3, 0.5, 0.7, 1.0]:
    R_fisher = fisher_weighted_residual(Z, sub, num_classes, train_idx, alpha=alpha)
    all_results.append(eval_scores(R_fisher, Y, train_idx, val_idx, test_idx, f"fisher_smooth_a{alpha}"))

R_ppca = ppca_scoring(Z, sub, num_classes, train_idx)
all_results.append(eval_scores(R_ppca, Y, train_idx, val_idx, test_idx, "ppca_smooth"))

for beta in [0.5, 1.0, 2.0, 5.0]:
    R_promo = promotion_scoring(Z, sub, num_classes, beta=beta)
    all_results.append(eval_scores(R_promo, Y, train_idx, val_idx, test_idx, f"promotion_smooth_b{beta}"))

R_adapt, sub_adapt = adaptive_subdim_residual(Z, Y, train_idx, num_classes, max_dim=32, gap_ratio=0.5)
all_results.append(eval_scores(R_adapt, Y, train_idx, val_idx, test_idx, "adaptive_dim_smooth"))

# --- New algorithms on RAW features ---
print("\n--- New algorithms (raw features) ---")
for alpha in [0.3, 0.5, 0.7, 1.0]:
    R_fisher_r = fisher_weighted_residual(X, sub_raw, num_classes, train_idx, alpha=alpha)
    all_results.append(eval_scores(R_fisher_r, Y, train_idx, val_idx, test_idx, f"fisher_raw_a{alpha}"))

R_ppca_r = ppca_scoring(X, sub_raw, num_classes, train_idx)
all_results.append(eval_scores(R_ppca_r, Y, train_idx, val_idx, test_idx, "ppca_raw"))

for beta in [0.5, 1.0, 2.0, 5.0]:
    R_promo_r = promotion_scoring(X, sub_raw, num_classes, beta=beta)
    all_results.append(eval_scores(R_promo_r, Y, train_idx, val_idx, test_idx, f"promotion_raw_b{beta}"))

R_adapt_r, sub_adapt_r = adaptive_subdim_residual(X, Y, train_idx, num_classes, max_dim=32, gap_ratio=0.5)
all_results.append(eval_scores(R_adapt_r, Y, train_idx, val_idx, test_idx, "adaptive_dim_raw"))

# --- Algorithm 5: PPCA + Fisher hybrid ---
# Combine PPCA's noise-aware scoring with Fisher weighting
print("\n--- Hybrid: PPCA + Fisher (smooth and raw) ---")

def ppca_fisher_hybrid(Z, sub, num_classes, train_idx, alpha=0.5, noise_floor=1e-4):
    """
    PPCA scoring with Fisher-weighted within-subspace Mahalanobis.
    Instead of uniform Mahalanobis, weight each direction by Fisher ratio.
    """
    n, D = Z.shape
    R = torch.zeros(n, num_classes)
    tr_t = torch.tensor(train_idx, dtype=torch.long)
    mu_global = Z[tr_t].mean(dim=0)

    for c in range(num_classes):
        B_c = sub['B_dict'][c]
        mu_c = sub['mu_dict'][c]
        S_c = sub['S_dict'][c]
        n_c = sub['n_c_dict'][c]
        d_c = B_c.shape[1]

        sigma_sq = (S_c ** 2) / max(n_c - 1, 1)
        sigma_sq = sigma_sq.clamp(min=noise_floor)

        # Fisher weights
        within_var = sigma_sq
        between_var = torch.zeros(d_c)
        for c2 in range(num_classes):
            if c2 == c or c2 not in sub['mu_dict']:
                continue
            n_c2 = sub['n_c_dict'][c2]
            proj = B_c.t() @ (sub['mu_dict'][c2] - mu_global)
            between_var += n_c2 * proj ** 2
        fisher = between_var / (within_var + 1e-8)
        f_max = fisher.max().clamp(min=1e-8)
        fisher_w = (1 - alpha) + alpha * (fisher / f_max)

        # Noise variance
        mask_c = (Y[tr_t] == c)
        diff_train = Z[tr_t][mask_c] - mu_c
        total_var = (diff_train ** 2).sum().item() / max(n_c - 1, 1)
        explained_var = sigma_sq.sum().item()
        remaining_dims = max(D - d_c, 1)
        sigma_noise_sq = max((total_var - explained_var) / remaining_dims, noise_floor)

        diff = Z - mu_c
        proj = diff @ B_c
        # Fisher-weighted Mahalanobis
        maha = (proj ** 2 * fisher_w.unsqueeze(0) / sigma_sq.unsqueeze(0)).sum(dim=1)
        proj_back = proj @ B_c.t()
        ortho_sq = ((diff - proj_back) ** 2).sum(dim=1) / sigma_noise_sq
        log_det = sigma_sq.log().sum() + remaining_dims * np.log(sigma_noise_sq)

        R[:, c] = maha + ortho_sq + log_det

    return R

for alpha in [0.3, 0.5, 0.7]:
    R_pf = ppca_fisher_hybrid(Z, sub, num_classes, train_idx, alpha=alpha)
    all_results.append(eval_scores(R_pf, Y, train_idx, val_idx, test_idx, f"ppca_fisher_smooth_a{alpha}"))
    R_pf_r = ppca_fisher_hybrid(X, sub_raw, num_classes, train_idx, alpha=alpha)
    all_results.append(eval_scores(R_pf_r, Y, train_idx, val_idx, test_idx, f"ppca_fisher_raw_a{alpha}"))

# --- Algorithm 6: Dual-channel best-of with new scorers ---
print("\n--- Dual-channel: pick best of smooth vs raw per algorithm ---")
# For each new algorithm, pick the better channel by val_acc
va_t = torch.tensor(val_idx, dtype=torch.long)
for r in all_results:
    pass  # already have val_acc

# --- Summary ---
print("\n" + "="*60)
print("SUMMARY (sorted by test_acc)")
print("="*60)
all_results.sort(key=lambda x: x['test_acc'], reverse=True)
for i, r in enumerate(all_results):
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {r['test_acc']:.4f} val={r['val_acc']:.4f} {r['name']}{marker}")

# Save results
out_path = os.path.join(RESULTS_DIR, "cornell_split0_candidates.json")
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")

# ============================================================
# Part 4: Multi-lambda grid for best candidates
# ============================================================
print("\n" + "="*60)
print("PART 4: LAMBDA GRID FOR TOP CANDIDATES")
print("="*60)

# Find top 3 smooth-based algorithms by val_acc
smooth_results = [r for r in all_results if 'smooth' in r['name']]
smooth_results.sort(key=lambda x: x['val_acc'], reverse=True)
top_smooth_names = []
for r in smooth_results[:5]:
    # Extract algorithm type
    name = r['name']
    if name.startswith('fisher_smooth'):
        top_smooth_names.append(('fisher', float(name.split('_a')[1]) if '_a' in name else 0.5))
    elif name.startswith('ppca_fisher_smooth'):
        top_smooth_names.append(('ppca_fisher', float(name.split('_a')[1]) if '_a' in name else 0.5))
    elif name == 'ppca_smooth':
        top_smooth_names.append(('ppca', 0))
    elif name.startswith('promotion_smooth'):
        top_smooth_names.append(('promotion', float(name.split('_b')[1]) if '_b' in name else 1.0))

print(f"Top smooth candidates: {top_smooth_names}")

lam_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
dim_grid = [8, 12, 16, 24]

grid_results = []
for lam_val in lam_grid:
    Z_g = tikhonov_smooth_src(X, L, lam_val)
    for sd in dim_grid:
        sub_g = build_subspaces_src(Z_g, Y, train_idx, num_classes, sd)
        sub_raw_g = build_subspaces_src(X, Y, train_idx, num_classes, sd)

        # Plain residual baseline
        _p, nr = classify_by_residual(Z_g, sub_g, num_classes)
        R_base = -nr
        res = eval_scores(R_base, Y, train_idx, val_idx, test_idx,
                         f"plain_smooth_lam{lam_val}_d{sd}")
        res['lam'] = lam_val
        res['sub_dim'] = sd
        grid_results.append(res)

        # Plain raw
        _p, nr = classify_by_residual(X, sub_raw_g, num_classes)
        R_base_r = -nr
        res = eval_scores(R_base_r, Y, train_idx, val_idx, test_idx,
                         f"plain_raw_d{sd}")
        res['lam'] = 0
        res['sub_dim'] = sd
        grid_results.append(res)

        # Top candidates
        for algo_name, param in top_smooth_names:
            if algo_name == 'fisher':
                R_g = fisher_weighted_residual(Z_g, sub_g, num_classes, train_idx, alpha=param)
                res = eval_scores(R_g, Y, train_idx, val_idx, test_idx,
                                 f"fisher_smooth_a{param}_lam{lam_val}_d{sd}")
                # Also raw
                R_gr = fisher_weighted_residual(X, sub_raw_g, num_classes, train_idx, alpha=param)
                res_r = eval_scores(R_gr, Y, train_idx, val_idx, test_idx,
                                   f"fisher_raw_a{param}_d{sd}")
            elif algo_name == 'ppca':
                R_g = ppca_scoring(Z_g, sub_g, num_classes, train_idx)
                res = eval_scores(R_g, Y, train_idx, val_idx, test_idx,
                                 f"ppca_smooth_lam{lam_val}_d{sd}")
                R_gr = ppca_scoring(X, sub_raw_g, num_classes, train_idx)
                res_r = eval_scores(R_gr, Y, train_idx, val_idx, test_idx,
                                   f"ppca_raw_d{sd}")
            elif algo_name == 'ppca_fisher':
                R_g = ppca_fisher_hybrid(Z_g, sub_g, num_classes, train_idx, alpha=param)
                res = eval_scores(R_g, Y, train_idx, val_idx, test_idx,
                                 f"ppca_fisher_smooth_a{param}_lam{lam_val}_d{sd}")
                R_gr = ppca_fisher_hybrid(X, sub_raw_g, num_classes, train_idx, alpha=param)
                res_r = eval_scores(R_gr, Y, train_idx, val_idx, test_idx,
                                   f"ppca_fisher_raw_a{param}_d{sd}")
            elif algo_name == 'promotion':
                R_g = promotion_scoring(Z_g, sub_g, num_classes, beta=param)
                res = eval_scores(R_g, Y, train_idx, val_idx, test_idx,
                                 f"promotion_smooth_b{param}_lam{lam_val}_d{sd}")
                R_gr = promotion_scoring(X, sub_raw_g, num_classes, beta=param)
                res_r = eval_scores(R_gr, Y, train_idx, val_idx, test_idx,
                                   f"promotion_raw_b{param}_d{sd}")
            else:
                continue
            res['lam'] = lam_val
            res['sub_dim'] = sd
            grid_results.append(res)
            res_r['lam'] = 0
            res_r['sub_dim'] = sd
            grid_results.append(res_r)

# Summary of grid search
print("\n" + "="*60)
print("GRID SEARCH SUMMARY (top 20 by val_acc)")
print("="*60)
grid_results.sort(key=lambda x: x['val_acc'], reverse=True)
for i, r in enumerate(grid_results[:20]):
    print(f"  val={r['val_acc']:.4f} test={r['test_acc']:.4f} {r['name']}")

# Save grid results
out_path2 = os.path.join(RESULTS_DIR, "cornell_split0_grid.json")
with open(out_path2, 'w') as f:
    json.dump(grid_results, f, indent=2)
print(f"\nSaved to {out_path2}")

print("\n=== DONE ===")
