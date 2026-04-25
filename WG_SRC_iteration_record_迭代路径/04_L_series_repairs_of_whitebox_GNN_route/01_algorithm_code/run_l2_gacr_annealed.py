"""
L2-A Innovation: GACR + Tau Annealing + delta_R Normalization
Strategy: Minimal diff from Agent4 (proven val=0.6933, test=0.6920).
Only change: replace standard coding_rate with graph_aware_coding_rate in hop weighting.
All other mechanisms identical to Agent4.

This avoids regression: Agent4 architecture is validated, we only upgrade the CR function.

Key parameters (same as Agent4 except hidden_dim/subspace_dim):
  - hidden_dim=64, subspace_dim=16  (from spec)
  - eps=0.5  (Agent4 uses 0.5, stable)
  - eta=0.5  (Agent4 proven)
  - cosine tau annealing
  - delta_R normalization (of GACR delta_R)
  - hop diversity loss
"""

import os, sys, pickle, time, math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR    = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR     = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "l2_gacr_ann_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "l2_gacr_ann_result.txt")

output_lines = []

def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_cora(data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs  = []
    for n in names:
        with open(f"{data_dir}/ind.cora.{n}", 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_raw    = [int(l.strip()) for l in open(f"{data_dir}/ind.cora.test.index")]
    test_idx_sorted = sorted(test_idx_raw)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_raw, :] = features[test_idx_sorted, :]
    features = np.array(features.todense(), dtype=np.float32)
    labels = np.vstack((ally, ty))
    labels[test_idx_raw, :] = labels[test_idx_sorted, :]
    labels = np.argmax(labels, axis=1)
    n = features.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0
            adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    adj_norm   = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    lap        = np.eye(n, dtype=np.float32) - adj_norm
    train_idx  = list(range(140))
    val_idx    = list(range(200, 500))
    test_idx   = test_idx_sorted[:1000]
    return features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx

# ── Coding Rates ─────────────────────────────────────────────────────────────

def coding_rate(Z, eps=0.5):
    """Standard R(Z) = 0.5 * log det(I + d/(n*eps^2) * Z^T Z)"""
    n, d  = Z.shape
    coeff = d / (n * eps * eps)
    M     = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    _, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

def coding_rate_gradient(Z, eps=0.5):
    """dR/dZ = coeff * Z @ (I + coeff * Z^T Z)^{-1}"""
    n, d  = Z.shape
    coeff = d / (n * eps * eps)
    M     = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    M_inv = torch.linalg.inv(M)
    return coeff * (Z @ M_inv)

def graph_aware_coding_rate(Z, L, alpha, eps=0.5):
    """
    R_graph(Z) = 0.5 * log det(I + coeff * Z^T (I + alpha*L) Z)
    Graph-smoothed coding rate: higher R_graph => smooth on graph AND information-rich.
    Used ONLY for hop weight computation (not for gradient step).
    """
    n, d       = Z.shape
    coeff      = d / (n * eps * eps)
    smoothed_Z = Z + alpha * (L @ Z)
    M          = torch.eye(d, device=Z.device) + coeff * (Z.t() @ smoothed_Z)
    _, logdet  = torch.linalg.slogdet(M)
    return 0.5 * logdet

# ── Cosine Tau Annealing (Agent4) ────────────────────────────────────────────

def cosine_tau(epoch, T, tau_min=0.1, tau_max=1.0):
    """tau: large early (exploration) -> small late (exploitation)"""
    return tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(math.pi * epoch / T))

# ── Hop Diversity Loss (Agent4) ──────────────────────────────────────────────

def hop_diversity_loss(w):
    """Penalize uniform hop weights: loss = -var(w)"""
    return -w.var()

# ── Core Layer: GACR weighting + Agent4 gradient step ────────────────────────

class GACRAnnealedHopLayer(nn.Module):
    """
    Hybrid design:
      - Hop WEIGHT: graph_aware_coding_rate (GACR) with delta_R normalization + cosine tau
      - Gradient STEP: standard coding_rate_gradient (proven stable in Agent4)
      - This separates the "which hop is informative" (GACR) from "how to update" (std CR gradient)

    Architecture identical to Agent4 CRGainHopLayerAnnealed, except:
      - hop weight uses GACR delta_R (not standard CR delta_R)
      - adds log_alpha parameter for graph smoothing weight
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 log_alpha_init=-2.0):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.subspace_dim  = subspace_dim
        self.num_hops      = num_hops
        self.eta           = eta
        self.eps           = eps
        self.lambda_lap    = lambda_lap
        self.lambda_sparse = lambda_sparse

        # Learnable graph smoothing weight: alpha = exp(log_alpha) > 0
        self.log_alpha = nn.Parameter(torch.tensor(log_alpha_init))

        # Per-hop projection matrices W_k: in_dim -> subspace_dim
        self.W = nn.ModuleList([
            nn.Linear(in_dim, subspace_dim, bias=False)
            for _ in range(num_hops + 1)
        ])

        # Output projection: in_dim -> out_dim
        self.out_proj = nn.Linear(in_dim, out_dim, bias=False)

        # Learnable soft-threshold
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))

        # LayerNorm
        self.ln = nn.LayerNorm(out_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, H, hop_feats, L, tau, return_H_half=False):
        """
        H:            [n, in_dim]
        hop_feats:    list of K+1 tensors [n, in_dim]
        L:            [n, n] normalized Laplacian
        tau:          scalar float, cosine annealed
        """
        alpha = self.alpha
        K     = self.num_hops

        # Step 1: project each hop into subspace
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]

        # Step 2: GACR per hop -> marginal gain -> normalize -> softmax/tau
        #   GACR delta_R measures "graph-smoothness-aware information gain" of each hop
        R_gacr_list  = [graph_aware_coding_rate(Z_list[k], L, alpha, self.eps)
                        for k in range(K + 1)]
        dR_gacr_list = [R_gacr_list[0]] + [R_gacr_list[k] - R_gacr_list[k-1]
                                            for k in range(1, K+1)]
        dR_gacr      = torch.stack(dR_gacr_list)   # [K+1]

        # Normalize to amplify relative differences (fix degeneration to uniform)
        dR_mean      = dR_gacr.mean()
        dR_std       = dR_gacr.std() + 1e-6
        dR_norm      = (dR_gacr - dR_mean) / dR_std   # [K+1]

        # Hop weights: w retains grad for diversity loss
        w = F.softmax(dR_norm / tau, dim=0)   # [K+1]

        # Step 3: STANDARD gradient step (proven stable, same as Agent4)
        #   Use standard CR gradient (not GACR gradient) to avoid scale issues
        grad_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            g_Zk         = coding_rate_gradient(Z_list[k], self.eps)   # [n, subspace_dim]
            g_H          = g_Zk @ self.W[k].weight                     # [n, in_dim]
            grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # Step 4: dimension alignment
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 5: proximal operator — soft threshold + LayerNorm
        thr    = self.threshold.abs().unsqueeze(0)
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        if return_H_half:
            return H_out, w, dR_norm.detach(), dR_gacr.detach(), H_half.detach()
        return H_out, w, dR_norm.detach(), dR_gacr.detach()


# ── Full Model ───────────────────────────────────────────────────────────────

class GACRAnnealedGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05,
                 dropout=0.6, log_alpha_init=-2.0):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = GACRAnnealedHopLayer(
            in_dim, hidden_dim, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, log_alpha_init
        )
        self.layer2 = GACRAnnealedHopLayer(
            hidden_dim, num_classes, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, log_alpha_init
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        hops = [H]
        cur  = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L, tau, return_H_half=False):
        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        if return_H_half:
            H1, w1, dR1n, dR1r, Hh1 = self.layer1(H, hops1, L, tau, return_H_half=True)
        else:
            H1, w1, dR1n, dR1r = self.layer1(H, hops1, L, tau)
            Hh1 = None
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        if return_H_half:
            H2, w2, dR2n, dR2r, Hh2 = self.layer2(H1, hops2, L, tau, return_H_half=True)
        else:
            H2, w2, dR2n, dR2r = self.layer2(H1, hops2, L, tau)
            Hh2 = None

        if return_H_half:
            return H2, (w1, w2), (dR1n, dR2n), (dR1r, dR2r), (Hh1, Hh2)
        return H2, (w1, w2), (dR1n, dR2n), (dR1r, dR2r)


# ── Losses (train nodes only — strict no label leakage) ──────────────────────

def mcr2_loss(Z, y, num_classes, eps=0.5):
    """MCR2: maximize Delta_R = R(Z) - mean_k R(Z_k), train nodes only"""
    R_total     = coding_rate(Z, eps)
    R_class_sum = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        R_class_sum = R_class_sum + coding_rate(Z[mask], eps)
    return -(R_total - R_class_sum / num_classes)

def orth_loss(Z, y, num_classes):
    """Orthogonality loss, train nodes only"""
    means = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 1:
            means.append(torch.zeros(Z.shape[1], device=Z.device))
        else:
            means.append(Z[mask].mean(0))
    M    = torch.stack(means)
    M    = F.normalize(M, dim=1)
    gram = M @ M.t()
    eye  = torch.eye(num_classes, device=Z.device)
    return (gram - eye).pow(2).sum() / (num_classes * num_classes)


# ── Training ─────────────────────────────────────────────────────────────────

def run_experiment():
    cfg = dict(
        hidden_dim    = 32,     # same as Agent4: avoids overfitting (only 140 train nodes)
        subspace_dim  = 8,
        eps           = 0.5,
        num_hops      = 2,
        eta           = 0.5,
        lambda_lap    = 0.3,
        lambda_sparse = 0.05,
        lambda_mcr    = 0.005,
        lambda_orth   = 0.005,
        lambda_div    = 0.01,
        log_alpha_init= -2.0,
        tau_min       = 0.1,
        tau_max       = 1.0,
        dropout       = 0.6,
        lr            = 0.005,
        wd            = 1e-3,
        epochs        = 200,
        patience      = 50,
        seed          = 42,
    )

    log("=" * 60)
    log("Experiment: L2-A GACR + Cosine Tau Annealing + delta_R Norm")
    log(f"  Design: GACR hop-weighting + std CR gradient (hybrid)")
    log(f"Config: {cfg}")
    log("=" * 60)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim   = features.shape
    num_classes = int(labels.max()) + 1
    log(f"Data: n={n}, in_dim={in_dim}, num_classes={num_classes}")
    log(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    device = torch.device('cpu')
    X      = torch.tensor(features,  dtype=torch.float32, device=device)
    Y      = torch.tensor(labels,    dtype=torch.long,    device=device)
    A_norm = torch.tensor(adj_norm,  dtype=torch.float32, device=device)
    L_mat  = torch.tensor(lap,       dtype=torch.float32, device=device)

    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask   = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    model = GACRAnnealedGNN(
        in_dim        = in_dim,
        hidden_dim    = cfg['hidden_dim'],
        num_classes   = num_classes,
        subspace_dim  = cfg['subspace_dim'],
        num_hops      = cfg['num_hops'],
        eta           = cfg['eta'],
        eps           = cfg['eps'],
        lambda_lap    = cfg['lambda_lap'],
        lambda_sparse = cfg['lambda_sparse'],
        dropout       = cfg['dropout'],
        log_alpha_init= cfg['log_alpha_init'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0
    T             = cfg['epochs']

    t0 = time.time()

    for epoch in range(1, T + 1):
        tau = cosine_tau(epoch - 1, T, cfg['tau_min'], cfg['tau_max'])

        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (dR1n, dR2n), (dR1r, dR2r) = model(X, A_norm, L_mat, tau)

        # All losses strictly train nodes only — no label leakage
        ce   = F.cross_entropy(logits[train_mask], Y[train_mask])
        mcr  = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)
        div  = hop_diversity_loss(w1) + hop_diversity_loss(w2)

        loss = (ce
                + cfg['lambda_mcr']  * mcr
                + cfg['lambda_orth'] * orth
                + cfg['lambda_div']  * div)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X, A_norm, L_mat, tau, return_H_half=True)
            logits_e, (w1e, w2e), (dR1ne, dR2ne), (dR1re, dR2re), (Hh1, Hh2) = out
            pred     = logits_e.argmax(dim=1)
            val_acc  = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == Y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            alpha1   = model.layer1.alpha.item()
            alpha2   = model.layer2.alpha.item()
            hh1_norm = Hh1.norm().item()
            hh2_norm = Hh2.norm().item()

            w1_str   = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            w2_str   = '[' + ', '.join(f'{v:.3f}' for v in w2e.tolist()) + ']'
            dR1n_str = '[' + ', '.join(f'{v:.3f}' for v in dR1ne.tolist()) + ']'
            dR2n_str = '[' + ', '.join(f'{v:.3f}' for v in dR2ne.tolist()) + ']'
            dR1r_str = '[' + ', '.join(f'{v:.5f}' for v in dR1re.tolist()) + ']'
            dR2r_str = '[' + ', '.join(f'{v:.5f}' for v in dR2re.tolist()) + ']'
            w1_var   = w1e.var().item()
            w2_var   = w2e.var().item()

            log(f"Epoch {epoch:4d} | tau={tau:.4f} | loss={loss.item():.4f} | "
                f"val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  [L2-A] Layer1: alpha={alpha1:.4f} | H_half_norm={hh1_norm:.2f}")
            log(f"         hop_w={w1_str} | var={w1_var:.5f}")
            log(f"         delta_R_norm={dR1n_str}")
            log(f"         delta_R_raw(GACR)={dR1r_str}")
            log(f"  [L2-A] Layer2: alpha={alpha2:.4f} | H_half_norm={hh2_norm:.2f}")
            log(f"         hop_w={w2_str} | var={w2_var:.5f}")
            log(f"         delta_R_norm={dR2n_str}")
            log(f"         delta_R_raw(GACR)={dR2r_str}")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log("=" * 60)

    g1_val,  g1_test  = 0.7033, 0.6800
    a4_val,  a4_test  = 0.6933, 0.6920
    a5_val,  a5_test  = 0.7067, 0.6870
    log("")
    log("Comparison with baselines:")
    log(f"  G1   baseline : val={g1_val:.4f}, test={g1_test:.4f}")
    log(f"  Agent4 (tau)  : val={a4_val:.4f}, test={a4_test:.4f}")
    log(f"  Agent5 (GACR) : val={a5_val:.4f}, test={a5_test:.4f}")
    log(f"  L2-A (ours)   : val={best_val_acc:.4f}, test={best_test_acc:.4f}")
    log(f"  vs G1   => val {best_val_acc - g1_val:+.4f} | test {best_test_acc - g1_test:+.4f}")
    log(f"  vs A4   => val {best_val_acc - a4_val:+.4f} | test {best_test_acc - a4_test:+.4f}")
    log(f"  vs A5   => val {best_val_acc - a5_val:+.4f} | test {best_test_acc - a5_test:+.4f}")

    return best_val_acc, best_test_acc


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L2-A Innovation: GACR + Cosine Tau Annealing + delta_R Normalization")
    log(f"OUT_DIR: {OUT_DIR}")
    log(f"Output will be saved to: {OUTPUT_PATH}")
    log("")

    best_val, best_test = run_experiment()

    result_lines = [
        "L2-A GACR+Annealed Result",
        "=" * 40,
        "Config: l2_gacr_annealed (hybrid: GACR weighting + std CR gradient)",
        "  hidden_dim=32, subspace_dim=8, num_hops=2",
        "  eps=0.5, eta=0.5, lambda_lap=0.3 (same as Agent4 proven values)",
        "  epochs=200, patience=50",
        "  tau_min=0.1, tau_max=1.0 (cosine annealing)",
        "  lambda_div=0.01 (hop diversity regularization)",
        "  delta_R normalization: ON (GACR delta_R)",
        "  graph-aware coding rate: ON for hop weights (alpha learnable)",
        "-" * 40,
        f"Best Val  Acc: {best_val:.4f}",
        f"Best Test Acc: {best_test:.4f}",
        "-" * 40,
        "G1   Baseline: val=0.7033, test=0.6800",
        "Agent4 (tau) : val=0.6933, test=0.6920",
        "Agent5 (GACR): val=0.7067, test=0.6870",
        f"Delta vs G1  : val={best_val - 0.7033:+.4f}, test={best_test - 0.6800:+.4f}",
        f"Delta vs A4  : val={best_val - 0.6933:+.4f}, test={best_test - 0.6920:+.4f}",
        f"Delta vs A5  : val={best_val - 0.7067:+.4f}, test={best_test - 0.6870:+.4f}",
        "=" * 40,
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))

    save_output()
    log(f"Output saved to {OUTPUT_PATH}")
    log(f"Result saved to {RESULT_PATH}")
    log("Done.")


