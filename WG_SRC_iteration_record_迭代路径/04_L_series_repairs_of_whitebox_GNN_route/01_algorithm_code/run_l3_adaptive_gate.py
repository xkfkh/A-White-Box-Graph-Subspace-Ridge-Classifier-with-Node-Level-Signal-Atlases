"""
L3 Innovation: Adaptive Gate for Hop Weighting (FINAL)

Key insight from v1/v2 debugging:
  - Using gate weights inside coding_rate_gradient step is ineffective:
    CR gradient is tiny (coeff~0.012), so d_loss/d_w ~ 0, gate never learns.
  - Solution: gate weights applied to DIRECT hop aggregation (like weighted APPNP),
    then feed into white-box proximal structure (soft-threshold + LayerNorm).
  - CR gradient step retained but with FIXED equal weights (interpretability preserved).
  - Gate learns from strong CE loss gradient directly.

Architecture per layer:
  1. Compute hop features: H_k = A^k H  for k=0..K
  2. Project each hop: Z_k = W_k(H_k)  [n, hidden]
  3. Gate MLP on hop statistics (lap_quad from Z_k) -> softmax -> w_k
  4. Weighted aggregation: H_agg = sum_k w_k * Z_k  [n, hidden]
  5. CR gradient regularization (fixed equal weights): grad_contrib
  6. H_half = H_agg + eta * grad_contrib - eta * lambda_lap * L @ H_agg
  7. Soft-threshold + LayerNorm -> H_out

Config: hidden_dim=64, subspace_dim=16 (=hidden in layer1), num_hops=2,
        dropout=0.6, lr=0.005, wd=1e-3, epochs=300, patience=50, seed=42
"""

import os, sys, pickle, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "l3_gate_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "l3_gate_result.txt")

output_lines = []

def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

# ── Data Loading ────────────────────────────────────────────────────────────────

def load_cora(data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs = []
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
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    lap = np.eye(n, dtype=np.float32) - adj_norm
    train_idx = list(range(140))
    val_idx   = list(range(200, 500))
    test_idx  = test_idx_sorted[:1000]
    return features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx

# ── Coding Rate Utilities ───────────────────────────────────────────────────────

def coding_rate(Z, eps=0.5):
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    sign, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

def coding_rate_gradient(Z, eps=0.5):
    """dR/dZ = coeff * Z @ (I + coeff * Z^T Z)^{-1}"""
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    M_inv = torch.linalg.inv(M)
    return coeff * (Z @ M_inv)

# ── MCR2 / Orth Losses ─────────────────────────────────────────────────────────

def mcr2_loss(Z, y, num_classes, eps=0.5):
    R_total = coding_rate(Z, eps)
    R_class_sum = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        R_class_sum = R_class_sum + coding_rate(Z[mask], eps)
    return -(R_total - R_class_sum / num_classes)

def orth_loss(Z, y, num_classes):
    means = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 1:
            means.append(torch.zeros(Z.shape[1], device=Z.device))
        else:
            means.append(Z[mask].mean(0))
    M = F.normalize(torch.stack(means), dim=1)
    gram = M @ M.t()
    eye  = torch.eye(num_classes, device=Z.device)
    return (gram - eye).pow(2).sum() / (num_classes * num_classes)

# ── Adaptive Gate Hop Layer (FINAL) ────────────────────────────────────────────

class AdaptiveGateHopLayer(nn.Module):
    """
    Semi-white-box GNN layer (FINAL design):

    Gate weights applied to DIRECT hop aggregation:
      H_agg = sum_k w_k * (W_k @ H_k)
    where w_k = softmax(gate_MLP(stats(Z_k)))

    This gives gate strong gradient signal from CE loss directly.
    CR gradient step applied AFTER aggregation for proximal regularization.

    Gate features per hop (computed on Z_k = W_k @ H_k, dim=out_dim):
      - mean_norm: avg per-node norm (captures representation scale)
      - std_norm:  std of per-node norms (captures inter-node diversity)
      - lap_quad:  tr(Z^T L Z)/(n*d) (captures smoothness on graph)
      - rank_proxy: nuclear norm / (n * sqrt(d)) (proxy for rank/diversity)
    """
    def __init__(self, in_dim, out_dim, num_hops=2,
                 eta=0.1, eps=0.5, lambda_lap=0.1, lambda_sparse=0.05):
        super().__init__()
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.num_hops    = num_hops
        self.eta         = eta
        self.eps         = eps
        self.lambda_lap  = lambda_lap

        K = num_hops + 1

        # Per-hop projection: in_dim -> out_dim
        self.W = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(K)
        ])

        # Gate MLP: 4K stats -> K weights
        gate_in = K * 4
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_in * 2),
            nn.GELU(),
            nn.Linear(gate_in * 2, K)
        )

        # Learnable soft-threshold
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))

        # LayerNorm
        self.ln = nn.LayerNorm(out_dim)

    def _hop_stats(self, Zk, L):
        """
        4 statistics on projected hop features Zk [n, d]:
          - mean_norm: mean of per-node L2 norm
          - std_norm:  std of per-node L2 norm
          - lap_quad:  tr(Z^T L Z) / (n*d)
          - rank_proxy: sum of singular values / (n * sqrt(d))
        """
        node_norms = Zk.norm(dim=1)           # [n]
        mean_norm  = node_norms.mean()
        std_norm   = node_norms.std().clamp(min=1e-6)

        LZk = L @ Zk                          # [n, d]
        lap_quad = (LZk * Zk).sum() / (Zk.shape[0] * Zk.shape[1])

        # Rank proxy via Frobenius norm (faster than SVD)
        # ||Z||_F / (n * sqrt(d)) ~ avg singular value proxy
        rank_proxy = Zk.norm() / (Zk.shape[0] ** 0.5 * Zk.shape[1] ** 0.5)

        return mean_norm, std_norm, lap_quad, rank_proxy

    def forward(self, H, hop_feats, L):
        """
        H:         [n, in_dim]
        hop_feats: list of K tensors [n, in_dim]
        L:         [n, n] normalized Laplacian
        Returns:   H_out [n, out_dim], w [K] hop weights
        """
        K = self.num_hops + 1

        # ── Step 1: project each hop ──
        Z_list = [self.W[k](hop_feats[k]) for k in range(K)]  # each [n, out_dim]

        # ── Step 2: compute gate statistics on projected features ──
        stats = []
        raw_stats = []
        for k in range(K):
            s = self._hop_stats(Z_list[k], L)
            raw_stats.append(s)
            stats.extend(list(s))

        gate_input_raw = torch.stack(stats)  # [4K]

        # Normalize to stabilize gate training
        g_mean = gate_input_raw.mean()
        g_std  = gate_input_raw.std().clamp(min=1e-6)
        gate_input = (gate_input_raw - g_mean) / g_std

        raw_w = self.gate(gate_input.unsqueeze(0)).squeeze(0)  # [K]
        w     = F.softmax(raw_w, dim=0)                        # [K]

        # ── Step 3: gate-weighted aggregation (STRONG gradient path) ──
        # H_agg = sum_k w_k * Z_k  [n, out_dim]
        # Gradient: d_CE/d_w_k = d_CE/d_H_agg * Z_k  (direct, large)
        H_agg = sum(w[k] * Z_list[k] for k in range(K))

        # ── Step 4: CR gradient regularization on aggregated rep ──
        # Applied with small eta to not overshadow the learned aggregation
        g_agg = coding_rate_gradient(H_agg, self.eps)  # [n, out_dim]
        H_half = H_agg + self.eta * g_agg - self.eta * self.lambda_lap * (L @ H_agg)

        # ── Step 5: proximal operator — soft threshold + LayerNorm ──
        thr    = self.threshold.abs().unsqueeze(0)
        H_soft = H_half.sign() * F.relu(H_half.abs() - thr)
        H_out  = self.ln(H_soft)

        lap_quads = torch.stack([s[2] for s in raw_stats])  # [K]
        return H_out, w.detach(), lap_quads.detach()

# ── Full Model ──────────────────────────────────────────────────────────────────

class AdaptiveGateGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_hops=2,
                 eta=0.1, eps=0.5, lambda_lap=0.1, lambda_sparse=0.05, dropout=0.6):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = AdaptiveGateHopLayer(
            in_dim, hidden_dim, num_hops, eta, eps, lambda_lap, lambda_sparse
        )
        self.layer2 = AdaptiveGateHopLayer(
            hidden_dim, num_classes, num_hops, eta, eps, lambda_lap, lambda_sparse
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        hops = [H]
        cur = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L):
        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        H1, w1, lq1 = self.layer1(H, hops1, L)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        H2, w2, lq2 = self.layer2(H1, hops2, L)

        return H2, (w1, w2), (lq1, lq2)

# ── Training ────────────────────────────────────────────────────────────────────

def run_experiment():
    cfg = dict(
        hidden_dim    = 64,
        num_hops      = 2,
        eta           = 0.1,
        eps           = 0.5,
        lambda_lap    = 0.1,
        lambda_sparse = 0.05,
        lambda_mcr    = 0.005,
        lambda_orth   = 0.005,
        dropout       = 0.6,
        lr            = 0.005,
        wd            = 1e-3,
        epochs        = 300,
        patience      = 50,
        seed          = 42,
    )

    log("=" * 60)
    log("Experiment: L3 Adaptive Gate GNN (FINAL)")
    log("  Gate weights on DIRECT hop aggregation (Z_k = W_k @ H_k)")
    log("  Strong CE gradient -> gate; CR gradient step for regularization")
    log("  Gate features: mean_norm, std_norm, lap_quad, rank_proxy on Z_k")
    log(f"Config: {cfg}")
    log("=" * 60)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim = features.shape
    num_classes = int(labels.max()) + 1
    log(f"Data: n={n}, in_dim={in_dim}, num_classes={num_classes}")
    log(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    K = cfg['num_hops'] + 1
    gate_in = K * 4
    log(f"Gate MLP: in={gate_in}, hidden={gate_in*2}, out={K}")
    log("")

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

    model = AdaptiveGateGNN(
        in_dim      = in_dim,
        hidden_dim  = cfg['hidden_dim'],
        num_classes = num_classes,
        num_hops    = cfg['num_hops'],
        eta         = cfg['eta'],
        eps         = cfg['eps'],
        lambda_lap  = cfg['lambda_lap'],
        lambda_sparse=cfg['lambda_sparse'],
        dropout     = cfg['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {total_params}")
    log("")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0

    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (lq1, lq2) = model(X, A_norm, L_mat)

        ce   = F.cross_entropy(logits[train_mask], Y[train_mask])
        mcr  = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        loss = ce + cfg['lambda_mcr'] * mcr + cfg['lambda_orth'] * orth
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_e, (w1e, w2e), (lq1e, lq2e) = model(X, A_norm, L_mat)
            pred     = logits_e.argmax(dim=1)
            val_acc  = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == Y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if epoch % 50 == 0:
            w1_str  = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            w2_str  = '[' + ', '.join(f'{v:.3f}' for v in w2e.tolist()) + ']'
            lq1_str = '[' + ', '.join(f'{v:.4f}' for v in lq1e.tolist()) + ']'
            lq2_str = '[' + ', '.join(f'{v:.4f}' for v in lq2e.tolist()) + ']'
            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  Layer1 hop_w={w1_str}  lap_quad={lq1_str}")
            log(f"  Layer2 hop_w={w2_str}  lap_quad={lq2_str}")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log("=" * 60)

    g1_val  = 0.7033
    g1_test = 0.6800
    val_diff  = best_val_acc  - g1_val
    test_diff = best_test_acc - g1_test
    log("")
    log("Comparison with G1 baseline (val=0.7033, test=0.6800):")
    log(f"  Val  diff: {val_diff:+.4f}  ({'better' if val_diff > 0 else 'worse'})")
    log(f"  Test diff: {test_diff:+.4f}  ({'better' if test_diff > 0 else 'worse'})")

    return best_val_acc, best_test_acc, val_diff, test_diff

# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L3 Innovation: Adaptive Gate for Hop Weighting (FINAL)")
    log(f"OUT_DIR: {OUT_DIR}")
    log(f"Output: {OUTPUT_PATH}")
    log(f"Result: {RESULT_PATH}")
    log("")

    best_val, best_test, val_diff, test_diff = run_experiment()

    save_output()

    g1_val  = 0.7033
    g1_test = 0.6800
    result_lines = [
        "L3 Adaptive Gate Result",
        "=" * 40,
        "Config: adaptive_gate",
        "  hidden_dim=64, subspace_dim=16, num_hops=2",
        "  Gate MLP: stats-driven hop weights",
        "  Gate applied to direct hop aggregation (Z_k=W_k@H_k)",
        "  CR gradient step retained for proximal regularization",
        "-" * 40,
        f"Best Val  Acc: {best_val:.4f}",
        f"Best Test Acc: {best_test:.4f}",
        "-" * 40,
        f"G1 Baseline: val={g1_val:.4f}, test={g1_test:.4f}",
        f"Delta Val:  {val_diff:+.4f}",
        f"Delta Test: {test_diff:+.4f}",
        "=" * 40,
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))

    log(f"Output saved to {OUTPUT_PATH}")
    log(f"Result saved to {RESULT_PATH}")
    log("Done.")


