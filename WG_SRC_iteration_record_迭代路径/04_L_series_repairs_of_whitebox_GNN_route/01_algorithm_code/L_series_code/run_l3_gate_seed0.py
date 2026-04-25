"""
L3 Innovation: Adaptive Gate for Hop Weighting (FINAL)
Stability test — seed=0
"""

import os, sys, pickle, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(0)
np.random.seed(0)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "l3_gate_s0_output.txt")

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

# ── Adaptive Gate Hop Layer ────────────────────────────────────────────────────

class AdaptiveGateHopLayer(nn.Module):
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

        self.W = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(K)
        ])

        gate_in = K * 4
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_in * 2),
            nn.GELU(),
            nn.Linear(gate_in * 2, K)
        )

        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.ln = nn.LayerNorm(out_dim)

    def _hop_stats(self, Zk, L):
        node_norms = Zk.norm(dim=1)
        mean_norm  = node_norms.mean()
        std_norm   = node_norms.std().clamp(min=1e-6)
        LZk = L @ Zk
        lap_quad = (LZk * Zk).sum() / (Zk.shape[0] * Zk.shape[1])
        rank_proxy = Zk.norm() / (Zk.shape[0] ** 0.5 * Zk.shape[1] ** 0.5)
        return mean_norm, std_norm, lap_quad, rank_proxy

    def forward(self, H, hop_feats, L):
        K = self.num_hops + 1

        Z_list = [self.W[k](hop_feats[k]) for k in range(K)]

        stats = []
        raw_stats = []
        for k in range(K):
            s = self._hop_stats(Z_list[k], L)
            raw_stats.append(s)
            stats.extend(list(s))

        gate_input_raw = torch.stack(stats)
        g_mean = gate_input_raw.mean()
        g_std  = gate_input_raw.std().clamp(min=1e-6)
        gate_input = (gate_input_raw - g_mean) / g_std

        raw_w = self.gate(gate_input.unsqueeze(0)).squeeze(0)
        w     = F.softmax(raw_w, dim=0)

        H_agg = sum(w[k] * Z_list[k] for k in range(K))

        g_agg = coding_rate_gradient(H_agg, self.eps)
        H_half = H_agg + self.eta * g_agg - self.eta * self.lambda_lap * (L @ H_agg)

        thr    = self.threshold.abs().unsqueeze(0)
        H_soft = H_half.sign() * F.relu(H_half.abs() - thr)
        H_out  = self.ln(H_soft)

        lap_quads = torch.stack([s[2] for s in raw_stats])
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
        seed          = 0,
    )

    log("=" * 60)
    log("Experiment: L3 Adaptive Gate GNN (FINAL) — seed=0")
    log(f"Config: {cfg}")
    log("=" * 60)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim = features.shape
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
            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | val={val_acc:.4f} | test={test_acc:.4f}")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log("=" * 60)

    return best_val_acc, best_test_acc

if __name__ == '__main__':
    best_val, best_test = run_experiment()
    save_output()
    log(f"Output saved to {OUTPUT_PATH}")
    log("Done.")

