"""
L3 Hyper-Search: Systematic hyperparameter search on G1 (CRGainGNN) architecture.

Search grid:
  hidden_dim:   [64, 128]
  lr:           [0.005, 0.01]
  dropout:      [0.5, 0.6]
  lambda_mcr:   [0.001, 0.005]

Fixed:
  epochs=400, patience=50, seed=42
  subspace_dim=16, num_hops=2, eta=0.5, eps=0.5
  lambda_lap=0.3, lambda_sparse=0.05, lambda_orth=0.005
  wd=1e-3, tau_init=1.0

Total: 2x2x2x2 = 16 configurations.
"""

import os, sys, pickle, time, itertools
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "l3_hyper_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "l3_hyper_result.txt")

output_lines = []

def log(msg):
    print(msg, flush=True)
    output_lines.append(str(msg))

def save_output():
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

# ── Data Loading ───────────────────────────────────────────────────────────────

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

# ── Coding Rate ────────────────────────────────────────────────────────────────

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

# ── CRGainHopLayer ─────────────────────────────────────────────────────────────

class CRGainHopLayer(nn.Module):
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.subspace_dim  = subspace_dim
        self.num_hops      = num_hops
        self.eta           = eta
        self.eps           = eps
        self.lambda_lap    = lambda_lap
        self.lambda_sparse = lambda_sparse

        # Learnable temperature (only learned parameter for hop weighting)
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

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
    def tau(self):
        return self.log_tau.exp()

    def forward(self, H, hop_feats, L):
        tau = self.tau
        K = self.num_hops

        # Step 1: project each hop
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]

        # Step 2: compute coding rate for each hop, then marginal gain
        R_list = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        delta_R_list = []
        for k in range(K + 1):
            if k == 0:
                delta_R_list.append(R_list[k])
            else:
                delta_R_list.append(R_list[k] - R_list[k - 1])
        delta_R_tensor = torch.stack(delta_R_list)

        # Step 3: hop weights via softmax over delta_R/tau
        w = F.softmax(delta_R_tensor / tau, dim=0)

        # Step 4: gradient step
        grad_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)
            g_H  = g_Zk @ self.W[k].weight
            grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # Step 5: dimension alignment
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 6: proximal operator
        thr   = self.threshold.abs().unsqueeze(0)
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        return H_out, w.detach(), delta_R_tensor.detach()


# ── Full Model ─────────────────────────────────────────────────────────────────

class CRGainGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0, dropout=0.6):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = CRGainHopLayer(
            in_dim, hidden_dim, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, tau_init
        )
        self.layer2 = CRGainHopLayer(
            hidden_dim, num_classes, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, tau_init
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
        H1, w1, R1 = self.layer1(H, hops1, L)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        H2, w2, R2 = self.layer2(H1, hops2, L)

        return H2, (w1, w2), (R1, R2)


# ── Losses ─────────────────────────────────────────────────────────────────────

def mcr2_loss(Z, y, num_classes, eps=0.5):
    R_total = coding_rate(Z, eps)
    R_class_sum = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        Zc = Z[mask]
        R_class_sum = R_class_sum + coding_rate(Zc, eps)
    return -(R_total - R_class_sum / num_classes)

def orth_loss(Z, y, num_classes):
    means = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 1:
            means.append(torch.zeros(Z.shape[1], device=Z.device))
        else:
            means.append(Z[mask].mean(0))
    M = torch.stack(means)
    M = F.normalize(M, dim=1)
    gram = M @ M.t()
    eye  = torch.eye(num_classes, device=Z.device)
    return (gram - eye).pow(2).sum() / (num_classes * num_classes)


# ── Training ───────────────────────────────────────────────────────────────────

# Preload data once
_data_cache = None

def get_data():
    global _data_cache
    if _data_cache is None:
        _data_cache = load_cora(DATA_DIR)
    return _data_cache

def run_config(cfg_idx, cfg):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx = get_data()
    n, in_dim = features.shape
    num_classes = int(labels.max()) + 1

    device = torch.device('cpu')
    X      = torch.tensor(features, dtype=torch.float32, device=device)
    Y      = torch.tensor(labels,   dtype=torch.long,    device=device)
    A_norm = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    L_mat  = torch.tensor(lap,      dtype=torch.float32, device=device)

    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask   = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    model = CRGainGNN(
        in_dim       = in_dim,
        hidden_dim   = cfg['hidden_dim'],
        num_classes  = num_classes,
        subspace_dim = cfg['subspace_dim'],
        num_hops     = cfg['num_hops'],
        eta          = cfg['eta'],
        eps          = cfg['eps'],
        lambda_lap   = cfg['lambda_lap'],
        lambda_sparse= cfg['lambda_sparse'],
        tau_init     = cfg['tau_init'],
        dropout      = cfg['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0
    epochs        = cfg['epochs']
    patience      = cfg['patience']

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (R1, R2) = model(X, A_norm, L_mat)

        ce   = F.cross_entropy(logits[train_mask], Y[train_mask])
        mcr  = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        loss = ce + cfg['lambda_mcr'] * mcr + cfg['lambda_orth'] * orth
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_e, _, _ = model(X, A_norm, L_mat)
            pred      = logits_e.argmax(dim=1)
            val_acc   = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            test_acc  = (pred[test_mask] == Y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    elapsed = time.time() - t0
    tag = (f"[{cfg_idx:02d}/16] hidden={cfg['hidden_dim']:3d} lr={cfg['lr']:.3f} "
           f"drop={cfg['dropout']:.1f} lmcr={cfg['lambda_mcr']:.3f}")
    log(f"{tag} => val={best_val_acc:.4f} test={best_test_acc:.4f} "
        f"({elapsed:.0f}s, stopped ep {epoch})")

    return best_val_acc, best_test_acc


# ── Search Grid ────────────────────────────────────────────────────────────────

BASE = dict(
    subspace_dim  = 16,
    num_hops      = 2,
    eta           = 0.5,
    eps           = 0.5,
    lambda_lap    = 0.3,
    lambda_sparse = 0.05,
    lambda_orth   = 0.005,
    wd            = 1e-3,
    tau_init      = 1.0,
    epochs        = 400,
    patience      = 50,
    seed          = 42,
)

HIDDEN_DIMS  = [64, 128]
LRS          = [0.005, 0.01]
DROPOUTS     = [0.5, 0.6]
LAMBDA_MCRS  = [0.001, 0.005]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L3 Hyper-Search: G1 (CRGainGNN) architecture")
    log("=" * 60)
    log(f"Grid: hidden={HIDDEN_DIMS}, lr={LRS}, dropout={DROPOUTS}, lambda_mcr={LAMBDA_MCRS}")
    log(f"Fixed: subspace_dim=16, num_hops=2, eta=0.5, eps=0.5")
    log(f"       lambda_lap=0.3, lambda_sparse=0.05, lambda_orth=0.005")
    log(f"       wd=1e-3, tau_init=1.0, epochs=400, patience=50, seed=42")
    log(f"Total configs: {len(HIDDEN_DIMS)*len(LRS)*len(DROPOUTS)*len(LAMBDA_MCRS)}")
    log("=" * 60)

    results = []
    cfg_idx = 0

    for hidden, lr, dropout, lmcr in itertools.product(
            HIDDEN_DIMS, LRS, DROPOUTS, LAMBDA_MCRS):
        cfg_idx += 1
        cfg = dict(BASE)
        cfg['hidden_dim']   = hidden
        cfg['lr']           = lr
        cfg['dropout']      = dropout
        cfg['lambda_mcr']   = lmcr

        val_acc, test_acc = run_config(cfg_idx, cfg)
        results.append({
            'hidden_dim':   hidden,
            'lr':           lr,
            'dropout':      dropout,
            'lambda_mcr':   lmcr,
            'val_acc':      val_acc,
            'test_acc':     test_acc,
        })
        save_output()

    # Sort by val_acc descending
    results_sorted = sorted(results, key=lambda r: r['val_acc'], reverse=True)

    log("")
    log("=" * 62)
    log("L3 Hyper-Search Results (G1 architecture)")
    log("=" * 62)
    log(f"{'Rank':>4} | {'hidden':>6} | {'lr':>6} | {'dropout':>7} | {'lmcr':>6} | {'Val':>6} | {'Test':>6}")
    log("-" * 62)
    for rank, r in enumerate(results_sorted, 1):
        log(f"{rank:>4} | {r['hidden_dim']:>6} | {r['lr']:>6.3f} | "
            f"{r['dropout']:>7.1f} | {r['lambda_mcr']:>6.3f} | "
            f"{r['val_acc']:>6.4f} | {r['test_acc']:>6.4f}")
    log("=" * 62)

    best = results_sorted[0]
    log(f"Best: hidden={best['hidden_dim']}, lr={best['lr']}, "
        f"dropout={best['dropout']}, lambda_mcr={best['lambda_mcr']}")
    log(f"  Val={best['val_acc']:.4f}, Test={best['test_acc']:.4f}")
    log(f"G1 baseline: val=0.7033, test=0.6800")

    save_output()

    # Write result file
    result_lines = [
        "L3 Hyper-Search Results (G1 architecture)",
        "=" * 58,
        f"{'Rank':>4} | {'hidden':>6} | {'lr':>6} | {'dropout':>7} | {'lmcr':>6} | {'Val':>6} | {'Test':>6}",
        "-" * 58,
    ]
    for rank, r in enumerate(results_sorted, 1):
        result_lines.append(
            f"{rank:>4} | {r['hidden_dim']:>6} | {r['lr']:>6.3f} | "
            f"{r['dropout']:>7.1f} | {r['lambda_mcr']:>6.3f} | "
            f"{r['val_acc']:>6.4f} | {r['test_acc']:>6.4f}"
        )
    result_lines += [
        "=" * 58,
        f"Best: hidden={best['hidden_dim']}, lr={best['lr']}, "
        f"dropout={best['dropout']}, lambda_mcr={best['lambda_mcr']}",
        f"  Val={best['val_acc']:.4f}, Test={best['test_acc']:.4f}",
        "G1 baseline: val=0.7033, test=0.6800",
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))

    print(f"\nResults saved to {RESULT_PATH}", flush=True)
    print("Done.", flush=True)

