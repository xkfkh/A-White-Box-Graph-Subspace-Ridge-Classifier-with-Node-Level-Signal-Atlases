"""
L1 Innovation: Cosine Tau Annealing + Hop Diversity Regularization
Fix: Layer1 hop_weights degenerating to uniform [0.333, 0.333, 0.333]

Three mechanisms:
  1. Cosine tau annealing: tau(epoch) = tau_min + 0.5*(tau_max-tau_min)*(1+cos(pi*t/T))
     - Early: large tau (exploration), Late: small tau (exploitation)
  2. delta_R normalization: normalize before softmax to amplify relative differences
  3. Hop diversity regularization: penalize uniform hop weights via negative variance

Config: hidden=32, subspace=8, epochs=150, patience=30 (same as g1_small_clean)
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

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "l1_tau_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "l1_tau_result.txt")

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
    """R(Z) = 0.5 * log det(I + d/(n*eps^2) * Z^T Z)"""
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    sign, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

def coding_rate_gradient(Z, eps=0.5):
    """dR/dZ = coeff * Z * (I + coeff * Z^T Z)^{-1}"""
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    M_inv = torch.linalg.inv(M)
    return coeff * (Z @ M_inv)

# ── Cosine Tau Annealing ───────────────────────────────────────────────────────

def cosine_tau(epoch, T, tau_min=0.1, tau_max=1.0):
    """
    tau(epoch) = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * epoch / T))
    epoch: 0-indexed current epoch
    T: total epochs
    """
    return tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(math.pi * epoch / T))

# ── Hop Diversity Loss ─────────────────────────────────────────────────────────

def hop_diversity_loss(w):
    """
    Penalize uniform hop weights by maximizing variance.
    loss = -var(w)  (minimize this = maximize variance = encourage non-uniform)
    w: [K+1] hop weights (after softmax, sum=1)
    """
    return -w.var()

# ── CRGainHopLayer with Tau Annealing + delta_R Normalization ─────────────────

class CRGainHopLayerAnnealed(nn.Module):
    """
    CRGainHopLayer with:
    - External tau (cosine annealing, not learned)
    - delta_R normalization before softmax
    - Returns hop_weights for diversity loss computation
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.subspace_dim  = subspace_dim
        self.num_hops      = num_hops
        self.eta           = eta
        self.eps           = eps
        self.lambda_lap    = lambda_lap
        self.lambda_sparse = lambda_sparse

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

    def forward(self, H, hop_feats, L, tau, return_H_half=False):
        """
        H:         [n, in_dim]
        hop_feats: list of K+1 tensors [n, in_dim]
        L:         [n, n] normalized Laplacian
        tau:       scalar float, externally provided (cosine annealed)
        return_H_half: if True, also return H_half for diagnostics
        Returns:   H_out [n, out_dim], hop_weights [K+1], delta_R_norm [K+1], delta_R_raw [K+1]
        """
        K = self.num_hops

        # Step 1: project each hop
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]

        # Step 2: coding rate per hop, then marginal gain
        R_list = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        delta_R_list = []
        for k in range(K + 1):
            if k == 0:
                delta_R_list.append(R_list[k])
            else:
                delta_R_list.append(R_list[k] - R_list[k - 1])
        delta_R_tensor = torch.stack(delta_R_list)  # [K+1]

        # Step 3: delta_R normalization (amplify relative differences)
        delta_R_mean = delta_R_tensor.mean()
        delta_R_std  = delta_R_tensor.std() + 1e-6
        delta_R_norm = (delta_R_tensor - delta_R_mean) / delta_R_std  # [K+1]

        # Step 4: hop weights via softmax(delta_R_norm / tau)
        w = F.softmax(delta_R_norm / tau, dim=0)  # [K+1]

        # Step 5: gradient step
        grad_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, subspace_dim]
            g_H  = g_Zk @ self.W[k].weight                    # [n, in_dim]
            grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # Step 6: dimension alignment
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 7: proximal operator — soft threshold + LayerNorm
        thr    = self.threshold.abs().unsqueeze(0)
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        if return_H_half:
            return H_out, w, delta_R_norm.detach(), delta_R_tensor.detach(), H_half.detach()
        return H_out, w, delta_R_norm.detach(), delta_R_tensor.detach()


# ── Full Model ─────────────────────────────────────────────────────────────────

class CRGainGNNAnnealed(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05, dropout=0.6):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = CRGainHopLayerAnnealed(
            in_dim, hidden_dim, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse
        )
        self.layer2 = CRGainHopLayerAnnealed(
            hidden_dim, num_classes, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        hops = [H]
        cur = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L, tau, return_H_half=False):
        # Layer 1
        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        if return_H_half:
            H1, w1, dR1_norm, dR1_raw, Hh1 = self.layer1(H, hops1, L, tau, return_H_half=True)
        else:
            H1, w1, dR1_norm, dR1_raw = self.layer1(H, hops1, L, tau)
            Hh1 = None
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        # Layer 2
        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        if return_H_half:
            H2, w2, dR2_norm, dR2_raw, Hh2 = self.layer2(H1, hops2, L, tau, return_H_half=True)
        else:
            H2, w2, dR2_norm, dR2_raw = self.layer2(H1, hops2, L, tau)
            Hh2 = None

        if return_H_half:
            return H2, (w1, w2), (dR1_norm, dR2_norm), (dR1_raw, dR2_raw), (Hh1, Hh2)
        return H2, (w1, w2), (dR1_norm, dR2_norm), (dR1_raw, dR2_raw)


# ── MCR2 Loss (train nodes only) ───────────────────────────────────────────────

def mcr2_loss(Z, y, num_classes, eps=0.5):
    """MCR2: maximize Delta_R = R(Z) - mean_k R(Z_k), train nodes only"""
    R_total = coding_rate(Z, eps)
    R_class_sum = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        Zc = Z[mask]
        R_class_sum = R_class_sum + coding_rate(Zc, eps)
    delta_R = R_total - R_class_sum / num_classes
    return -delta_R

def orth_loss(Z, y, num_classes):
    """Orthogonality loss, train nodes only"""
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

def run_experiment():
    cfg = dict(
        hidden_dim    = 32,
        subspace_dim  = 8,
        num_hops      = 2,
        eta           = 0.5,
        eps           = 0.5,
        lambda_lap    = 0.3,
        lambda_sparse = 0.05,
        lambda_mcr    = 0.005,
        lambda_orth   = 0.005,
        lambda_div    = 0.01,    # hop diversity regularization weight
        dropout       = 0.6,
        lr            = 0.005,
        wd            = 1e-3,
        epochs        = 150,
        patience      = 30,
        seed          = 42,
        tau_min       = 0.1,
        tau_max       = 1.0,
    )

    log("=" * 60)
    log("Experiment: L1 Tau Annealing + Hop Diversity Regularization")
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

    model = CRGainGNNAnnealed(
        in_dim       = in_dim,
        hidden_dim   = cfg['hidden_dim'],
        num_classes  = num_classes,
        subspace_dim = cfg['subspace_dim'],
        num_hops     = cfg['num_hops'],
        eta          = cfg['eta'],
        eps          = cfg['eps'],
        lambda_lap   = cfg['lambda_lap'],
        lambda_sparse= cfg['lambda_sparse'],
        dropout      = cfg['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0
    T = cfg['epochs']

    t0 = time.time()

    for epoch in range(1, T + 1):
        # Cosine tau annealing (0-indexed epoch for formula)
        tau = cosine_tau(epoch - 1, T, cfg['tau_min'], cfg['tau_max'])

        model.train()
        optimizer.zero_grad()

        logits, (w1, w2), (dR1_norm, dR2_norm), (dR1_raw, dR2_raw) = model(
            X, A_norm, L_mat, tau
        )

        # Cross-entropy (train nodes only — no label leakage)
        ce = F.cross_entropy(logits[train_mask], Y[train_mask])

        # MCR2 loss (train nodes only)
        mcr = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])

        # Orthogonality loss (train nodes only)
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        # Hop diversity loss: penalize uniform weights in both layers
        div1 = hop_diversity_loss(w1)
        div2 = hop_diversity_loss(w2)
        div  = div1 + div2

        loss = (ce
                + cfg['lambda_mcr']  * mcr
                + cfg['lambda_orth'] * orth
                + cfg['lambda_div']  * div)

        loss.backward()
        optimizer.step()

        # Validation
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

        # Print every 10 epochs
        if epoch % 10 == 0:
            w1_str   = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            w2_str   = '[' + ', '.join(f'{v:.3f}' for v in w2e.tolist()) + ']'
            dR1n_str = '[' + ', '.join(f'{v:.3f}' for v in dR1ne.tolist()) + ']'
            dR2n_str = '[' + ', '.join(f'{v:.3f}' for v in dR2ne.tolist()) + ']'
            dR1r_str = '[' + ', '.join(f'{v:.3f}' for v in dR1re.tolist()) + ']'
            w1_var   = w1e.var().item()
            w2_var   = w2e.var().item()

            log(f"Epoch {epoch:4d} | tau={tau:.4f} | loss={loss.item():.4f} | "
                f"val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  Layer1: hop_w={w1_str} | var={w1_var:.5f}")
            log(f"          dR_norm={dR1n_str}")
            log(f"          dR_raw={dR1r_str}")
            log(f"  Layer2: hop_w={w2_str} | var={w2_var:.5f}")
            log(f"          dR_norm={dR2n_str}")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log("=" * 60)

    return best_val_acc, best_test_acc


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L1 Innovation: Cosine Tau Annealing + Hop Diversity Regularization")
    log(f"OUT_DIR: {OUT_DIR}")
    log("")

    best_val, best_test = run_experiment()

    result_lines = [
        "L1 Tau Annealing Result",
        "=" * 40,
        "Config: l1_tau_annealing",
        "  hidden_dim=32, subspace_dim=8, num_hops=2",
        "  epochs=150, patience=30",
        "  tau_min=0.1, tau_max=1.0 (cosine annealing)",
        "  lambda_div=0.01 (hop diversity regularization)",
        "  delta_R normalization: ON",
        "-" * 40,
        f"Best Val  Acc: {best_val:.4f}",
        f"Best Test Acc: {best_test:.4f}",
        "-" * 40,
        "G1 Baseline: val=0.7033, test=0.6800",
        f"Delta Val:  {best_val - 0.7033:+.4f}",
        f"Delta Test: {best_test - 0.6800:+.4f}",
        "=" * 40,
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))

    save_output()
    log(f"Output saved to {OUTPUT_PATH}")
    log(f"Result saved to {RESULT_PATH}")
    log("Done.")

