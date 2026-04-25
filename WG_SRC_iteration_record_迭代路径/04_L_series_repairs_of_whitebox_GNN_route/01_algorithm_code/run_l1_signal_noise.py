"""
L1 Innovation: PRISM-style Signal-Noise Decomposition for Hop Aggregation

Core idea: decompose multi-hop aggregation into signal flow (low-freq) and noise flow (high-freq)
using differential weights driven by both coding rate gain (delta_R) and Laplacian energy (E_k).

Signal weight: w_signal_k = softmax(delta_R_k / tau_s)   -- high delta_R = good signal
Noise weight:  w_noise_k  = softmax(E_k / tau_n)          -- high E_k = high-freq noise

Combined score: score_k = delta_R_k / tau_s - lambda_noise * E_k / tau_n
Final weight:   w_k = softmax(score_k)

Laplacian energy: E_k = tr(Z_k^T L Z_k) / (n * d)
  - Low E_k -> low-freq (smooth) -> signal
  - High E_k -> high-freq (rough) -> noise

Config: small (hidden=32, subspace=8, num_hops=2, epochs=150, patience=30)
Baseline G1: val=0.7033, test=0.6800
"""

import os, sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "l1_sn_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "l1_sn_result.txt")

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
    import scipy.sparse as sp
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

def laplacian_energy(Z, L):
    """
    E_k = tr(Z^T L Z) / (n * d)
    Z: [n, d], L: [n, n]
    Low E -> low-freq (signal), High E -> high-freq (noise)
    """
    n, d = Z.shape
    # tr(Z^T L Z) = sum_ij (L Z)_ij * Z_ij
    LZ = L @ Z          # [n, d]
    energy = (Z * LZ).sum() / (n * d)
    return energy

# ── SNHopLayer: Signal-Noise Decomposition ─────────────────────────────────────

class SNHopLayer(nn.Module):
    """
    Signal-Noise decomposed hop aggregation layer.

    For each hop k in {0, 1, ..., K}:
      Z_k = A^k H W_k                          (projected hop-k features)
      delta_R_k = R_k - R_{k-1}                (coding rate marginal gain)
      E_k = tr(Z_k^T L Z_k) / (n*d)           (Laplacian energy, freq indicator)

    Combined score (signal - noise):
      score_k = delta_R_k / tau_s - lambda_noise * E_k / tau_n
      w_k = softmax(score_k)                   (final hop weights)

    Gradient step:
      H_half = H + eta * sum_k w_k * grad_R(Z_k) @ W_k^T
                 - eta * lambda_lap * L @ H

    Proximal step:
      H_out = LayerNorm(soft_threshold(H_half, threshold))
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_s_init=1.0, tau_n_init=1.0, lambda_noise=0.5):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.subspace_dim  = subspace_dim
        self.num_hops      = num_hops
        self.eta           = eta
        self.eps           = eps
        self.lambda_lap    = lambda_lap
        self.lambda_sparse = lambda_sparse
        self.lambda_noise  = lambda_noise

        # Learnable signal temperature (tau_s)
        self.log_tau_s = nn.Parameter(torch.tensor(float(tau_s_init)).log())
        # Learnable noise temperature (tau_n)
        self.log_tau_n = nn.Parameter(torch.tensor(float(tau_n_init)).log())

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
    def tau_s(self):
        return self.log_tau_s.exp()

    @property
    def tau_n(self):
        return self.log_tau_n.exp()

    def forward(self, H, hop_feats, L, return_H_half=False):
        """
        H:         [n, in_dim]
        hop_feats: list of K+1 tensors [n, in_dim]
        L:         [n, n] normalized Laplacian
        return_H_half: if True, also return H_half for diagnostics
        Returns:   H_out, w (final weights), delta_R, E_k, w_signal, w_noise
        """
        tau_s = self.tau_s
        tau_n = self.tau_n
        K = self.num_hops

        # Step 1: project each hop
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]

        # Step 2: coding rate per hop, then marginal gain (delta_R)
        R_list = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        delta_R_list = []
        for k in range(K + 1):
            if k == 0:
                delta_R_list.append(R_list[k])
            else:
                delta_R_list.append(R_list[k] - R_list[k - 1])
        delta_R_tensor = torch.stack(delta_R_list)  # [K+1]

        # Step 3: Laplacian energy per hop (frequency indicator)
        # Use detach for E_k to avoid double-differentiating through L (L is fixed)
        E_list = [laplacian_energy(Z_list[k], L) for k in range(K + 1)]
        E_tensor = torch.stack(E_list)  # [K+1]

        # Step 4: signal and noise weights (for diagnostics)
        w_signal = F.softmax(delta_R_tensor / tau_s, dim=0)   # [K+1]
        w_noise  = F.softmax(E_tensor / tau_n, dim=0)          # [K+1]

        # Step 5: combined score -> final hop weights
        # score_k = delta_R_k / tau_s - lambda_noise * E_k / tau_n
        score = delta_R_tensor / tau_s - self.lambda_noise * E_tensor / tau_n
        w = F.softmax(score, dim=0)  # [K+1]

        # Step 6: gradient step using final weights w
        grad_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, subspace_dim]
            g_H  = g_Zk @ self.W[k].weight                    # [n, in_dim]
            grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # Step 7: dimension alignment
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 8: proximal operator — soft threshold + LayerNorm
        thr    = self.threshold.abs().unsqueeze(0)
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        if return_H_half:
            return H_out, w.detach(), delta_R_tensor.detach(), E_tensor.detach(), \
                   w_signal.detach(), w_noise.detach(), H_half.detach()
        return H_out, w.detach(), delta_R_tensor.detach(), E_tensor.detach(), \
               w_signal.detach(), w_noise.detach()


# ── Full Model ─────────────────────────────────────────────────────────────────

class SNGainGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05,
                 tau_s_init=1.0, tau_n_init=1.0,
                 lambda_noise=0.5, dropout=0.6):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = SNHopLayer(
            in_dim, hidden_dim, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse,
            tau_s_init, tau_n_init, lambda_noise
        )
        self.layer2 = SNHopLayer(
            hidden_dim, num_classes, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse,
            tau_s_init, tau_n_init, lambda_noise
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        hops = [H]
        cur = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L, return_H_half=False):
        # Layer 1
        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        if return_H_half:
            H1, w1, R1, E1, ws1, wn1, Hh1 = self.layer1(H, hops1, L, return_H_half=True)
        else:
            H1, w1, R1, E1, ws1, wn1 = self.layer1(H, hops1, L)
            Hh1 = None
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        # Layer 2
        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        if return_H_half:
            H2, w2, R2, E2, ws2, wn2, Hh2 = self.layer2(H1, hops2, L, return_H_half=True)
        else:
            H2, w2, R2, E2, ws2, wn2 = self.layer2(H1, hops2, L)
            Hh2 = None

        if return_H_half:
            return H2, (w1, w2), (R1, R2), (E1, E2), (ws1, ws2), (wn1, wn2), (Hh1, Hh2)
        return H2, (w1, w2), (R1, R2), (E1, E2), (ws1, ws2), (wn1, wn2)


# ── MCR2 Loss (train nodes only — no label leakage) ────────────────────────────

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
    """Orthogonality loss, train nodes only — no label leakage"""
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
        lambda_noise  = 0.5,
        dropout       = 0.6,
        lr            = 0.005,
        wd            = 1e-3,
        epochs        = 150,
        patience      = 30,
        seed          = 42,
        tau_s_init    = 1.0,
        tau_n_init    = 1.0,
    )

    log("=" * 60)
    log("Experiment: L1 Signal-Noise Decomposition")
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

    model = SNGainGNN(
        in_dim       = in_dim,
        hidden_dim   = cfg['hidden_dim'],
        num_classes  = num_classes,
        subspace_dim = cfg['subspace_dim'],
        num_hops     = cfg['num_hops'],
        eta          = cfg['eta'],
        eps          = cfg['eps'],
        lambda_lap   = cfg['lambda_lap'],
        lambda_sparse= cfg['lambda_sparse'],
        tau_s_init   = cfg['tau_s_init'],
        tau_n_init   = cfg['tau_n_init'],
        lambda_noise = cfg['lambda_noise'],
        dropout      = cfg['dropout'],
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

        out = model(X, A_norm, L_mat)
        logits, (w1, w2), (R1, R2), (E1, E2), (ws1, ws2), (wn1, wn2) = out

        # Cross-entropy (train nodes only — no label leakage)
        ce = F.cross_entropy(logits[train_mask], Y[train_mask])

        # MCR2 loss (train nodes only — no label leakage)
        mcr = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])

        # Orthogonality loss (train nodes only — no label leakage)
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        loss = ce + cfg['lambda_mcr'] * mcr + cfg['lambda_orth'] * orth
        loss.backward()
        optimizer.step()

        # Validation (eval mode)
        model.eval()
        with torch.no_grad():
            out_e = model(X, A_norm, L_mat, return_H_half=True)
            logits_e, (w1e, w2e), (R1e, R2e), (E1e, E2e), (ws1e, ws2e), (wn1e, wn2e), (Hh1, Hh2) = out_e
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
            hh1_norm = Hh1.norm().item()
            hh2_norm = Hh2.norm().item()

            tau_s1 = model.layer1.tau_s.item()
            tau_n1 = model.layer1.tau_n.item()
            tau_s2 = model.layer2.tau_s.item()
            tau_n2 = model.layer2.tau_n.item()

            w1_str  = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            ws1_str = '[' + ', '.join(f'{v:.3f}' for v in ws1e.tolist()) + ']'
            wn1_str = '[' + ', '.join(f'{v:.3f}' for v in wn1e.tolist()) + ']'
            E1_str  = '[' + ', '.join(f'{v:.4f}' for v in E1e.tolist()) + ']'
            R1_str  = '[' + ', '.join(f'{v:.3f}' for v in R1e.tolist()) + ']'

            w2_str  = '[' + ', '.join(f'{v:.3f}' for v in w2e.tolist()) + ']'
            ws2_str = '[' + ', '.join(f'{v:.3f}' for v in ws2e.tolist()) + ']'
            wn2_str = '[' + ', '.join(f'{v:.3f}' for v in wn2e.tolist()) + ']'
            E2_str  = '[' + ', '.join(f'{v:.4f}' for v in E2e.tolist()) + ']'
            R2_str  = '[' + ', '.join(f'{v:.3f}' for v in R2e.tolist()) + ']'

            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  [L1-SN] Layer1: H_half_norm={hh1_norm:.4f} | tau_s={tau_s1:.4f} | tau_n={tau_n1:.4f}")
            log(f"          final_w={w1_str}")
            log(f"          signal_w={ws1_str}")
            log(f"          noise_w={wn1_str}")
            log(f"          E_k={E1_str}  (low=signal, high=noise)")
            log(f"          delta_R={R1_str}")
            log(f"  [L1-SN] Layer2: H_half_norm={hh2_norm:.4f} | tau_s={tau_s2:.4f} | tau_n={tau_n2:.4f}")
            log(f"          final_w={w2_str}")
            log(f"          signal_w={ws2_str}")
            log(f"          noise_w={wn2_str}")
            log(f"          E_k={E2_str}  (low=signal, high=noise)")
            log(f"          delta_R={R2_str}")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log(f"G1 Baseline: val=0.7033 | test=0.6800")
    val_diff  = best_val_acc  - 0.7033
    test_diff = best_test_acc - 0.6800
    log(f"Delta vs G1: val={val_diff:+.4f} | test={test_diff:+.4f}")
    log("=" * 60)

    return best_val_acc, best_test_acc


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L1 Innovation: PRISM Signal-Noise Decomposition")
    log(f"OUT_DIR: {OUT_DIR}")
    log("")

    best_val, best_test = run_experiment()

    result_lines = [
        "L1 Signal-Noise Result",
        "=" * 40,
        "Config: l1_signal_noise_small",
        "  hidden_dim=32, subspace_dim=8, num_hops=2",
        "  epochs=150, patience=30, lambda_noise=0.5",
        "-" * 40,
        f"Best Val  Acc: {best_val:.4f}",
        f"Best Test Acc: {best_test:.4f}",
        "-" * 40,
        "G1 Baseline: val=0.7033, test=0.6800",
        f"Delta val:  {best_val - 0.7033:+.4f}",
        f"Delta test: {best_test - 0.6800:+.4f}",
        "=" * 40,
    ]
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))

    save_output()
    log(f"Output saved to {OUTPUT_PATH}")
    log(f"Result saved to {RESULT_PATH}")
    log("Done.")


