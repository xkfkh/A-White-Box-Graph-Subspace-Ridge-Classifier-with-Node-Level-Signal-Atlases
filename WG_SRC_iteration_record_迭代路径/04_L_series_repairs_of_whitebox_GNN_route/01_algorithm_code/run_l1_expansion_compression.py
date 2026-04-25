"""
L1 Innovation: Expansion + Compression Dual-Stream White-box GNN
================================================================
Based on run_g1_small_clean.py (hidden=32, subspace=8, epochs=150, patience=30)

Key innovation: adds class-conditional compression term to the gradient step,
forming a true closed-loop white-box (ReduNet-style expansion + compression).

Gradient step (full):
  H_half = H + eta * expand_contrib
              - eta * lambda_compress * compress_contrib
              - eta * lambda_lap * L @ H

Expansion term (original G1):
  expand_contrib = sum_k w_k * grad_R(Z_k) @ W_k^T

Compression term (new):
  compress_contrib = compute_compression(Z, soft_labels, num_classes)
  - train nodes: soft_labels from one-hot of true labels
  - other nodes: soft_labels from current logits softmax (no label leakage)

Soft labels for layer 1: uniform (1/C) for all nodes
Soft labels for layer 2: softmax of layer 1 output (train nodes use true one-hot)
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

OUTPUT_PATH = os.path.join(OUT_DIR, "l1_ec_output.txt")
RESULT_PATH = os.path.join(OUT_DIR, "l1_ec_result.txt")

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

# ── Compression Term ───────────────────────────────────────────────────────────

def compute_compression(Z, soft_labels, num_classes, eps=0.5):
    """
    Class-conditional compression gradient (ReduNet compression term).

    Z:           [n, d]  node features in subspace
    soft_labels: [n, C]  soft class membership (probabilities)
    Returns:     [n, d]  compression gradient contribution

    For each class c:
      Z_c = diag(gamma_c) @ Z  (soft-weighted features)
      grad_c = coeff_c * Z_c @ (I + coeff_c * Z_c^T Z_c)^{-1}
      compress += gamma_c * grad_c  (weighted sum back to node space)

    This pulls each node's representation toward its class subspace.
    """
    n, d = Z.shape
    compress = torch.zeros_like(Z)
    for c in range(num_classes):
        gamma_c = soft_labels[:, c]          # [n] soft weight for class c
        Z_c = gamma_c.unsqueeze(1) * Z       # [n, d] soft-weighted features
        # Use n_eff = gamma_c.sum() as effective sample count
        n_eff = gamma_c.sum().clamp(min=1.0)
        coeff = d / (n_eff * eps * eps)
        M_c = torch.eye(d, device=Z.device) + coeff * (Z_c.t() @ Z_c)
        M_c_inv = torch.linalg.inv(M_c)
        grad_c = coeff * (Z_c @ M_c_inv)    # [n, d]
        compress = compress + gamma_c.unsqueeze(1) * grad_c
    return compress

# ── CRGainHopLayer with Compression ───────────────────────────────────────────

class CRGainHopLayerEC(nn.Module):
    """
    Expansion + Compression dual-stream white-box hop layer.

    Gradient step:
      H_half = H + eta * expand_contrib
                 - eta * lambda_compress * compress_contrib
                 - eta * lambda_lap * L @ H

    expand_contrib  = sum_k w_k * grad_R(Z_k) @ W_k^T   (expansion)
    compress_contrib = compute_compression(Z_agg, soft_labels, C)  (compression)

    Z_agg is the weighted aggregate of hop projections used for compression.
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_classes, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 lambda_compress=0.1, tau_init=1.0):
        super().__init__()
        self.in_dim          = in_dim
        self.out_dim         = out_dim
        self.subspace_dim    = subspace_dim
        self.num_classes     = num_classes
        self.num_hops        = num_hops
        self.eta             = eta
        self.eps             = eps
        self.lambda_lap      = lambda_lap
        self.lambda_sparse   = lambda_sparse
        self.lambda_compress = lambda_compress

        # Learnable temperature
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

    def forward(self, H, hop_feats, L, soft_labels, return_norms=False):
        """
        H:           [n, in_dim]
        hop_feats:   list of K+1 tensors [n, in_dim]
        L:           [n, n] normalized Laplacian
        soft_labels: [n, C] soft class membership (no label leakage for unlabeled)
        return_norms: if True, also return expand_norm, compress_norm, H_half
        Returns: H_out [n, out_dim], hop_weights [K+1], delta_R [K+1]
        """
        tau = self.tau
        K = self.num_hops

        # Step 1: project each hop to subspace
        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]

        # Step 2: coding rate per hop, marginal gain
        R_list = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        delta_R_list = []
        for k in range(K + 1):
            if k == 0:
                delta_R_list.append(R_list[k])
            else:
                delta_R_list.append(R_list[k] - R_list[k - 1])
        delta_R_tensor = torch.stack(delta_R_list)  # [K+1]

        # Step 3: hop weights via softmax(delta_R / tau)
        w = F.softmax(delta_R_tensor / tau, dim=0)  # [K+1]

        # Step 4a: expansion gradient contribution
        expand_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, subspace_dim]
            g_H  = g_Zk @ self.W[k].weight                    # [n, in_dim]
            expand_contrib = expand_contrib + w[k] * g_H

        # Step 4b: compression gradient contribution
        # Use weighted aggregate Z for compression (same subspace as expansion)
        Z_agg = sum(w[k] * Z_list[k] for k in range(K + 1))  # [n, subspace_dim]
        compress_contrib_sub = compute_compression(
            Z_agg, soft_labels, self.num_classes, self.eps
        )  # [n, subspace_dim]
        # Back-project to in_dim space using weighted W^T
        compress_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            compress_contrib = compress_contrib + w[k] * (compress_contrib_sub @ self.W[k].weight)

        # Step 4c: full gradient step
        H_half = (H
                  + self.eta * expand_contrib
                  - self.eta * self.lambda_compress * compress_contrib
                  - self.eta * self.lambda_lap * (L @ H))

        # Step 5: dimension alignment
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 6: proximal operator — soft threshold + LayerNorm
        thr    = self.threshold.abs().unsqueeze(0)
        H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)
        H_out  = self.ln(H_soft)

        if return_norms:
            en = expand_contrib.norm().item()
            cn = compress_contrib.norm().item()
            return H_out, w.detach(), delta_R_tensor.detach(), en, cn, H_half.detach()
        return H_out, w.detach(), delta_R_tensor.detach()


# ── Full Model ─────────────────────────────────────────────────────────────────

class CRGainGNN_EC(nn.Module):
    """
    Two-layer GNN with expansion+compression dual-stream white-box layers.

    Soft label strategy (no label leakage):
      Layer 1: uniform soft labels (1/C) for all nodes
      Layer 2: softmax of layer 1 output; train nodes replaced with true one-hot
    """
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05, lambda_compress=0.1,
                 tau_init=1.0, dropout=0.6):
        super().__init__()
        self.num_hops     = num_hops
        self.dropout      = dropout
        self.num_classes  = num_classes

        self.layer1 = CRGainHopLayerEC(
            in_dim, hidden_dim, subspace_dim, num_classes, num_hops,
            eta, eps, lambda_lap, lambda_sparse, lambda_compress, tau_init
        )
        self.layer2 = CRGainHopLayerEC(
            hidden_dim, num_classes, subspace_dim, num_classes, num_hops,
            eta, eps, lambda_lap, lambda_sparse, lambda_compress, tau_init
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        hops = [H]
        cur = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L, train_mask, Y_train,
                return_norms=False):
        """
        H:          [n, in_dim]
        train_mask: [n] bool
        Y_train:    [n] long, true labels (only train_mask positions are valid)
        """
        n = H.shape[0]
        C = self.num_classes

        # ── Layer 1 soft labels: uniform (1/C) — no label info used ──────────
        soft1 = torch.full((n, C), 1.0 / C, device=H.device)

        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        if return_norms:
            H1, w1, R1, en1, cn1, Hh1 = self.layer1(
                H, hops1, L, soft1, return_norms=True)
        else:
            H1, w1, R1 = self.layer1(H, hops1, L, soft1)
            en1 = cn1 = Hh1 = None

        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        # ── Layer 2 soft labels: project H1 -> C dims for unlabeled, one-hot for train ─
        # Use layer2.out_proj (H1 -> num_classes) without gradient to get soft predictions
        with torch.no_grad():
            logits_soft = self.layer2.out_proj(H1.detach())  # [n, C]
            soft2 = F.softmax(logits_soft, dim=1)            # [n, C], no grad
        # Replace train nodes with true one-hot (only train nodes have labels)
        one_hot = torch.zeros(n, C, device=H.device)
        one_hot[train_mask] = F.one_hot(
            Y_train[train_mask], num_classes=C
        ).float()
        soft2 = soft2.clone()
        soft2[train_mask] = one_hot[train_mask]

        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        if return_norms:
            H2, w2, R2, en2, cn2, Hh2 = self.layer2(
                H1, hops2, L, soft2, return_norms=True)
        else:
            H2, w2, R2 = self.layer2(H1, hops2, L, soft2)
            en2 = cn2 = Hh2 = None

        if return_norms:
            return H2, (w1, w2), (R1, R2), (en1, cn1, Hh1), (en2, cn2, Hh2)
        return H2, (w1, w2), (R1, R2)


# ── MCR2 Loss ──────────────────────────────────────────────────────────────────

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
        hidden_dim       = 32,
        subspace_dim     = 8,
        num_hops         = 2,
        eta              = 0.5,
        eps              = 0.5,
        lambda_lap       = 0.3,
        lambda_sparse    = 0.05,
        lambda_compress  = 0.1,
        lambda_mcr       = 0.005,
        lambda_orth      = 0.005,
        dropout          = 0.6,
        lr               = 0.005,
        wd               = 1e-3,
        epochs           = 150,
        patience         = 30,
        seed             = 42,
        tau_init         = 1.0,
    )

    log("=" * 60)
    log("Experiment: L1 Expansion+Compression White-box GNN")
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

    model = CRGainGNN_EC(
        in_dim          = in_dim,
        hidden_dim      = cfg['hidden_dim'],
        num_classes     = num_classes,
        subspace_dim    = cfg['subspace_dim'],
        num_hops        = cfg['num_hops'],
        eta             = cfg['eta'],
        eps             = cfg['eps'],
        lambda_lap      = cfg['lambda_lap'],
        lambda_sparse   = cfg['lambda_sparse'],
        lambda_compress = cfg['lambda_compress'],
        tau_init        = cfg['tau_init'],
        dropout         = cfg['dropout'],
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

        logits, (w1, w2), (R1, R2) = model(X, A_norm, L_mat, train_mask, Y)

        ce   = F.cross_entropy(logits[train_mask], Y[train_mask])
        mcr  = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        loss = ce + cfg['lambda_mcr'] * mcr + cfg['lambda_orth'] * orth
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(X, A_norm, L_mat, train_mask, Y, return_norms=True)
            logits_e, (w1e, w2e), (R1e, R2e), (en1, cn1, Hh1), (en2, cn2, Hh2) = out
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
            w1_str = '[' + ', '.join(f'{v:.3f}' for v in w1e.tolist()) + ']'
            w2_str = '[' + ', '.join(f'{v:.3f}' for v in w2e.tolist()) + ']'
            R1_str = '[' + ', '.join(f'{v:.3f}' for v in R1e.tolist()) + ']'
            R2_str = '[' + ', '.join(f'{v:.3f}' for v in R2e.tolist()) + ']'
            tau1   = model.layer1.tau.item()
            tau2   = model.layer2.tau.item()

            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  [L1-EC] Layer1: expand_norm={en1:.4f} | compress_norm={cn1:.4f} | tau={tau1:.4f}")
            log(f"          hop_w={w1_str}")
            log(f"          delta_R={R1_str}")
            log(f"  [L1-EC] Layer2: expand_norm={en2:.4f} | compress_norm={cn2:.4f} | tau={tau2:.4f}")
            log(f"          hop_w={w2_str}")
            log(f"          delta_R={R2_str}")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log(f"G1 baseline: val=0.7033 | test=0.6800")
    delta_val  = best_val_acc  - 0.7033
    delta_test = best_test_acc - 0.6800
    log(f"Delta vs G1: val={delta_val:+.4f} | test={delta_test:+.4f}")
    log("=" * 60)

    return best_val_acc, best_test_acc


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L1 Innovation: Expansion + Compression Dual-Stream White-box GNN")
    log(f"OUT_DIR: {OUT_DIR}")
    log("")

    best_val, best_test = run_experiment()

    result_lines = [
        "L1 Expansion+Compression Result",
        "=" * 40,
        "Config: l1_ec (hidden=32, subspace=8, num_hops=2)",
        "  epochs=150, patience=30, lambda_compress=0.1",
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


