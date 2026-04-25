"""
L1 Innovation: ISTA-derived Proximal Operator (White-box Sparse Coding Step)

Core idea: Replace empirical soft-threshold proximal step with a theoretically
grounded ISTA step derived from the sparse coding objective:
    min_Z  lambda||Z||_1 + ||Z_half - D Z||^2_F

ISTA one-step update:
    Z_out = ReLU(Z_half + eta * D^T(Z_half - D Z_half) - eta*lambda)

where D is a learnable dictionary matrix (out_dim x out_dim).
Dictionary orthogonality is encouraged via: orth_D_loss = mean((D^T D - I)^2)

Config: g1_base_small (hidden=32, subspace=8, epochs=150, patience=30)
Baseline G1: val=0.7033, test=0.6800
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

OUTPUT_PATH = os.path.join(OUT_DIR, "l1_ista_output.txt")

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

# ── ISTA Proximal Layer ────────────────────────────────────────────────────────

class ISTAProximalLayer(nn.Module):
    """
    White-box proximal step derived from sparse coding objective:
        min_Z  lambda||Z||_1 + ||Z_half - D Z||^2_F

    ISTA one-step update:
        residual = Z_half - Z_half D^T D   (gradient of ||Z_half - D Z||^2 w.r.t. Z)
        Z_ista   = Z_half + eta * residual - eta * lambda
        Z_out    = ReLU(Z_ista)            (non-negative ISTA thresholding)

    D: learnable dictionary [dim x dim], encouraged to be orthonormal via orth loss.
    """
    def __init__(self, dim, lambda_sparse=0.05, eta_ista=0.1):
        super().__init__()
        self.dim = dim
        # Learnable dictionary D: dim x dim, initialized near identity
        self.D = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        # Learnable sparsity threshold (kept positive via abs())
        self.lambda_sparse = nn.Parameter(torch.tensor(float(lambda_sparse)))
        self.eta_ista = eta_ista
        self.ln = nn.LayerNorm(dim)

    def forward(self, Z_half):
        """
        Z_half: [n, dim]
        Returns: Z_out [n, dim], orth_D_loss scalar
        """
        D = self.D  # [dim, dim]

        # Gradient of ||Z_half - Z_half D^T||^2_F w.r.t. Z_half:
        # = 2 * (Z_half - Z_half D^T) * (-D^T)^T ... simplified:
        # Objective: ||Z_half - D Z_half^T||^2 in row-vector form:
        #   each row z: ||z - z D^T||^2 = ||z(I - D^T)||^2
        # Gradient w.r.t. z: 2 z (I - D^T)^T (I - D^T) ... but ISTA uses proximal gradient
        # Correct ISTA gradient step for min_Z ||Z_half - D Z||^2_F:
        #   treating Z as the variable, gradient = -2 D^T (Z_half - D Z)
        #   at Z = Z_half: gradient = -2 D^T (Z_half - D Z_half)
        #   = -2 D^T Z_half + 2 D^T D Z_half
        # ISTA step: Z_half - eta/2 * gradient = Z_half + eta*(D^T Z_half - D^T D Z_half)
        # In row-vector form (Z_half: [n, dim]):
        #   DZ   = Z_half @ D.t()   -> [n, dim]: each row is z D^T
        #   DDZ  = DZ @ D           -> [n, dim]: each row is z D^T D
        #   grad_term = DZ - DDZ    -> [n, dim]: z D^T (I - D)  ... gradient direction

        DZ   = Z_half @ D.t()          # [n, dim]
        DDZ  = DZ @ D                   # [n, dim]
        grad_term = DZ - DDZ            # [n, dim]: D^T(Z_half - D Z_half) in row form

        lam = self.lambda_sparse.abs()
        Z_ista = Z_half + self.eta_ista * grad_term - self.eta_ista * lam
        Z_out  = F.relu(Z_ista)         # ISTA soft-threshold (non-negative form)

        # Dictionary orthogonality regularization: D^T D ~ I
        orth_D_loss = ((D.t() @ D - torch.eye(self.dim, device=D.device)) ** 2).mean()

        return self.ln(Z_out), orth_D_loss

    def sparsity(self, Z_half):
        """Compute fraction of zeros in Z_out (for monitoring)."""
        with torch.no_grad():
            D = self.D
            DZ   = Z_half @ D.t()
            DDZ  = DZ @ D
            grad_term = DZ - DDZ
            lam = self.lambda_sparse.abs()
            Z_ista = Z_half + self.eta_ista * grad_term - self.eta_ista * lam
            Z_out  = F.relu(Z_ista)
            total  = Z_out.numel()
            zeros  = (Z_out == 0).sum().item()
            return zeros / total if total > 0 else 0.0


# ── CRGainHopLayer with ISTA Proximal ─────────────────────────────────────────

class CRGainHopLayerISTA(nn.Module):
    """
    CRGainHopLayer with ISTA-derived proximal step replacing soft-threshold.

    Steps 1-5 identical to G1 baseline.
    Step 6 (proximal): replaced with ISTAProximalLayer.
    """
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0, eta_ista=0.1):
        super().__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.subspace_dim  = subspace_dim
        self.num_hops      = num_hops
        self.eta           = eta
        self.eps           = eps
        self.lambda_lap    = lambda_lap

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

        # Per-hop projection matrices W_k: in_dim -> subspace_dim
        self.W = nn.ModuleList([
            nn.Linear(in_dim, subspace_dim, bias=False)
            for _ in range(num_hops + 1)
        ])

        # Output projection: in_dim -> out_dim
        self.out_proj = nn.Linear(in_dim, out_dim, bias=False)

        # ISTA proximal layer (replaces soft-threshold + LayerNorm)
        self.ista_prox = ISTAProximalLayer(
            dim=out_dim,
            lambda_sparse=lambda_sparse,
            eta_ista=eta_ista
        )

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, H, hop_feats, L):
        """
        H:         [n, in_dim]
        hop_feats: list of K+1 tensors [n, in_dim]
        L:         [n, n] normalized Laplacian
        Returns:   H_out [n, out_dim], hop_weights [K+1], delta_R [K+1], orth_D_loss scalar
        """
        tau = self.tau
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

        # Step 3: hop weights via softmax(delta_R / tau)
        w = F.softmax(delta_R_tensor / tau, dim=0)  # [K+1]

        # Step 4: gradient step
        grad_contrib = torch.zeros_like(H)
        for k in range(K + 1):
            g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, subspace_dim]
            g_H  = g_Zk @ self.W[k].weight                    # [n, in_dim]
            grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        # Step 5: dimension alignment
        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        # Step 6: ISTA proximal step (white-box sparse coding)
        H_out, orth_D_loss = self.ista_prox(H_out_pre)

        return H_out, w.detach(), delta_R_tensor.detach(), orth_D_loss


# ── Full Model ─────────────────────────────────────────────────────────────────

class CRGainGNN_ISTA(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5,
                 lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0, dropout=0.6, eta_ista=0.1):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = CRGainHopLayerISTA(
            in_dim, hidden_dim, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, tau_init, eta_ista
        )
        self.layer2 = CRGainHopLayerISTA(
            hidden_dim, num_classes, subspace_dim, num_hops,
            eta, eps, lambda_lap, lambda_sparse, tau_init, eta_ista
        )

    def _precompute_hops(self, H, adj_norm, num_hops):
        hops = [H]
        cur = H
        for _ in range(num_hops):
            cur = adj_norm @ cur
            hops.append(cur)
        return hops

    def forward(self, H, adj_norm, L):
        # Layer 1
        hops1 = self._precompute_hops(H, adj_norm, self.num_hops)
        H1, w1, R1, orth1 = self.layer1(H, hops1, L)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        # Layer 2
        hops2 = self._precompute_hops(H1, adj_norm, self.num_hops)
        H2, w2, R2, orth2 = self.layer2(H1, hops2, L)

        return H2, (w1, w2), (R1, R2), (orth1, orth2)


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
        hidden_dim    = 32,
        subspace_dim  = 8,
        num_hops      = 2,
        eta           = 0.5,
        eps           = 0.5,
        lambda_lap    = 0.3,
        lambda_sparse = 0.05,
        lambda_mcr    = 0.005,
        lambda_orth   = 0.005,
        lambda_orth_D = 0.01,    # dictionary orthogonality loss weight
        dropout       = 0.6,
        lr            = 0.005,
        wd            = 1e-3,
        epochs        = 150,
        patience      = 30,
        seed          = 42,
        tau_init      = 1.0,
        eta_ista      = 0.1,
    )

    log("=" * 60)
    log("Experiment: L1 ISTA Proximal (White-box Sparse Coding)")
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

    model = CRGainGNN_ISTA(
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
        eta_ista     = cfg['eta_ista'],
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

        logits, (w1, w2), (R1, R2), (orth1, orth2) = model(X, A_norm, L_mat)

        # Cross-entropy (train nodes only — no label leakage)
        ce = F.cross_entropy(logits[train_mask], Y[train_mask])

        # MCR2 loss (train nodes only)
        mcr = mcr2_loss(logits[train_mask], Y[train_mask], num_classes, eps=cfg['eps'])

        # Class subspace orthogonality loss (train nodes only)
        orth = orth_loss(logits[train_mask], Y[train_mask], num_classes)

        # Dictionary orthogonality loss (D^T D ~ I, no label dependency)
        orth_D = (orth1 + orth2) * 0.5

        loss = (ce
                + cfg['lambda_mcr']    * mcr
                + cfg['lambda_orth']   * orth
                + cfg['lambda_orth_D'] * orth_D)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_e, (w1e, w2e), (R1e, R2e), (orth1e, orth2e) = model(X, A_norm, L_mat)
            pred     = logits_e.argmax(dim=1)
            val_acc  = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == Y[test_mask]).float().mean().item()

            # Compute ISTA sparsity on full graph (monitoring only)
            # Re-run layer1 forward to get H_out_pre for sparsity check
            hops1_e = model._precompute_hops(X, A_norm, model.num_hops)
            # Get H_out_pre from layer1 (before ISTA prox)
            # We compute sparsity via the ista_prox.sparsity helper
            # Need H_out_pre: run partial forward
            tau1 = model.layer1.tau
            K = model.layer1.num_hops
            Z_list1 = [model.layer1.W[k](hops1_e[k]) for k in range(K + 1)]
            R_list1 = [coding_rate(Z_list1[k], model.layer1.eps) for k in range(K + 1)]
            dR1 = []
            for k in range(K + 1):
                dR1.append(R_list1[k] if k == 0 else R_list1[k] - R_list1[k-1])
            w1_tmp = F.softmax(torch.stack(dR1) / tau1, dim=0)
            gc1 = torch.zeros_like(X)
            for k in range(K + 1):
                g = coding_rate_gradient(Z_list1[k], model.layer1.eps)
                gc1 = gc1 + w1_tmp[k] * (g @ model.layer1.W[k].weight)
            H_half1 = X + model.layer1.eta * gc1 - model.layer1.eta * model.layer1.lambda_lap * (L_mat @ X)
            H_pre1 = model.layer1.out_proj(H_half1) if model.layer1.in_dim != model.layer1.out_dim else H_half1
            sparsity1 = model.layer1.ista_prox.sparsity(H_pre1)

            orth_D_val = ((orth1e + orth2e) * 0.5).item()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1

        # Print every 10 epochs
        if epoch % 10 == 0:
            lam1 = model.layer1.ista_prox.lambda_sparse.abs().item()
            lam2 = model.layer2.ista_prox.lambda_sparse.abs().item()
            log(f"Epoch {epoch:4d} | loss={loss.item():.4f} | val={val_acc:.4f} | test={test_acc:.4f}")
            log(f"  ista_sparsity={sparsity1:.4f} | D_orth_loss={orth_D_val:.6f} | lambda=[{lam1:.4f},{lam2:.4f}]")

        if patience_cnt >= cfg['patience']:
            log(f"Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"FINAL: best_val={best_val_acc:.4f} | best_test={best_test_acc:.4f} | time={elapsed:.1f}s")
    log("=" * 60)
    log(f"G1 baseline: val=0.7033, test=0.6800")
    delta_val  = best_val_acc  - 0.7033
    delta_test = best_test_acc - 0.6800
    log(f"Delta vs G1: val={delta_val:+.4f}, test={delta_test:+.4f}")

    return best_val_acc, best_test_acc


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log("L1 Innovation: ISTA Proximal Operator")
    log(f"OUT_DIR: {OUT_DIR}")
    log("")

    best_val, best_test = run_experiment()

    save_output()
    log(f"Output saved to {OUTPUT_PATH}")
    log("Done.")

