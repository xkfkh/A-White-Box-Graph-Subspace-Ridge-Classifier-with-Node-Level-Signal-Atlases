import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR    = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR     = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
HIDDEN_DIM  = 64
NUM_HOPS    = 2          # scales: A^0 X, A^1 X, A^2 X
DROPOUT     = 0.6
LR          = 0.01       # slightly higher to escape flat regions
WD          = 1e-3       # stronger regularisation vs 5e-4
EPOCHS      = 400
PATIENCE    = 100
SEED        = 0
LAMBDA_MCR  = 0.05       # stronger MCR2: guide subspace separation from the start
MCR_WARMUP  = 0          # active from epoch 1

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_cora(data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs = []
    for n in names:
        with open(f"{data_dir}/ind.cora.{n}", 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_raw = [int(l.strip()) for l in open(f"{data_dir}/ind.cora.test.index")]
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
    train_idx = list(range(140))
    val_idx   = list(range(200, 500))
    test_idx  = test_idx_sorted[:1000]
    return features, labels, adj_norm, train_idx, val_idx, test_idx

# ─────────────────────────────────────────────
# MCR2 helpers  (Z L2-normalised for scale stability)
# ─────────────────────────────────────────────
def coding_rate(Z, eps=0.5):
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    sign, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

def mcr2_loss(Z, y, num_classes, eps=0.5):
    # L2-normalise rows: coding_rate stays O(1) regardless of embedding magnitude
    Z = F.normalize(Z, p=2, dim=1)
    R_total = coding_rate(Z, eps)
    R_class_sum = 0.0
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() < 2:
            continue
        Zc = Z[mask]
        R_class_sum = R_class_sum + coding_rate(Zc, eps)
    # maximise DeltaR = R_total - mean_k R_k  =>  minimise its negative
    return -(R_total - R_class_sum / num_classes)

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class MultiScaleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_hops=2, dropout=0.6):
        super().__init__()
        K = num_hops + 1
        self.K = K
        # independent projection per hop
        self.hop_linears = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(K)
        ])
        self.hop_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(K)
        ])
        # learnable attention weights over hops
        self.alpha = nn.Parameter(torch.zeros(K))   # init 0 -> uniform after softmax
        # classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, hop_feats):
        embeddings = []
        for k in range(self.K):
            e = self.hop_linears[k](hop_feats[k])
            e = self.hop_bns[k](e)
            e = F.relu(e)
            embeddings.append(e)

        w = F.softmax(self.alpha, dim=0)                         # [K]
        H = sum(w[k] * embeddings[k] for k in range(self.K))    # [n, hidden]

        logits = self.classifier(self.dropout_layer(H))
        return logits, H, w

# ─────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────
def accuracy(logits, labels, mask):
    preds = logits[mask].argmax(dim=1)
    return (preds == labels[mask]).float().mean().item()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # -- Load --
    features, labels, adj_norm, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim = features.shape
    num_classes = int(labels.max()) + 1

    X = torch.tensor(features)
    Y = torch.tensor(labels, dtype=torch.long)
    A = torch.tensor(adj_norm)

    # Pre-compute hop features once
    hop_feats = [X]
    cur = X
    for _ in range(1, NUM_HOPS + 1):
        cur = A @ cur
        hop_feats.append(cur)

    print(f"Hop features: {[tuple(h.shape) for h in hop_feats]}")

    # Masks
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]   = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx] = True

    # -- Model --
    model = MultiScaleGCN(in_dim, HIDDEN_DIM, num_classes, NUM_HOPS, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-4
    )

    best_val = best_test = 0.0
    no_improve = 0
    best_state = None

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log(f"L3 Multi-Scale GCN | Cora | hidden={HIDDEN_DIM} hops={NUM_HOPS} "
        f"lambda_mcr={LAMBDA_MCR} warmup={MCR_WARMUP} SEED={SEED}")
    log("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # -- Train step --
        model.train()
        logits, H, w = model(hop_feats)
        ce = F.cross_entropy(logits[train_mask], Y[train_mask])

        # MCR2 warm-up: only activate after enough CE convergence
        if epoch > MCR_WARMUP:
            mcr = mcr2_loss(H[train_mask], Y[train_mask], num_classes)
            loss = ce + LAMBDA_MCR * mcr
        else:
            mcr = torch.tensor(0.0)
            loss = ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # -- Eval --
        model.eval()
        with torch.no_grad():
            logits_e, H_e, w_e = model(hop_feats)
            val_acc  = accuracy(logits_e, Y, val_mask)
            test_acc = accuracy(logits_e, Y, test_mask)

        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val  = val_acc
            best_test = test_acc
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            w_np = w_e.detach().numpy()
            hop_str = "  ".join(f"h{k}={w_np[k]:.3f}" for k in range(NUM_HOPS + 1))
            log(f"Ep{epoch:4d}  loss={loss.item():.4f}  ce={ce.item():.4f}  "
                f"mcr={mcr.item():.4f}  val={val_acc:.4f}  test={test_acc:.4f}  "
                f"[{hop_str}]")

        if no_improve >= PATIENCE:
            log(f"Early stop at epoch {epoch}  (no val improvement for {PATIENCE} epochs)")
            break

    log("=" * 70)
    log(f"Best Val  Acc: {best_val:.4f}")
    log(f"Best Test Acc: {best_test:.4f}")
    log("=" * 70)

    # Restore best model & report final hop weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, w_final = model(hop_feats)
    w_np = w_final.detach().numpy()
    log("Hop attention weights at best checkpoint:")
    for k in range(NUM_HOPS + 1):
        log(f"  hop{k} (A^{k}X) = {w_np[k]:.4f}")

    # -- Write output log --
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "l3_ms_s0_output.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nOutput log  -> {out_path}")

    print(f"\n[STABILITY] SEED={SEED}  val={best_val:.4f}  test={best_test:.4f}")


if __name__ == "__main__":
    main()

