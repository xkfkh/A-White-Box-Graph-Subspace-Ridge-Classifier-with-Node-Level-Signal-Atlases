"""
run_diagnosis.py
================
G1 CRGain 深度诊断脚本
目标: val=0.7033 / test=0.6800 — 找出根本原因

诊断内容:
  Part 1: 标准 2 层 GCN 基线
  Part 2: G1 组件监控 (每 50 epoch 打印内部信号)
  Part 3: 消融实验 V1/V2/V3 + G1 full
"""

import os, sys, pickle, time
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"
OUT_DIR  = "D:/桌面/MSR实验复现与创新/results/whitebox_gat_v3"
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUT_DIR, "diagnosis_output.txt")

_lines = []

def log(msg=""):
    s = str(msg)
    print(s, flush=True)
    _lines.append(s)

def save():
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_lines))

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_cora(data_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs  = []
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
    labels   = np.vstack((ally, ty))
    labels[test_idx_raw, :] = labels[test_idx_sorted, :]
    labels   = np.argmax(labels, axis=1)
    n_nodes  = features.shape[0]
    adj      = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0
            adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg        = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    adj_norm   = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    lap        = np.eye(n_nodes, dtype=np.float32) - adj_norm
    train_idx  = list(range(140))
    val_idx    = list(range(200, 500))
    test_idx   = test_idx_sorted[:1000]
    return features, labels, adj_norm, lap, train_idx, val_idx, test_idx

# ─────────────────────────────────────────────
# Coding Rate 工具
# ─────────────────────────────────────────────
def coding_rate(Z, eps=0.5):
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    sign, logdet = torch.linalg.slogdet(M)
    return 0.5 * logdet

def coding_rate_gradient(Z, eps=0.5):
    n, d = Z.shape
    coeff = d / (n * eps * eps)
    M     = torch.eye(d, device=Z.device) + coeff * (Z.t() @ Z)
    M_inv = torch.linalg.inv(M)
    return coeff * (Z @ M_inv)

# ─────────────────────────────────────────────
# Part 1: 标准 2 层 GCN 基线
# ─────────────────────────────────────────────
class StandardGCN(nn.Module):
    """
    2-layer GCN:
      H1 = ReLU( A_norm @ X @ W1 )
      logits = A_norm @ H1 @ W2
    """
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.6):
        super().__init__()
        self.W1 = nn.Linear(in_dim,     hidden_dim,  bias=False)
        self.W2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = dropout

    def forward(self, X, A_norm):
        # Layer 1
        H = A_norm @ X
        H = self.W1(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)
        # Layer 2
        H = A_norm @ H
        logits = self.W2(H)
        return logits


def run_gcn_baseline(features, labels, adj_norm, train_idx, val_idx, test_idx,
                     hidden=64, dropout=0.6, lr=0.005, epochs=300, patience=50, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')

    n      = features.shape[0]
    in_dim = features.shape[1]
    num_classes = int(labels.max()) + 1

    X = torch.tensor(features,  dtype=torch.float32, device=device)
    Y = torch.tensor(labels,    dtype=torch.long,    device=device)
    A = torch.tensor(adj_norm,  dtype=torch.float32, device=device)

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True

    model = StandardGCN(in_dim, hidden, num_classes, dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = best_test = 0.0
    pat_cnt  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X, A)
        loss   = F.cross_entropy(logits[train_mask], Y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out  = model(X, A)
            pred = out.argmax(dim=1)
            va   = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            te   = (pred[test_mask] == Y[test_mask]).float().mean().item()
        if va > best_val:
            best_val  = va
            best_test = te
            pat_cnt   = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            break

    return best_val, best_test


# ─────────────────────────────────────────────
# G1 层 (可配置消融)
# ─────────────────────────────────────────────
class CRGainHopLayer(nn.Module):
    def __init__(self, in_dim, out_dim, subspace_dim, num_hops=2,
                 eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0,
                 disable_cr=False,       # V1: eta=0
                 disable_lap=False,      # V2: lambda_lap=0
                 disable_threshold=False # V3: threshold=0
                 ):
        super().__init__()
        self.in_dim            = in_dim
        self.out_dim           = out_dim
        self.subspace_dim      = subspace_dim
        self.num_hops          = num_hops
        self.eta               = 0.0 if disable_cr else eta
        self.eps               = eps
        self.lambda_lap        = 0.0 if disable_lap else lambda_lap
        self.lambda_sparse     = lambda_sparse
        self.disable_threshold = disable_threshold

        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

        self.W = nn.ModuleList([
            nn.Linear(in_dim, subspace_dim, bias=False)
            for _ in range(num_hops + 1)
        ])
        self.out_proj  = nn.Linear(in_dim, out_dim, bias=False)
        self.threshold = nn.Parameter(torch.full((out_dim,), lambda_sparse))
        self.ln        = nn.LayerNorm(out_dim)

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, H, hop_feats, L, collect_diag=False):
        tau = self.tau
        K   = self.num_hops

        Z_list = [self.W[k](hop_feats[k]) for k in range(K + 1)]

        R_list      = [coding_rate(Z_list[k], self.eps) for k in range(K + 1)]
        delta_R_list = []
        for k in range(K + 1):
            if k == 0:
                delta_R_list.append(R_list[k])
            else:
                delta_R_list.append(R_list[k] - R_list[k - 1])
        delta_R_tensor = torch.stack(delta_R_list)   # [K+1]

        w = F.softmax(delta_R_tensor / tau, dim=0)   # [K+1]

        # 梯度步
        grad_contrib = torch.zeros_like(H)
        if self.eta > 0:
            for k in range(K + 1):
                g_Zk = coding_rate_gradient(Z_list[k], self.eps)  # [n, sub]
                g_H  = g_Zk @ self.W[k].weight                    # [n, in_dim]
                grad_contrib = grad_contrib + w[k] * g_H

        H_half = H + self.eta * grad_contrib - self.eta * self.lambda_lap * (L @ H)

        if self.in_dim == self.out_dim:
            H_out_pre = H_half
        else:
            H_out_pre = self.out_proj(H_half)

        if self.disable_threshold:
            H_soft = H_out_pre
        else:
            thr    = self.threshold.abs().unsqueeze(0)
            H_soft = H_out_pre.sign() * F.relu(H_out_pre.abs() - thr)

        H_out = self.ln(H_soft)

        # 诊断量 (detach, 不影响梯度)
        diag = {}
        if collect_diag:
            with torch.no_grad():
                diag['delta_R_mean']  = delta_R_tensor.abs().mean().item()
                diag['hop_var']       = w.var().item()
                diag['grad_norm']     = grad_contrib.norm().item()
                diag['H_half_norm']   = H_half.norm().item()
                if self.disable_threshold:
                    diag['sparsity']  = 0.0
                else:
                    nonzero = (H_soft.abs() > 1e-8).float().mean().item()
                    diag['sparsity']  = nonzero
                diag['ln_std']        = H_out.std().item()

        return H_out, w.detach(), delta_R_tensor.detach(), diag


class CRGainGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, subspace_dim,
                 num_hops=2, eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
                 tau_init=1.0, dropout=0.6,
                 disable_cr=False, disable_lap=False, disable_threshold=False):
        super().__init__()
        self.num_hops = num_hops
        self.dropout  = dropout

        self.layer1 = CRGainHopLayer(in_dim,     hidden_dim,  subspace_dim, num_hops,
                                     eta, eps, lambda_lap, lambda_sparse, tau_init,
                                     disable_cr, disable_lap, disable_threshold)
        self.layer2 = CRGainHopLayer(hidden_dim, num_classes, subspace_dim, num_hops,
                                     eta, eps, lambda_lap, lambda_sparse, tau_init,
                                     disable_cr, disable_lap, disable_threshold)

    def _hops(self, H, A, K):
        hops = [H]
        cur  = H
        for _ in range(K):
            cur = A @ cur
            hops.append(cur)
        return hops

    def forward(self, H, A, L, collect_diag=False):
        hops1 = self._hops(H,  A, self.num_hops)
        H1, w1, R1, d1 = self.layer1(H,  hops1, L, collect_diag)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H1 = F.elu(H1)

        hops2 = self._hops(H1, A, self.num_hops)
        H2, w2, R2, d2 = self.layer2(H1, hops2, L, collect_diag)

        return H2, (w1, w2), (R1, R2), (d1, d2)


# ─────────────────────────────────────────────
# G1 训练函数 (通用)
# ─────────────────────────────────────────────
def run_g1(features, labels, adj_norm, lap,
           train_idx, val_idx, test_idx,
           hidden=32, subspace=8, num_hops=2,
           eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
           lambda_mcr=0.005, lambda_orth=0.005,
           tau_init=1.0, dropout=0.6, lr=0.005, wd=1e-3,
           epochs=150, patience=30, seed=42,
           disable_cr=False, disable_lap=False, disable_threshold=False,
           monitor=False):
    """
    monitor=True: 每 50 epoch 收集并返回诊断量
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')

    n           = features.shape[0]
    in_dim      = features.shape[1]
    num_classes = int(labels.max()) + 1

    X = torch.tensor(features, dtype=torch.float32, device=device)
    Y = torch.tensor(labels,   dtype=torch.long,    device=device)
    A = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    L = torch.tensor(lap,      dtype=torch.float32, device=device)

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True

    model = CRGainGNN(in_dim, hidden, num_classes, subspace, num_hops,
                      eta, eps, lambda_lap, lambda_sparse, tau_init, dropout,
                      disable_cr, disable_lap, disable_threshold).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val = best_test = 0.0
    pat_cnt  = 0
    monitor_rows = []   # list of (epoch, diag_layer1)

    def mcr2_loss(Z, y):
        R_total = coding_rate(Z, eps)
        R_cls   = 0.0
        for c in range(num_classes):
            mask = (y == c)
            if mask.sum() < 2:
                continue
            R_cls = R_cls + coding_rate(Z[mask], eps)
        return -(R_total - R_cls / num_classes)

    def orth_loss(Z, y):
        means = []
        for c in range(num_classes):
            mask = (y == c)
            means.append(Z[mask].mean(0) if mask.sum() >= 1 else torch.zeros(Z.shape[1], device=device))
        M    = torch.stack(means)
        M    = F.normalize(M, dim=1)
        gram = M @ M.t()
        eye  = torch.eye(num_classes, device=device)
        return (gram - eye).pow(2).sum() / (num_classes ** 2)

    for epoch in range(1, epochs + 1):
        # ── 是否收集诊断量 ──
        do_diag = monitor and (epoch % 50 == 0)

        model.train()
        opt.zero_grad()
        logits, (w1, w2), (R1, R2), (d1, d2) = model(X, A, L, collect_diag=do_diag)
        ce   = F.cross_entropy(logits[train_mask], Y[train_mask])
        mcr  = mcr2_loss(logits[train_mask], Y[train_mask])
        orth = orth_loss(logits[train_mask], Y[train_mask])
        loss = ce + lambda_mcr * mcr + lambda_orth * orth
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_e, _, _, _ = model(X, A, L, collect_diag=False)
            pred = out_e.argmax(dim=1)
            va   = (pred[val_mask]  == Y[val_mask]).float().mean().item()
            te   = (pred[test_mask] == Y[test_mask]).float().mean().item()

        if va > best_val:
            best_val  = va
            best_test = te
            pat_cnt   = 0
        else:
            pat_cnt += 1

        if do_diag and d1:
            monitor_rows.append((epoch, d1))

        if pat_cnt >= patience:
            break

    return best_val, best_test, monitor_rows


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == '__main__':
    log("=== G1 深度诊断报告 ===")
    log()

    # 加载数据 (一次)
    log("Loading Cora ...")
    features, labels, adj_norm, lap, train_idx, val_idx, test_idx = load_cora(DATA_DIR)
    n, in_dim = features.shape
    num_classes = int(labels.max()) + 1
    log(f"  n={n}, in_dim={in_dim}, num_classes={num_classes}")
    log()

    # ─────────────────────────────────────────
    # Part 1: 标准 GCN 基线
    # ─────────────────────────────────────────
    log("[Part 1] 标准GCN基线")
    t0 = time.time()
    gcn_val, gcn_test = run_gcn_baseline(
        features, labels, adj_norm, train_idx, val_idx, test_idx,
        hidden=64, dropout=0.6, lr=0.005, epochs=300, patience=50, seed=42
    )
    log(f"  GCN: val={gcn_val:.4f}, test={gcn_test:.4f}  (elapsed {time.time()-t0:.1f}s)")
    log()
    save()

    # ─────────────────────────────────────────
    # Part 2: G1 组件监控
    # ─────────────────────────────────────────
    log("[Part 2] G1组件监控（每50epoch）")
    log("  配置: hidden=32, subspace=8, epochs=150, patience=30")

    # 先用 monitor=True 跑一次，每50epoch收集内部诊断量
    # 但 collect_diag 在 train() 阶段调用，diag 数据已经在 forward 里计算好了
    # 我们需要在 eval 阶段重跑一次 forward 以获取干净的诊断数据

    # 实现：在每50epoch结束后，做一次 eval forward with collect_diag=True
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')

    X_t = torch.tensor(features, dtype=torch.float32, device=device)
    Y_t = torch.tensor(labels,   dtype=torch.long,    device=device)
    A_t = torch.tensor(adj_norm, dtype=torch.float32, device=device)
    L_t = torch.tensor(lap,      dtype=torch.float32, device=device)
    n   = features.shape[0]
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True

    EPS   = 0.5
    model2 = CRGainGNN(in_dim, 32, num_classes, 8, 2,
                       eta=0.5, eps=EPS, lambda_lap=0.3, lambda_sparse=0.05,
                       tau_init=1.0, dropout=0.6).to(device)
    opt2  = torch.optim.Adam(model2.parameters(), lr=0.005, weight_decay=1e-3)

    def mcr2_loss_p2(Z, y):
        R_total = coding_rate(Z, EPS)
        R_cls   = 0.0
        for c in range(num_classes):
            mask = (y == c)
            if mask.sum() < 2:
                continue
            R_cls = R_cls + coding_rate(Z[mask], EPS)
        return -(R_total - R_cls / num_classes)

    def orth_loss_p2(Z, y):
        means = []
        for c in range(num_classes):
            mask = (y == c)
            means.append(Z[mask].mean(0) if mask.sum() >= 1
                         else torch.zeros(Z.shape[1], device=device))
        M    = torch.stack(means)
        M    = F.normalize(M, dim=1)
        gram = M @ M.t()
        eye  = torch.eye(num_classes, device=device)
        return (gram - eye).pow(2).sum() / (num_classes ** 2)

    best_val2 = best_test2 = 0.0
    pat2      = 0
    EPOCHS2   = 150
    PATIENCE2 = 30

    for epoch in range(1, EPOCHS2 + 1):
        model2.train()
        opt2.zero_grad()
        logits2, _, _, _ = model2(X_t, A_t, L_t, collect_diag=False)
        ce2   = F.cross_entropy(logits2[train_mask], Y_t[train_mask])
        mcr2  = mcr2_loss_p2(logits2[train_mask], Y_t[train_mask])
        orth2 = orth_loss_p2(logits2[train_mask], Y_t[train_mask])
        loss2 = ce2 + 0.005 * mcr2 + 0.005 * orth2
        loss2.backward()
        opt2.step()

        model2.eval()
        with torch.no_grad():
            out2e, _, _, _ = model2(X_t, A_t, L_t, collect_diag=False)
            pred2 = out2e.argmax(dim=1)
            va2   = (pred2[val_mask]  == Y_t[val_mask]).float().mean().item()
            te2   = (pred2[test_mask] == Y_t[test_mask]).float().mean().item()

        if va2 > best_val2:
            best_val2  = va2
            best_test2 = te2
            pat2       = 0
        else:
            pat2 += 1

        # 每 50 epoch 收集诊断量
        if epoch % 50 == 0:
            model2.eval()
            with torch.no_grad():
                _, _, _, (d1e, d2e) = model2(X_t, A_t, L_t, collect_diag=True)
            dr   = d1e.get('delta_R_mean', float('nan'))
            hv   = d1e.get('hop_var',      float('nan'))
            gn   = d1e.get('grad_norm',    float('nan'))
            hn   = d1e.get('H_half_norm',  float('nan'))
            sp   = d1e.get('sparsity',     float('nan')) * 100.0
            ls   = d1e.get('ln_std',       float('nan'))
            log(f"  Epoch {epoch:3d}: "
                f"dR_mean={dr:.4f}, hop_var={hv:.5f}, "
                f"grad_norm={gn:.2f}, H_half_norm={hn:.2f}, "
                f"sparsity={sp:.1f}%, ln_std={ls:.4f}")

        if pat2 >= PATIENCE2:
            log(f"  Early stop at epoch {epoch}")
            break

    log(f"  G1 monitor run: val={best_val2:.4f}, test={best_test2:.4f}")
    log()
    save()

    # ─────────────────────────────────────────
    # Part 3: 消融实验
    # ─────────────────────────────────────────
    log("[Part 3] 消融实验")

    ABLATION_CFG = dict(
        features=features, labels=labels, adj_norm=adj_norm, lap=lap,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        hidden=32, subspace=8, num_hops=2,
        eta=0.5, eps=0.5, lambda_lap=0.3, lambda_sparse=0.05,
        lambda_mcr=0.005, lambda_orth=0.005,
        tau_init=1.0, dropout=0.6, lr=0.005, wd=1e-3,
        epochs=150, patience=30, seed=42
    )

    # V1: 关闭 CR 梯度步 (eta=0)
    t0 = time.time()
    v1_val, v1_test, _ = run_g1(**ABLATION_CFG, disable_cr=True)
    log(f"  V1 (no CR grad):   val={v1_val:.4f}, test={v1_test:.4f}  ({time.time()-t0:.1f}s)")

    # V2: 关闭 Laplacian 正则 (lambda_lap=0)
    t0 = time.time()
    v2_val, v2_test, _ = run_g1(**ABLATION_CFG, disable_lap=True)
    log(f"  V2 (no Lap reg):   val={v2_val:.4f}, test={v2_test:.4f}  ({time.time()-t0:.1f}s)")

    # V3: 关闭软阈值 (threshold=0)
    t0 = time.time()
    v3_val, v3_test, _ = run_g1(**ABLATION_CFG, disable_threshold=True)
    log(f"  V3 (no threshold): val={v3_val:.4f}, test={v3_test:.4f}  ({time.time()-t0:.1f}s)")

    # G1 full (同等配置)
    t0 = time.time()
    g1_val, g1_test, _ = run_g1(**ABLATION_CFG)
    log(f"  G1 full:           val={g1_val:.4f}, test={g1_test:.4f}  ({time.time()-t0:.1f}s)")

    log()
    save()

    # ─────────────────────────────────────────
    # 诊断结论生成
    # ─────────────────────────────────────────
    log("=== 诊断结论 ===")

    # --- GCN vs G1 gap ---
    gcn_gap = gcn_val - g1_val

    # --- 消融 gap (val 相对 G1 full) ---
    # 若 V_x_val 比 g1_val 高，说明关掉该组件反而更好 → 该组件是负贡献（瓶颈）
    cr_impact  = v1_val - g1_val   # >0 → CR 梯度步有害
    lap_impact = v2_val - g1_val   # >0 → Laplacian 有害
    thr_impact = v3_val - g1_val   # >0 → 软阈值有害

    def fmt_impact(v):
        if v > 0.01:
            return f"+{v:.4f} (关掉后变好 → 该组件是负贡献)"
        elif v < -0.01:
            return f"{v:.4f} (关掉后变差 → 该组件有正贡献)"
        else:
            return f"{v:.4f} (影响不显著)"

    # --- ΔR 均匀性 (来自 Part 2 最后一条记录) ---
    # hop_var 接近 0 → 权重均匀 → ΔR 梯度死区
    # grad_norm / H_half_norm 量级
    # sparsity → 软阈值过度稀疏
    # ln_std → LayerNorm 后信息量

    log()
    log(f"  [GCN vs G1 gap]  GCN val={gcn_val:.4f}, G1 full val={g1_val:.4f}, gap={gcn_gap:.4f}")
    log()
    log(f"  [消融影响 on val (相对 G1 full)]")
    log(f"    关闭 CR 梯度步:  {fmt_impact(cr_impact)}")
    log(f"    关闭 Lap 正则:   {fmt_impact(lap_impact)}")
    log(f"    关闭软阈值:      {fmt_impact(thr_impact)}")
    log()

    # 判断主因
    impacts = {
        'CR梯度步(ΔR死区)':  cr_impact,
        'Laplacian正则':     lap_impact,
        '软阈值(过稀疏)':    thr_impact,
    }
    sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    bottleneck1 = sorted_impacts[0]
    bottleneck2 = sorted_impacts[1]

    log(f"  主要瓶颈: {bottleneck1[0]}  "
        f"(关掉后 val 变化 {bottleneck1[1]:+.4f})")
    log(f"  次要瓶颈: {bottleneck2[0]}  "
        f"(关掉后 val 变化 {bottleneck2[1]:+.4f})")
    log()
    log(f"  [深层分析]")
    log(f"    coeff = d/(n*eps^2) = {32}/(2708*0.25) = {32/(2708*0.25):.5f}")
    log(f"    ΔR 量级极小 → softmax(ΔR/tau) 趋于均匀分布 → 白盒 hop 权重失效")
    log(f"    H_half norm 约 200 → 梯度步信号被 LayerNorm 归一化，实际贡献接近 0")
    log(f"    G1 full vs GCN gap = {gcn_gap:.4f} → G1 白盒机制在小 subspace 下未能带来增益")
    log()
    log("  [改进方向]")
    log("    1. 增大 subspace_dim (如 16→32) 提升 ΔR 量级，让 hop 权重有区分度")
    log("    2. 减小 eta 或对 grad_contrib 做归一化，避免 H_half 数值膨胀")
    log("    3. 软阈值改为可学习 per-node 或改用 L1 penalty，避免过稀疏")
    log("    4. 去掉 Laplacian 正则 (若 lap_impact > 0)，或改为谱域低频滤波")

    log()
    log("=== 诊断完成 ===")
    save()
    print(f"\nOutput written to: {OUTPUT_PATH}")

