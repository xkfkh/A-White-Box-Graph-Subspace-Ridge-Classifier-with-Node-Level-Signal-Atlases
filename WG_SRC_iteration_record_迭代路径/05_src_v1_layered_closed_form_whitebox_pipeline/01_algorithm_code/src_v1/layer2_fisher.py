"""
Layer 2 (V2): Fisher-LDA 判别子空间 + 类内 PCA 子空间

改进动机（来自 Layer 5 智能体的反馈）：
    原始 Layer 2 只用类内 PCA，导致：
    1. sub_dim>=32 时 var_ratio=1.0，子空间解释了全部方差，R_perp 失去判别力
    2. Mahalanobis 距离在训练集上过拟合（训练节点参与了子空间构造）
    3. 投影能量占比 I_c 对训练节点总是 1.0

    Fisher-LDA 子空间直接最大化类间/类内散度比，是更好的判别子空间。
    在 Cora 上 Fisher-LDA + 最近质心达到 test=0.814，比 PCA R_perp 的 0.792 提升 2.2pp。

数学定义：
    1. 计算类内散度矩阵 S_W 和类间散度矩阵 S_B：
       S_W = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
       S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T

    2. 求解广义特征值问题：
       S_B w = lambda S_W w
       等价于 S_W^{-1} S_B w = lambda w

    3. 取最大的 d 个特征向量作为投影矩阵 W [D, d]

    4. 投影: Z_proj = Z @ W  [n, d]

    5. 在投影空间中计算每类的均值和协方差

符号说明：
    S_W    : [D, D]  类内散度矩阵
    S_B    : [D, D]  类间散度矩阵
    W      : [D, d]  Fisher 判别方向（特征向量）
    mu_c   : [D]     类 c 在原始空间的均值
    mu_c_proj : [d]  类 c 在 Fisher 空间的均值

几何意义：
    Fisher-LDA 找到的方向同时满足：
    - 类间距离最大（不同类的投影均值尽量远）
    - 类内距离最小（同类样本的投影尽量紧凑）
    这是白盒可解释的最优判别方向。

    LDA 最多有 C-1 个非零特征值（C 个类均值在 C-1 维子空间中），
    所以 d <= C-1 是理论上限。但实际中可以取更多方向来保留更多信息。
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def build_fisher_subspace(Z, Y, train_idx, num_classes, sub_dim=8, reg=1e-4):
    """
    Fisher-LDA 判别子空间构造。

    参数：
        Z:           [n, D]  平滑后节点特征
        Y:           [n]     标签
        train_idx:   list    训练节点索引
        num_classes: int     类别数
        sub_dim:     int     子空间维度（建议 6~10，不超过 C-1 太多）
        reg:         float   S_W 正则化参数

    返回 dict：
        W:              [D, d]           Fisher 投影矩阵
        mu_dict:        {c: Tensor[D]}   原始空间类均值
        mu_proj_dict:   {c: Tensor[d]}   Fisher 空间类均值
        cov_proj_dict:  {c: Tensor[d,d]} Fisher 空间类协方差
        eigenvalues:    [d]              Fisher 特征值（判别力指标）
        sub_dim_actual: int              实际子空间维度
        n_c_dict:       {c: int}         每类训练样本数
    """
    train_t = torch.tensor(train_idx, dtype=torch.long)
    Z_tr = Z[train_t]
    Y_tr = Y[train_t]
    n_train = len(train_idx)
    D = Z.shape[1]

    # 全局均值
    mu_global = Z_tr.mean(dim=0)  # [D]

    # 类内散度 S_W 和类间散度 S_B
    S_W = torch.zeros(D, D)
    S_B = torch.zeros(D, D)
    mu_dict = {}
    n_c_dict = {}

    for c in range(num_classes):
        mask = (Y_tr == c)
        n_c = mask.sum().item()
        n_c_dict[c] = n_c

        if n_c < 1:
            mu_dict[c] = mu_global.clone()
            continue

        Z_c = Z_tr[mask]
        mu_c = Z_c.mean(dim=0)
        mu_dict[c] = mu_c

        if n_c >= 2:
            Z_centered = Z_c - mu_c
            S_W += Z_centered.t() @ Z_centered

        diff = (mu_c - mu_global).unsqueeze(1)  # [D, 1]
        S_B += n_c * (diff @ diff.t())

    # 正则化 S_W（Friedman 收缩估计）
    # S_W_reg = (1 - shrinkage) * S_W + shrinkage * tr(S_W)/D * I
    # 这在高维小样本下比简单加 reg*I 更稳定
    trace_SW = torch.trace(S_W)
    shrinkage = min(1.0, reg * D / (trace_SW + 1e-10))
    S_W = (1 - shrinkage) * S_W + (shrinkage * trace_SW / D + reg) * torch.eye(D)

    # 求解广义特征值问题 S_B v = lambda S_W v
    # 用 Cholesky 分解转化为标准对称特征值问题:
    #   S_W = L L^T  (Cholesky)
    #   令 u = L^T v, 则 S_B v = lambda L L^T v
    #   L^{-1} S_B L^{-T} u = lambda u  (标准对称问题)
    #   v = L^{-T} u  (变换回原空间)
    try:
        L_chol = torch.linalg.cholesky(S_W)
        # Step 1: 计算 L^{-1} S_B
        LinvSB = torch.linalg.solve_triangular(L_chol, S_B, upper=False)
        # Step 2: 计算 (L^{-1} S_B) L^{-T} = L^{-1} S_B L^{-T}
        S_B_transformed = torch.linalg.solve_triangular(
            L_chol, LinvSB.t(), upper=False
        ).t()
        # 强制对称（消除浮点误差）
        S_B_transformed = 0.5 * (S_B_transformed + S_B_transformed.t())
        eigenvalues, eigenvectors_transformed = torch.linalg.eigh(S_B_transformed)
        # 变换回原始空间: v = L^{-T} u
        eigenvectors = torch.linalg.solve_triangular(
            L_chol.t(), eigenvectors_transformed, upper=True
        )
    except Exception:
        # 回退到直接求逆
        S_W_inv = torch.linalg.inv(S_W)
        M = S_W_inv @ S_B
        M = 0.5 * (M + M.t())
        eigenvalues, eigenvectors = torch.linalg.eigh(M)

    # 取最大的 sub_dim 个
    idx_sorted = eigenvalues.argsort(descending=True)
    d = min(sub_dim, num_classes - 1, D, n_train)
    d = max(d, 1)

    W = eigenvectors[:, idx_sorted[:d]].contiguous()  # [D, d]
    evals = eigenvalues[idx_sorted[:d]]

    # 在 Fisher 空间中计算类均值和协方差
    Z_proj_all = Z @ W  # [n, d]
    mu_proj_dict = {}
    cov_proj_dict = {}

    for c in range(num_classes):
        mask = (Y_tr == c)
        n_c = mask.sum().item()

        if n_c < 1:
            mu_proj_dict[c] = torch.zeros(d)
            cov_proj_dict[c] = torch.eye(d)
            continue

        Z_c_proj = Z_proj_all[train_t][mask]
        mu_proj_dict[c] = Z_c_proj.mean(dim=0)

        if n_c >= 2:
            Z_c_centered = Z_c_proj - mu_proj_dict[c]
            cov_proj_dict[c] = (Z_c_centered.t() @ Z_c_centered) / (n_c - 1) + reg * torch.eye(d)
        else:
            cov_proj_dict[c] = torch.eye(d)

    return {
        'W': W,
        'mu_dict': mu_dict,
        'mu_proj_dict': mu_proj_dict,
        'cov_proj_dict': cov_proj_dict,
        'eigenvalues': evals,
        'sub_dim_actual': d,
        'n_c_dict': n_c_dict,
    }


def classify_fisher_centroid(Z, fisher_result, num_classes):
    """
    Fisher 空间最近质心分类器。

    Score_c(z) = -||W^T z - mu_c^{proj}||^2

    predict = argmax_c Score_c(z)

    这是最简单、最干净的白盒分类器。
    """
    W = fisher_result['W']
    Z_proj = Z @ W  # [n, d]
    n = Z.shape[0]

    scores = torch.zeros(n, num_classes)
    for c in range(num_classes):
        mu_c = fisher_result['mu_proj_dict'][c]
        scores[:, c] = -((Z_proj - mu_c) ** 2).sum(dim=1)

    return scores.argmax(dim=1), scores


def classify_fisher_mahalanobis(Z, fisher_result, num_classes):
    """
    Fisher 空间 Mahalanobis 距离分类器。

    Score_c(z) = -(z_proj - mu_c)^T Sigma_c^{-1} (z_proj - mu_c)

    比最近质心更精细，考虑了各方向的方差。
    """
    W = fisher_result['W']
    Z_proj = Z @ W
    n = Z.shape[0]

    scores = torch.zeros(n, num_classes)
    for c in range(num_classes):
        mu_c = fisher_result['mu_proj_dict'][c]
        cov_inv = torch.linalg.inv(fisher_result['cov_proj_dict'][c])
        diff = Z_proj - mu_c
        scores[:, c] = -(diff @ cov_inv * diff).sum(dim=1)

    return scores.argmax(dim=1), scores

