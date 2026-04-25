"""
Layer 2: 类内 PCA/SVD 子空间构造
Phase 2 智能体A

数学定义：
对每类 c，给定平滑后特征矩阵 Z [n, D]：
  1. X_c = Z[train_idx where Y==c]          [n_c, D]
  2. mu_c = mean(X_c, axis=0)               [D]
  3. X_c_centered = X_c - mu_c              [n_c, D]
  4. SVD: X_c_centered = U_c diag(S_c) V_c^T
     U_c [n_c, k], S_c [k], V_c [D, k], k=min(n_c,D)
  5. B_c = V_c[:, :d]                       [D, d]  右奇异向量 = 主方向基
  6. Sigma_c = diag(S_c[:d]^2 / (n_c-1))   [d, d]  子空间内协方差

几何意义：
  B_c 的列向量是类 c 样本在 D 维空间中变化最大的 d 个方向。
  投影坐标 p = B_c^T (z - mu_c) 是 z 在子空间中的低维坐标。
  Sigma_c 描述各主方向上的方差大小，用于马氏距离归一化。

小数值例子 (D=4, d=2, n_c=5)：
  X_c_centered [5,4] -> SVD -> V_c [4,4]
  B_c = V_c[:,:2]  [4,2]，是 4 维空间中的 2 维子空间基
  对新样本 z [4]：投影坐标 = B_c^T (z - mu_c)  [2]
  投影能量 = ||B_c^T (z - mu_c)||^2  (标量)
"""

import os
import torch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def build_class_subspaces(Z, Y, train_idx, num_classes, sub_dim=32):
    """
    对每类构造 PCA/SVD 子空间。

    参数：
      Z:           [n, D]  平滑后节点特征
      Y:           [n]     节点标签 (0~C-1)
      train_idx:   list    训练集节点索引
      num_classes: int     类别数
      sub_dim:     int     子空间维度 d

    返回 dict：
      mu_dict:        {c: Tensor[D]}      每类均值
      B_dict:         {c: Tensor[D, d]}   子空间正交基（右奇异向量）
      Sigma_dict:     {c: Tensor[d, d]}   子空间内对角协方差
      S_dict:         {c: Tensor[d]}      前 d 个奇异值
      var_ratio_dict: {c: float}          前 d 个奇异值解释的方差比例
      sub_dim_actual: {c: int}            实际子空间维度（可能 < sub_dim）
      n_c_dict:       {c: int}            每类训练样本数
    """
    train_t = torch.tensor(train_idx, dtype=torch.long)
    Z_tr = Z[train_t]   # [n_train, D]
    Y_tr = Y[train_t]   # [n_train]
    D = Z.shape[1]

    mu_dict, B_dict, Sigma_dict, S_dict = {}, {}, {}, {}
    var_ratio_dict, sub_dim_actual, n_c_dict = {}, {}, {}

    for c in range(num_classes):
        mask = (Y_tr == c)
        n_c = mask.sum().item()
        n_c_dict[c] = n_c

        if n_c < 2:
            d = 1
            mu_dict[c] = torch.zeros(D)
            B_dict[c] = torch.zeros(D, d)
            Sigma_dict[c] = torch.zeros(d, d)
            S_dict[c] = torch.zeros(d)
            var_ratio_dict[c] = 0.0
            sub_dim_actual[c] = d
            continue

        X_c = Z_tr[mask]                    # [n_c, D]
        mu_c = X_c.mean(dim=0)              # [D]
        Xc = X_c - mu_c                     # [n_c, D] 中心化

        # SVD: full_matrices=False -> U[n_c,k], S[k], Vt[k,D], k=min(n_c,D)
        _, S_full, Vt = torch.linalg.svd(Xc, full_matrices=False)
        V = Vt.t()                          # [D, k]

        k = S_full.shape[0]
        d = min(sub_dim, k)

        B_c = V[:, :d].clone()              # [D, d]
        S_c = S_full[:d].clone()            # [d]
        var_c = S_c ** 2 / max(n_c - 1, 1) # [d] 各主方向方差
        Sigma_c = torch.diag(var_c)         # [d, d]

        total_var = (S_full ** 2).sum().item()
        expl_var = (S_c ** 2).sum().item()
        ratio = expl_var / total_var if total_var > 1e-12 else 0.0

        mu_dict[c] = mu_c
        B_dict[c] = B_c
        Sigma_dict[c] = Sigma_c
        S_dict[c] = S_c
        var_ratio_dict[c] = ratio
        sub_dim_actual[c] = d

    return {
        'mu_dict': mu_dict,
        'B_dict': B_dict,
        'Sigma_dict': Sigma_dict,
        'S_dict': S_dict,
        'var_ratio_dict': var_ratio_dict,
        'sub_dim_actual': sub_dim_actual,
        'n_c_dict': n_c_dict,
    }


def project_to_subspace(Z, subspace_result):
    """
    将所有节点投影到每个类的子空间。

    返回：
      proj_dict: {c: Tensor[n, d]}  proj_dict[c][i] = B_c^T (z_i - mu_c)
    """
    proj_dict = {}
    for c, B_c in subspace_result['B_dict'].items():
        mu_c = subspace_result['mu_dict'][c]
        proj_dict[c] = (Z - mu_c) @ B_c    # [n, d]
    return proj_dict


def classify_by_proj_energy(Z, subspace_result, num_classes):
    """
    分类器 1：argmax_c ||B_c^T (z - mu_c)||^2

    几何意义：选择使投影能量最大的类（z 在哪个类子空间中"解释度"最高）。
    问题：高维子空间天然有更大投影能量，需配合维度归一化。
    """
    n = Z.shape[0]
    scores = torch.zeros(n, num_classes)
    proj_dict = project_to_subspace(Z, subspace_result)
    for c in range(num_classes):
        if c in proj_dict:
            d_c = max(subspace_result['sub_dim_actual'][c], 1)
            scores[:, c] = (proj_dict[c] ** 2).sum(dim=1) / d_c
    return scores.argmax(dim=1), scores


def classify_by_residual(Z, subspace_result, num_classes):
    """
    分类器 2：argmin_c ||(I - B_c B_c^T)(z - mu_c)||^2

    几何意义：选择使 z 到类子空间距离最小的类。
    残差 = 总距离 - 投影能量 = ||z-mu_c||^2 - ||B_c^T(z-mu_c)||^2
    """
    n = Z.shape[0]
    neg_resid = torch.zeros(n, num_classes)
    proj_dict = project_to_subspace(Z, subspace_result)
    for c in range(num_classes):
        mu_c = subspace_result['mu_dict'][c]
        z_c = Z - mu_c                              # [n, D]
        dist_sq = (z_c ** 2).sum(dim=1)             # [n]
        proj_sq = (proj_dict[c] ** 2).sum(dim=1)    # [n]
        residual = dist_sq - proj_sq                # [n]
        neg_resid[:, c] = -residual
    return neg_resid.argmax(dim=1), neg_resid


def classify_by_mahalanobis(Z, subspace_result, num_classes):
    """
    分类器 3：argmin_c (z-mu_c)^T B_c Sigma_c^{-1} B_c^T (z-mu_c)

    展开：sum_k [(B_c^T(z-mu_c))_k]^2 / sigma_k^2

    几何意义：在子空间内用各主方向的方差归一化，
    方差小的方向（更"确定"的方向）贡献更大的判别力。
    这是 LDA 在子空间内的白盒近似。
    """
    n = Z.shape[0]
    neg_dist = torch.zeros(n, num_classes)
    proj_dict = project_to_subspace(Z, subspace_result)
    for c in range(num_classes):
        sigma_sq = torch.diag(subspace_result['Sigma_dict'][c]).clamp(min=1e-8)  # [d]
        proj = proj_dict[c]                                 # [n, d]
        maha = (proj ** 2 / sigma_sq.unsqueeze(0)).sum(dim=1)  # [n]
        neg_dist[:, c] = -maha
    return neg_dist.argmax(dim=1), neg_dist


def classify_by_subspace(Z, subspace_result, num_classes):
    """默认分类器：使用投影能量（维度归一化）。"""
    return classify_by_proj_energy(Z, subspace_result, num_classes)


def principal_angles_between_subspaces(B1, B2):
    """
    计算两个子空间之间的主角度。

    数学：
      B1 [D, d1], B2 [D, d2] 是正交基
      M = B1^T B2  [d1, d2]
      主角度 = arccos(svdvals(M))

    几何意义：主角度越接近 pi/2，两个子空间越正交（类间分离越好）。

    返回：angles [min(d1,d2)] 主角度（弧度）
    """
    M = B1.t() @ B2
    S = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    return torch.acos(S)
