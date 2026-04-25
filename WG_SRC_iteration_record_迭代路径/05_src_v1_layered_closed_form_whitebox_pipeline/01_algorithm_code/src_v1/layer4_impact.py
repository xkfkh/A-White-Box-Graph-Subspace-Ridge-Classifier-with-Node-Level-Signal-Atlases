"""
Layer 4: 内部差距作用量化层

同时使用两个度量：

(A) 子空间外残差 R_perp_c(z)
    R_perp_c(z) = ||(I - B_c B_c^T)(z - mu_c)||^2

    符号说明：
        B_c    : [D, d]  类 c 子空间正交基
        mu_c   : [D]     类 c 均值
        I      : [D, D]  单位矩阵
        (I - B_c B_c^T) : 子空间正交补投影算子

    几何意义：
        R_perp_c(z) 衡量 z 偏离 mu_c 后，有多少"残留"在子空间外面。
        - R_perp_c 小：z 的偏差几乎完全被子空间解释，z 很可能属于类 c
        - R_perp_c 大：z 有大量信息不在类 c 子空间中，z 不太可能属于类 c

    数值小例子 (D=4, d=2):
        B_c 张成前两维，z - mu_c = [1, 1, 2, 3]
        B_c B_c^T (z-mu_c) = [1, 1, 0, 0]  (子空间内分量)
        (I - B_c B_c^T)(z-mu_c) = [0, 0, 2, 3]  (子空间外分量)
        R_perp_c = 0 + 0 + 4 + 9 = 13

(B) 子空间内 Mahalanobis 差异 M_c(z)
    a_c(z) = B_c^T (z - mu_c)                    [d]
    M_c(z) = a_c(z)^T (Sigma_c + eta*I)^{-1} a_c(z)   标量

    符号说明：
        a_c(z)  : [d]    z 在类 c 子空间中的坐标
        Sigma_c : [d, d] 类 c 在子空间中的协方差（对角矩阵，来自 Layer 2）
        eta     : 正则化参数，防止协方差奇异

    几何意义：
        M_c(z) 是 z 在子空间内的 Mahalanobis 距离。
        它不仅看 z 在子空间中离均值多远，还考虑各方向的方差：
        - 方差大的方向（类内变化大）：距离贡献小（这个方向上偏离是正常的）
        - 方差小的方向（类内变化小）：距离贡献大（这个方向上偏离是异常的）

    数值小例子 (d=2):
        a_c = [2, 0.5]
        Sigma_c = diag(4, 0.25)  (第一维方差大，第二维方差小)
        (Sigma_c + eta*I)^{-1} ≈ diag(1/4, 1/0.25) = diag(0.25, 4)
        M_c = 2^2 * 0.25 + 0.5^2 * 4 = 1 + 1 = 2
        虽然 a_c 在第一维偏离更大(2 vs 0.5)，但归一化后贡献相同。
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_orthogonal_residual(Z, subspace_result, num_classes):
    """
    计算子空间外残差 R_perp_c(z) = ||(I - B_c B_c^T)(z - mu_c)||^2

    参数：
        Z:               [n, D]  平滑后节点特征
        subspace_result:  Layer 2 输出
        num_classes:      int

    返回：
        R_perp:          [n, C]  子空间外残差矩阵
                         R_perp[i, c] = R_perp_c(z_i)
    """
    n = Z.shape[0]
    R_perp = torch.zeros(n, num_classes)

    for c in range(num_classes):
        if c not in subspace_result['B_dict']:
            R_perp[:, c] = 1e10
            continue

        mu_c = subspace_result['mu_dict'][c]   # [D]
        B_c = subspace_result['B_dict'][c]     # [D, d]

        diff = Z - mu_c                        # [n, D]

        # 子空间内投影: B_c B_c^T (z - mu_c)
        proj_coord = diff @ B_c                # [n, d]
        proj_back = proj_coord @ B_c.t()       # [n, D]

        # 子空间外残差
        residual = diff - proj_back            # [n, D]
        R_perp[:, c] = (residual ** 2).sum(dim=1)  # [n]

    return R_perp


def compute_mahalanobis_distance(Z, subspace_result, num_classes, eta=1e-4):
    """
    计算子空间内 Mahalanobis 距离 M_c(z)。

    M_c(z) = a_c^T (Sigma_c + eta*I)^{-1} a_c
    其中 a_c = B_c^T (z - mu_c)

    参数：
        Z:               [n, D]  平滑后节点特征
        subspace_result:  Layer 2 输出
        num_classes:      int
        eta:             float   正则化参数

    返回：
        M_dist:          [n, C]  Mahalanobis 距离矩阵
    """
    n = Z.shape[0]
    M_dist = torch.zeros(n, num_classes)

    for c in range(num_classes):
        if c not in subspace_result['B_dict']:
            M_dist[:, c] = 1e10
            continue

        mu_c = subspace_result['mu_dict'][c]       # [D]
        B_c = subspace_result['B_dict'][c]         # [D, d]
        Sigma_c = subspace_result['Sigma_dict'][c] # [d, d]

        diff = Z - mu_c                            # [n, D]
        a_c = diff @ B_c                           # [n, d]

        # (Sigma_c + eta * I)^{-1}
        # Sigma_c 是对角矩阵，所以逆也是对角的
        sigma_diag = torch.diag(Sigma_c).clamp(min=0) + eta  # [d]
        inv_sigma = 1.0 / sigma_diag               # [d]

        # M_c(z) = sum_k a_c_k^2 / sigma_k
        M_dist[:, c] = (a_c ** 2 * inv_sigma.unsqueeze(0)).sum(dim=1)  # [n]

    return M_dist


def compute_impact_metrics(Z, subspace_result, num_classes, Y, train_idx, eta=1e-4):
    """
    计算完整的 Layer 4 影响指标。

    返回 dict:
        R_perp:          [n, C]  子空间外残差
        M_dist:          [n, C]  Mahalanobis 距离
        fisher_ratio_R:  float   R_perp 的 Fisher 判别比
        fisher_ratio_M:  float   M_dist 的 Fisher 判别比
    """
    R_perp = compute_orthogonal_residual(Z, subspace_result, num_classes)
    M_dist = compute_mahalanobis_distance(Z, subspace_result, num_classes, eta)

    # 计算 Fisher 比（用于 Layer 5 权重确定）
    train_t = torch.tensor(train_idx, dtype=torch.long)
    Y_tr = Y[train_t]

    fisher_R = _fisher_for_distance(R_perp[train_t], Y_tr, num_classes)
    fisher_M = _fisher_for_distance(M_dist[train_t], Y_tr, num_classes)

    return {
        'R_perp': R_perp,
        'M_dist': M_dist,
        'fisher_ratio_R': fisher_R,
        'fisher_ratio_M': fisher_M,
    }


def _fisher_for_distance(dist_matrix, Y, num_classes):
    """
    对距离矩阵计算 Fisher 判别比。

    对距离度量，好的判别力意味着：
    - 真实类的距离小（类内距离小）
    - 其他类的距离大（类间距离大）

    Fisher ratio = (mean_other - mean_self)^2 / (var_self + var_other)
    """
    self_dists = []
    other_dists = []

    for i in range(Y.shape[0]):
        c = Y[i].item()
        self_dists.append(dist_matrix[i, c].item())
        for j in range(num_classes):
            if j != c:
                other_dists.append(dist_matrix[i, j].item())

    if not self_dists or not other_dists:
        return 0.0

    self_t = torch.tensor(self_dists)
    other_t = torch.tensor(other_dists)

    mu_self = self_t.mean()
    mu_other = other_t.mean()
    var_self = self_t.var().clamp(min=1e-12)
    var_other = other_t.var().clamp(min=1e-12)

    fisher = ((mu_other - mu_self) ** 2) / (var_self + var_other)
    return fisher.item()

