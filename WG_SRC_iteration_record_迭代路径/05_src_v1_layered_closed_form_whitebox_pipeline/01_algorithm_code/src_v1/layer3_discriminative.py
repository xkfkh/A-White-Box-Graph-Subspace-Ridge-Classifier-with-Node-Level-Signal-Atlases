"""
Layer 3 V2: 子空间判别方向加权

核心创新：
    不是简单计算 R_perp = ||z - mu_c||^2 - ||B_c^T(z-mu_c)||^2，
    而是对 B_c 的每个方向 b_k 计算其"判别独特性"权重，
    只保留那些真正能区分类 c 与其他类的方向的投影能量。

数学定义：
    对类 c 的子空间基 B_c = [b_1, ..., b_d]，第 k 个方向的判别权重为：

    w_k^{(c)} = 1 - max_{c' != c} ||B_{c'}^T b_k||^2

符号说明：
    b_k        : [D]     B_c 的第 k 个列向量（第 k 个主方向）
    B_{c'}     : [D, d]  类 c' 的子空间正交基
    B_{c'}^T b_k : [d]   b_k 在类 c' 子空间中的投影坐标
    ||B_{c'}^T b_k||^2 : 标量，b_k 被类 c' 子空间"解释"的比例

    w_k = 1 意味着 b_k 完全不被任何其他类子空间解释（高判别力）
    w_k = 0 意味着 b_k 完全在某个其他类的子空间中（零判别力）

几何意义：
    在高维空间中，不同类的子空间可能有重叠方向。
    传统 R_perp 把所有子空间方向等权对待，
    但重叠方向对分类没有帮助（两个类都能解释这个方向）。

    判别加权只保留每个类"独有"的方向——
    这些方向是该类区别于其他类的关键特征。

    这等价于在子空间中做一个"白盒注意力"：
    给判别力强的方向更大权重，给共享方向更小权重。

数值小例子 (D=4, d=2):
    类 0 的子空间: B_0 = [[1,0], [0,1], [0,0], [0,0]]  (前两维)
    类 1 的子空间: B_1 = [[1,0], [0,0], [0,1], [0,0]]  (第1和第3维)

    B_0 的第1列 b_1 = [1,0,0,0]:
      ||B_1^T b_1||^2 = 1^2 = 1.0  (完全在类1子空间中)
      w_1 = 1 - 1.0 = 0.0  (这个方向没有判别力)

    B_0 的第2列 b_2 = [0,1,0,0]:
      ||B_1^T b_2||^2 = 0  (不在类1子空间中)
      w_2 = 1 - 0 = 1.0  (这个方向有判别力)

    加权 R_perp: 只用第2个方向的投影能量来判别

判别加权 R_perp 公式：
    R_disc_c(z) = ||z - mu_c||^2 - sum_k w_k * (b_k^T (z - mu_c))^2

代码实现：
    对每类 c:
    1. 遍历 B_c 的每个方向 b_k
    2. 计算 w_k = 1 - max_{c'!=c} ||B_{c'}^T b_k||^2
    3. 用加权投影能量替代原始投影能量
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_discriminative_weights(B_dict, num_classes):
    """
    计算每类每方向的判别权重。

    参数：
        B_dict:      {c: Tensor[D, d]}  每类的子空间正交基
        num_classes: int

    返回：
        weights_dict: {c: Tensor[d]}    每类每方向的判别权重
    """
    weights_dict = {}

    for c in range(num_classes):
        if c not in B_dict:
            continue
        B_c = B_dict[c]  # [D, d]
        d = B_c.shape[1]
        weights = torch.ones(d)

        for k in range(d):
            b_k = B_c[:, k]  # [D]
            max_overlap = 0.0

            for c2 in range(num_classes):
                if c2 == c or c2 not in B_dict:
                    continue
                B_c2 = B_dict[c2]  # [D, d2]
                # ||B_c2^T b_k||^2 = b_k 被类 c2 子空间解释的比例
                overlap = (b_k @ B_c2).pow(2).sum().item()
                max_overlap = max(max_overlap, overlap)

            weights[k] = 1.0 - max_overlap

        # 下限截断，防止权重为零
        weights = weights.clamp(min=0.01)
        weights_dict[c] = weights

    return weights_dict


def compute_discriminative_rperp(Z, subspace_result, num_classes):
    """
    计算判别加权 R_perp。

    R_disc_c(z) = ||z - mu_c||^2 - sum_k w_k * (b_k^T (z - mu_c))^2

    参数：
        Z:               [n, D]  平滑后节点特征
        subspace_result:  Layer 2 输出
        num_classes:      int

    返回：
        R_disc:          [n, C]  判别加权残差矩阵
        disc_weights:    {c: Tensor[d]}  判别权重
    """
    n = Z.shape[0]
    B_dict = subspace_result['B_dict']
    mu_dict = subspace_result['mu_dict']

    disc_weights = compute_discriminative_weights(B_dict, num_classes)

    R_disc = torch.zeros(n, num_classes)

    for c in range(num_classes):
        if c not in B_dict:
            R_disc[:, c] = 1e10
            continue

        mu_c = mu_dict[c]     # [D]
        B_c = B_dict[c]       # [D, d]
        w_c = disc_weights[c] # [d]

        diff = Z - mu_c                              # [n, D]
        total_sq = (diff ** 2).sum(dim=1)            # [n]
        proj = diff @ B_c                            # [n, d]
        weighted_proj_sq = (proj ** 2) * w_c.unsqueeze(0)  # [n, d]

        R_disc[:, c] = total_sq - weighted_proj_sq.sum(dim=1)

    return R_disc, disc_weights


def analyze_discriminative_weights(disc_weights, num_classes):
    """
    分析判别权重的统计特性。

    返回 dict:
        avg_weight:     每类的平均权重
        min_weight:     每类的最小权重
        high_disc_dims: 每类有多少方向的权重 > 0.5（高判别力方向）
    """
    result = {}
    for c in range(num_classes):
        if c not in disc_weights:
            continue
        w = disc_weights[c]
        result[c] = {
            'avg_weight': w.mean().item(),
            'min_weight': w.min().item(),
            'max_weight': w.max().item(),
            'high_disc_dims': (w > 0.5).sum().item(),
            'total_dims': len(w),
        }
    return result

