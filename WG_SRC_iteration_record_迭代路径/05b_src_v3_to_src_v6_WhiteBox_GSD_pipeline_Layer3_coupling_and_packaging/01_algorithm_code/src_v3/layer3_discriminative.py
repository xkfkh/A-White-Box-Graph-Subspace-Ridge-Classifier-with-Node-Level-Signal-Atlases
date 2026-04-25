"""
Layer 3 V3: 子空间判别方向加权（反应版，可直接替换原 layer3_discriminative.py）
=======================================================================

核心思想
--------
原版：
    w_k^(c) = 1 - max_{c' != c} ||B_{c'}^T b_k||^2

本版改成三步：
1. 先把 max 改成整体交联网络：
    K_{c,c'} = B_c^T B_{c'}
    w_base,k^(c) = 1 / (1 + sum_{c'!=c} sum_j K_{c,c'}[k,j]^2)

2. 再把类间关系按符号拆开：
    M_{c,c'}^(+) = max(K_{c,c'}, 0)
    M_{c,c'}^(-) = max(-K_{c,c'}, 0)

3. 对当前样本 z，做样本相关的动态反应：
    u_c(z) = B_c^T (z - mu_c)
    r_{+,k}^(c)(z) = sum_{c'!=c} sum_j M_{c,c'}^(+)[k,j] * |u_{c',j}(z)|
    r_{-,k}^(c)(z) = sum_{c'!=c} sum_j M_{c,c'}^(-)[k,j] * |u_{c',j}(z)|

    w_tilde,k^(c)(z)
      = clip(
            w_base,k^(c)
            + eta_pos * r_{+,k}^(c)(z)
            - eta_neg * r_{-,k}^(c)(z),
            weight_min, weight_max
        )

最终判别残差：
    R_react,c(z) = ||z - mu_c||^2 - sum_k w_tilde,k^(c)(z) * u_{c,k}(z)^2

解释：
- w_base 解决“max 丢信息”的问题
- M^(+) / M^(-) 解决“只有抑制、没有增强”的问题
- |u_{c',j}(z)| 让不同子空间之间的影响进入“互乘量级”
- 整个模型仍然是白盒，可逐项解释
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_pairwise_reaction_matrices(B_dict, num_classes):
    """
    计算所有类对子空间的成对关系矩阵。

    返回：
        K_dict:      {(c, c2): Tensor[d_c, d_c2]}
        K_pos_dict:  {(c, c2): Tensor[d_c, d_c2]}
        K_neg_dict:  {(c, c2): Tensor[d_c, d_c2]}
    """
    K_dict, K_pos_dict, K_neg_dict = {}, {}, {}

    for c in range(num_classes):
        if c not in B_dict:
            continue
        B_c = B_dict[c]
        for c2 in range(num_classes):
            if c2 == c or c2 not in B_dict:
                continue
            K = B_c.t() @ B_dict[c2]                  # [d_c, d_c2]
            K_dict[(c, c2)] = K
            K_pos_dict[(c, c2)] = torch.clamp(K, min=0.0)
            K_neg_dict[(c, c2)] = torch.clamp(-K, min=0.0)

    return K_dict, K_pos_dict, K_neg_dict


def compute_discriminative_weights(B_dict, num_classes):
    """
    计算基础判别权重（整体交联网络版）。

    数学：
        w_base,k^(c) = 1 / (1 + sum_{c'!=c} sum_j K_{c,c'}[k,j]^2)

    返回：
        weights_dict: {c: Tensor[d_c]}
    """
    weights_dict = {}

    for c in range(num_classes):
        if c not in B_dict:
            continue
        B_c = B_dict[c]
        d_c = B_c.shape[1]
        agg = torch.zeros(d_c, dtype=B_c.dtype, device=B_c.device)

        for c2 in range(num_classes):
            if c2 == c or c2 not in B_dict:
                continue
            K = B_c.t() @ B_dict[c2]                 # [d_c, d_c2]
            agg += (K ** 2).sum(dim=1)               # [d_c]

        weights_dict[c] = 1.0 / (1.0 + agg)

    return weights_dict


def compute_discriminative_rperp(
    Z,
    subspace_result,
    num_classes,
    eta_pos=0.10,
    eta_neg=0.30,
    weight_min=0.01,
    weight_max=10.0,
):
    """
    计算反应版判别残差。

    这仍然保留原接口名字，所以可以直接替换原文件，不用改 pipeline。

    参数：
        Z:               [n, D]  平滑后节点特征
        subspace_result: Layer 2 输出
        num_classes:     int
        eta_pos:         正反应增强系数
        eta_neg:         负反应抑制系数
        weight_min:      动态权重下界
        weight_max:      动态权重上界

    返回：
        R_disc:          [n, C]
        disc_weights:    dict，包含 base_weights / dynamic_weights / reaction matrices
    """
    n = Z.shape[0]
    B_dict = subspace_result['B_dict']
    mu_dict = subspace_result['mu_dict']

    base_weights = compute_discriminative_weights(B_dict, num_classes)
    K_dict, K_pos_dict, K_neg_dict = compute_pairwise_reaction_matrices(B_dict, num_classes)

    R_disc = torch.zeros(n, num_classes, dtype=Z.dtype, device=Z.device)
    proj_dict = {}
    dyn_weights_dict = {}

    # 先统一算所有类对子空间坐标
    for c in range(num_classes):
        if c not in B_dict:
            continue
        proj_dict[c] = (Z - mu_dict[c]) @ B_dict[c]      # [n, d_c]

    abs_proj_dict = {c: proj.abs() for c, proj in proj_dict.items()}

    for c in range(num_classes):
        if c not in B_dict:
            R_disc[:, c] = 1e10
            continue

        mu_c = mu_dict[c]           # [D]
        u_c = proj_dict[c]          # [n, d_c]
        diff = Z - mu_c             # [n, D]
        total_sq = (diff ** 2).sum(dim=1)

        r_pos = torch.zeros_like(u_c)
        r_neg = torch.zeros_like(u_c)

        for c2 in range(num_classes):
            if c2 == c or c2 not in B_dict:
                continue
            r_pos += abs_proj_dict[c2] @ K_pos_dict[(c, c2)].t()
            r_neg += abs_proj_dict[c2] @ K_neg_dict[(c, c2)].t()

        dyn_w = base_weights[c].unsqueeze(0) + eta_pos * r_pos - eta_neg * r_neg
        dyn_w = dyn_w.clamp(min=weight_min, max=weight_max)    # [n, d_c]
        dyn_weights_dict[c] = dyn_w

        R_disc[:, c] = total_sq - (dyn_w * (u_c ** 2)).sum(dim=1)

    disc_weights = {
        'base_weights': base_weights,
        'dynamic_weights': dyn_weights_dict,
        'K': K_dict,
        'K_pos': K_pos_dict,
        'K_neg': K_neg_dict,
        'eta_pos': eta_pos,
        'eta_neg': eta_neg,
        'weight_min': weight_min,
        'weight_max': weight_max,
    }

    return R_disc, disc_weights


def analyze_discriminative_weights(disc_weights, num_classes=None, idx=None):
    """
    分析基础权重和动态权重。

    参数：
        disc_weights: compute_discriminative_rperp 的第二个返回值
        num_classes:  可选，不传则自动从 base_weights 推断
        idx:          可选，只统计某个节点子集（如 test_idx）

    返回：
        result: dict
    """
    base_weights = disc_weights['base_weights']
    dynamic_weights = disc_weights['dynamic_weights']

    if num_classes is None:
        cls_list = sorted(base_weights.keys())
    else:
        cls_list = list(range(num_classes))

    result = {}
    for c in cls_list:
        if c not in base_weights:
            continue
        w_base = base_weights[c]
        w_dyn = dynamic_weights[c]
        if idx is not None:
            w_dyn = w_dyn[idx]

        result[c] = {
            'base_avg_weight': w_base.mean().item(),
            'base_min_weight': w_base.min().item(),
            'base_max_weight': w_base.max().item(),
            'dyn_avg_weight': w_dyn.mean().item(),
            'dyn_min_weight': w_dyn.min().item(),
            'dyn_max_weight': w_dyn.max().item(),
            'high_disc_dims_base': (w_base > 0.5).sum().item(),
            'total_dims': len(w_base),
        }
    return result
