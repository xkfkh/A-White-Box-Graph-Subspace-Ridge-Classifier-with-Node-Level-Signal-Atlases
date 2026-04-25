"""
Layer 4 V2: 基于判别 margin 的 Mahalanobis 自适应修正

核心创新：
    不对所有节点统一施加 Mahalanobis 修正（这会引入噪声），
    而是只对"不确定"节点做修正。

    "不确定"节点 = 排名第一和第二的类之间 R_disc 差距很小的节点。
    这些节点是分类最可能出错的地方，需要额外的子空间内部信息来判别。

数学定义：
    1. margin(z) = R_disc_{c''}(z) - R_disc_{c*}(z)
       c* = argmin_c R_disc_c(z)  (初始最优类)
       c'' = argmin_{c!=c*} R_disc_c(z)  (次优类)

    2. uncertain(z) = (margin(z) < threshold)
       threshold = quantile_p(margin)  (取最小 p% 的节点)

    3. 对 uncertain 节点做 Mahalanobis 修正：
       R_final_c(z) = R_disc_c(z) + beta * M_c(z)
       M_c(z) = (B_c^T(z-mu_c))^T (Sigma_c + eta*I)^{-1} (B_c^T(z-mu_c))

    4. beta 闭式确定：
       beta = std(R_disc[uncertain]) / (2 * std(M[uncertain]))
       用方差匹配 + 0.5 缩放（保守修正）

符号说明：
    margin     : 标量，类间距离差（越大越确定）
    threshold  : 标量，不确定阈值（由 percentile 确定）
    beta       : 标量，Mahalanobis 修正强度
    M_c(z)     : 标量，z 在类 c 子空间内的 Mahalanobis 距离

几何意义：
    当 margin 大时，z 明显属于某个类，不需要修正。
    当 margin 小时，z 在两个类之间摇摆：
    - R_perp 只看"子空间外距离"，可能不够区分
    - M_c 看"子空间内异常度"，能提供额外信息
    - 如果 z 在类 c 子空间内的分布更"正常"（M_c 小），
      则更可能属于类 c

    beta 的 0.5 缩放是保守策略：
    因为 Mahalanobis 在训练集上有过拟合风险，
    所以只给它一半的权重。

数值小例子：
    假设节点 z 在类 0 和类 1 之间：
    R_disc_0(z) = 0.005, R_disc_1(z) = 0.006
    margin = 0.001（很小，不确定）

    M_0(z) = 3.2, M_1(z) = 5.8
    beta = 0.001（方差匹配后）

    R_final_0 = 0.005 + 0.001 * 3.2 = 0.0082
    R_final_1 = 0.006 + 0.001 * 5.8 = 0.0118
    差距从 0.001 扩大到 0.0036，更有信心预测类 0
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_margin(R_disc):
    """
    计算每个节点的分类 margin。

    margin(z) = R_disc_{second_best}(z) - R_disc_{best}(z)

    参数：
        R_disc: [n, C]  判别加权残差

    返回：
        margin: [n]     每个节点的 margin
        best:   [n]     每个节点的最优类
        second: [n]     每个节点的次优类
    """
    R_sorted, R_indices = R_disc.sort(dim=1)
    margin = R_sorted[:, 1] - R_sorted[:, 0]
    best = R_indices[:, 0]
    second = R_indices[:, 1]
    return margin, best, second


def mahalanobis_correction(Z, subspace_result, num_classes, R_disc,
                           percentile=0.10, eta=1e-4):
    """
    对不确定节点做 Mahalanobis 自适应修正。

    参数：
        Z:               [n, D]  平滑后节点特征
        subspace_result:  Layer 2 输出
        num_classes:      int
        R_disc:          [n, C]  Layer 3 的判别加权残差
        percentile:       float  不确定节点比例 (0~1)
        eta:             float   Mahalanobis 正则化

    返回：
        R_final:         [n, C]  修正后的残差
        correction_info: dict    修正信息
    """
    n = Z.shape[0]
    margin, best_cls, second_cls = compute_margin(R_disc)

    # 确定不确定节点
    threshold = margin.quantile(percentile).item()
    uncertain = margin < threshold
    n_uncertain = uncertain.sum().item()

    if n_uncertain < 2:
        return R_disc.clone(), {'n_uncertain': 0, 'beta': 0.0}

    # 计算 Mahalanobis 距离
    M = torch.zeros(n, num_classes)
    for c in range(num_classes):
        if c not in subspace_result['B_dict']:
            M[:, c] = 1e10
            continue

        mu_c = subspace_result['mu_dict'][c]
        B_c = subspace_result['B_dict'][c]
        Sigma_c = subspace_result['Sigma_dict'][c]

        diff = Z - mu_c
        a_c = diff @ B_c
        sigma_diag = torch.diag(Sigma_c).clamp(min=0) + eta
        M[:, c] = ((a_c ** 2) / sigma_diag.unsqueeze(0)).sum(dim=1)

    # 闭式确定 beta
    R_std = R_disc[uncertain].std().item()
    M_std = M[uncertain].std().item()
    beta = R_std / (M_std + 1e-10) * 0.5  # 0.5 = 保守缩放

    # 修正
    R_final = R_disc.clone()
    R_final[uncertain] = R_disc[uncertain] + beta * M[uncertain]

    correction_info = {
        'n_uncertain': n_uncertain,
        'beta': beta,
        'threshold': threshold,
        'margin_mean': margin.mean().item(),
        'margin_std': margin.std().item(),
    }

    return R_final, correction_info

