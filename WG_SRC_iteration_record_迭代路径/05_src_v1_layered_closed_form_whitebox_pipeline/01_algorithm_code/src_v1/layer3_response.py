"""
Layer 3: 子空间响应量化层 — 投影能量占比

数学定义：
    对每个节点 z (平滑后特征) 和每类 c，计算投影能量占比：

    I_c(z) = ||B_c^T (z - mu_c)||^2 / (||z - mu_c||^2 + eps)

符号说明：
    z      : [D]    平滑后的节点特征向量
    mu_c   : [D]    类 c 训练样本的均值
    B_c    : [D, d] 类 c 的子空间正交基（右奇异向量）
    eps    : 小正数，防止除零

几何意义：
    I_c(z) 衡量 z 偏离类 c 均值的方向中，有多少比例落在类 c 的子空间内。
    - I_c(z) 接近 1：z 的偏差几乎完全在子空间内，说明 z 与类 c 的变化模式一致
    - I_c(z) 接近 0：z 的偏差主要在子空间外，说明 z 不符合类 c 的结构

    这是一个归一化的量，不受 ||z - mu_c|| 的绝对大小影响，
    只看方向上的匹配程度。

数值小例子 (D=4, d=2):
    mu_c = [1, 0, 0, 0]
    B_c = [[1,0], [0,1], [0,0], [0,0]]  (前两维构成子空间)
    z = [2, 1, 0, 0]  => z - mu_c = [1, 1, 0, 0]
    ||B_c^T (z-mu_c)||^2 = 1^2 + 1^2 = 2
    ||z - mu_c||^2 = 2
    I_c(z) = 2/2 = 1.0  (完全在子空间内)

    z = [1, 0, 1, 1]  => z - mu_c = [0, 0, 1, 1]
    ||B_c^T (z-mu_c)||^2 = 0
    ||z - mu_c||^2 = 2
    I_c(z) = 0/2 = 0.0  (完全在子空间外)

代码实现：
    对所有节点批量计算，返回 [n, C] 的投影能量占比矩阵。
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_projection_energy_ratio(Z, subspace_result, num_classes, eps=1e-8):
    """
    计算每个节点对每类的投影能量占比 I_c(z)。

    参数：
        Z:               [n, D]  平滑后节点特征
        subspace_result:  Layer 2 输出的 dict，包含 mu_dict, B_dict
        num_classes:      int    类别数
        eps:             float   防除零

    返回：
        energy_ratio:    [n, C]  投影能量占比矩阵
                         energy_ratio[i, c] = I_c(z_i)
    """
    n = Z.shape[0]
    energy_ratio = torch.zeros(n, num_classes)

    for c in range(num_classes):
        if c not in subspace_result['B_dict']:
            continue

        mu_c = subspace_result['mu_dict'][c]   # [D]
        B_c = subspace_result['B_dict'][c]     # [D, d]

        diff = Z - mu_c                        # [n, D]
        total_sq = (diff ** 2).sum(dim=1)      # [n]

        proj = diff @ B_c                      # [n, d]
        proj_sq = (proj ** 2).sum(dim=1)       # [n]

        energy_ratio[:, c] = proj_sq / (total_sq + eps)

    return energy_ratio


def analyze_response(energy_ratio, Y, train_idx, num_classes):
    """
    分析投影能量占比的统计特性。

    返回 dict:
        class_self_ratio:  每类训练样本在自己子空间的平均 I_c
        class_other_ratio: 每类训练样本在其他子空间的平均 I_c
        fisher_ratio:      每个特征维度的 Fisher 比（用于 Layer 5 权重确定）
    """
    train_t = torch.tensor(train_idx, dtype=torch.long)
    E_tr = energy_ratio[train_t]  # [n_train, C]
    Y_tr = Y[train_t]

    class_self_ratio = {}
    class_other_ratio = {}

    for c in range(num_classes):
        mask_c = (Y_tr == c)
        if mask_c.sum() < 1:
            class_self_ratio[c] = 0.0
            class_other_ratio[c] = 0.0
            continue

        # 自己子空间的平均能量占比
        class_self_ratio[c] = E_tr[mask_c, c].mean().item()

        # 其他子空间的平均能量占比
        other_cols = [j for j in range(num_classes) if j != c]
        if other_cols:
            class_other_ratio[c] = E_tr[mask_c][:, other_cols].mean().item()
        else:
            class_other_ratio[c] = 0.0

    # Fisher 比: 对 energy_ratio 的每个类维度，计算类间/类内方差比
    # 用于 Layer 5 确定 omega_I 的权重
    fisher_I = _compute_fisher_ratio(E_tr, Y_tr, num_classes)

    return {
        'class_self_ratio': class_self_ratio,
        'class_other_ratio': class_other_ratio,
        'fisher_ratio_I': fisher_I,
    }


def _compute_fisher_ratio(scores, Y, num_classes):
    """
    计算 Fisher 判别比。

    Fisher ratio = between_class_variance / within_class_variance

    对 scores [n, C] 的每一列（每类得分），计算：
    - 类间方差：各类均值的方差
    - 类内方差：各类内部方差的加权平均

    返回标量 Fisher 比（对所有列取平均）。
    """
    C = scores.shape[1]
    total_fisher = 0.0
    count = 0

    for col in range(C):
        s = scores[:, col]  # [n]
        class_means = []
        class_vars = []
        class_sizes = []

        for c in range(num_classes):
            mask = (Y == c)
            nc = mask.sum().item()
            if nc < 2:
                continue
            sc = s[mask]
            class_means.append(sc.mean().item())
            class_vars.append(sc.var().item())
            class_sizes.append(nc)

        if len(class_means) < 2:
            continue

        grand_mean = sum(m * n for m, n in zip(class_means, class_sizes)) / sum(class_sizes)
        between_var = sum(n * (m - grand_mean) ** 2 for m, n in zip(class_means, class_sizes)) / sum(class_sizes)
        within_var = sum(n * v for n, v in zip(class_sizes, class_vars)) / sum(class_sizes)

        if within_var > 1e-12:
            total_fisher += between_var / within_var
            count += 1

    return total_fisher / max(count, 1)

