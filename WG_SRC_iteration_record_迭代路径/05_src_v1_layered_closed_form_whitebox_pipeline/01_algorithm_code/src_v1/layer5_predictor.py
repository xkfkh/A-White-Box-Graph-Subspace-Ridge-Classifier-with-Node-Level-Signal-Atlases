"""
Layer 5: 白盒融合判别层 — Fisher 比加权线性可解释评分器

数学定义：
    对每个节点 z 和每类 c，最终得分：

    Score_c(z) = omega_I * I_tilde_c(z) - omega_R * R_tilde_perp_c(z) - omega_M * M_tilde_c(z)

    predict(z) = argmax_c Score_c(z)

符号说明：
    I_tilde_c(z)       : 归一化后的投影能量占比（来自 Layer 3）
    R_tilde_perp_c(z)  : 归一化后的子空间外残差（来自 Layer 4）
    M_tilde_c(z)       : 归一化后的 Mahalanobis 距离（来自 Layer 4）
    omega_I, omega_R, omega_M : 权重，由 Fisher 比闭式确定

归一化方式：
    对每个度量，在训练集上计算均值和标准差，然后 z-score 归一化：
    I_tilde_c(z) = (I_c(z) - mean_I) / (std_I + eps)
    R_tilde_perp_c(z) = (R_perp_c(z) - mean_R) / (std_R + eps)
    M_tilde_c(z) = (M_c(z) - mean_M) / (std_M + eps)

    这样三个度量在同一量纲下，权重才有意义。

权重确定（Fisher 比归一化）：
    omega_I = F_I / (F_I + F_R + F_M)
    omega_R = F_R / (F_I + F_R + F_M)
    omega_M = F_M / (F_I + F_R + F_M)

    其中 F_I, F_R, F_M 分别是三个度量在训练集上的 Fisher 判别比。
    Fisher 比越大，说明该度量的判别力越强，应该给更大权重。

    这是闭式确定的，不需要网格搜索。

几何意义：
    Score_c(z) 综合了三个互补的白盒信号：
    1. I_c(z)：z 的偏差方向与类 c 子空间的匹配度（越大越好）
    2. R_perp_c(z)：z 到类 c 子空间的距离（越小越好，所以取负）
    3. M_c(z)：z 在类 c 子空间内的异常度（越小越好，所以取负）

    三者互补：
    - I_c 看方向匹配
    - R_perp 看距离远近
    - M_c 看分布符合度

数值小例子：
    假设 F_I=2.0, F_R=1.0, F_M=0.5
    omega_I = 2.0/3.5 = 0.571
    omega_R = 1.0/3.5 = 0.286
    omega_M = 0.5/3.5 = 0.143

    对节点 z, 类 c=0:
    I_tilde = 0.8, R_tilde = -0.3, M_tilde = 0.1
    Score_0 = 0.571*0.8 - 0.286*(-0.3) - 0.143*0.1
            = 0.457 + 0.086 - 0.014 = 0.529
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_normalization_stats(values, train_idx):
    """
    在训练集上计算归一化统计量（均值和标准差）。

    参数：
        values:    [n, C]  原始度量值
        train_idx: list    训练节点索引

    返回：
        mean_val:  float   训练集上的全局均值
        std_val:   float   训练集上的全局标准差
    """
    train_t = torch.tensor(train_idx, dtype=torch.long)
    v_train = values[train_t]  # [n_train, C]
    mean_val = v_train.mean().item()
    std_val = v_train.std().item()
    return mean_val, std_val


def normalize_scores(values, mean_val, std_val, eps=1e-8):
    """
    Z-score 归一化。

    参数：
        values:   [n, C]
        mean_val: float
        std_val:  float

    返回：
        normalized: [n, C]
    """
    return (values - mean_val) / (std_val + eps)


def compute_fisher_weights(fisher_I, fisher_R, fisher_M, eps=1e-10):
    """
    用 Fisher 比闭式确定三个度量的权重。

    omega_k = F_k / (F_I + F_R + F_M)

    参数：
        fisher_I: float  投影能量占比的 Fisher 比
        fisher_R: float  子空间外残差的 Fisher 比
        fisher_M: float  Mahalanobis 距离的 Fisher 比

    返回：
        omega_I, omega_R, omega_M: float
    """
    # 确保 Fisher 比非负
    f_I = max(fisher_I, 0.0)
    f_R = max(fisher_R, 0.0)
    f_M = max(fisher_M, 0.0)

    total = f_I + f_R + f_M + eps
    omega_I = f_I / total
    omega_R = f_R / total
    omega_M = f_M / total

    return omega_I, omega_R, omega_M


def predict(energy_ratio, R_perp, M_dist, train_idx,
            fisher_I, fisher_R, fisher_M):
    """
    Layer 5 完整预测流程。

    步骤：
    1. 在训练集上计算归一化统计量
    2. Z-score 归一化三个度量
    3. 用 Fisher 比确定权重
    4. 计算最终得分
    5. argmax 预测

    参数：
        energy_ratio: [n, C]  Layer 3 投影能量占比
        R_perp:       [n, C]  Layer 4 子空间外残差
        M_dist:       [n, C]  Layer 4 Mahalanobis 距离
        train_idx:    list    训练节点索引
        fisher_I:     float   I 的 Fisher 比
        fisher_R:     float   R_perp 的 Fisher 比
        fisher_M:     float   M_dist 的 Fisher 比

    返回：
        predictions:  [n]     预测标签
        scores:       [n, C]  最终得分矩阵
        details:      dict    详细分解信息
    """
    # Step 1: 归一化统计量
    mean_I, std_I = compute_normalization_stats(energy_ratio, train_idx)
    mean_R, std_R = compute_normalization_stats(R_perp, train_idx)
    mean_M, std_M = compute_normalization_stats(M_dist, train_idx)

    # Step 2: Z-score 归一化
    I_norm = normalize_scores(energy_ratio, mean_I, std_I)
    R_norm = normalize_scores(R_perp, mean_R, std_R)
    M_norm = normalize_scores(M_dist, mean_M, std_M)

    # Step 3: Fisher 比权重
    omega_I, omega_R, omega_M = compute_fisher_weights(fisher_I, fisher_R, fisher_M)

    # Step 4: 最终得分
    # I 越大越好（+），R_perp 越小越好（-），M 越小越好（-）
    scores = omega_I * I_norm - omega_R * R_norm - omega_M * M_norm

    # Step 5: 预测
    predictions = scores.argmax(dim=1)

    details = {
        'omega_I': omega_I,
        'omega_R': omega_R,
        'omega_M': omega_M,
        'I_norm': I_norm,
        'R_norm': R_norm,
        'M_norm': M_norm,
        'mean_I': mean_I, 'std_I': std_I,
        'mean_R': mean_R, 'std_R': std_R,
        'mean_M': mean_M, 'std_M': std_M,
        'fisher_I': fisher_I,
        'fisher_R': fisher_R,
        'fisher_M': fisher_M,
    }

    return predictions, scores, details


def evaluate(predictions, Y, train_idx, val_idx, test_idx):
    """
    评估预测精度。

    返回 dict: train_acc, val_acc, test_acc
    """
    Y_t = Y if isinstance(Y, torch.Tensor) else torch.tensor(Y)
    pred_t = predictions if isinstance(predictions, torch.Tensor) else torch.tensor(predictions)

    def acc(idx):
        idx_t = torch.tensor(idx, dtype=torch.long) if not isinstance(idx, torch.Tensor) else idx
        return (pred_t[idx_t] == Y_t[idx_t]).float().mean().item()

    return {
        'train_acc': acc(train_idx),
        'val_acc': acc(val_idx),
        'test_acc': acc(test_idx),
    }

