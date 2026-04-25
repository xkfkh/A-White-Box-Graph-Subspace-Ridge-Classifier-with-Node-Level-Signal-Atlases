"""
Layer 1: Tikhonov 图平滑

公式：
    Z = (I + lambda * L)^{-1} X

符号说明：
    X      : [n, D]  原始节点特征矩阵，n 个节点，每个节点 D 维
    L      : [n, n]  归一化图拉普拉斯，L = I - D^{-1/2} A D^{-1/2}
    lambda : 正标量，平滑强度（越大越平滑，越接近邻居均值）
    Z      : [n, D]  平滑后的节点特征矩阵

几何/信息论意义：
    Tikhonov 平滑等价于求解如下优化问题：
        min_Z  ||Z - X||_F^2 + lambda * tr(Z^T L Z)
    第一项：保持特征与原始特征接近（保真项）
    第二项：tr(Z^T L Z) = sum_{(i,j) in E} ||z_i - z_j||^2，
            惩罚相邻节点特征差异（平滑项）
    lambda 控制两者的权衡：
        lambda -> 0：Z ≈ X（不平滑）
        lambda -> inf：Z 趋向图上的常数函数（过度平滑）

数值小例子（3节点路径图）：
    X = [[1,0],[0,1],[1,1]]
    A = [[1,1,0],[1,1,1],[0,1,1]]（含自环）
    L = I - D^{-1/2} A D^{-1/2}
    lambda = 1.0
    求解 (I + L) Z = X，得到 Z 是 X 的图平滑版本

实现方式：
    用 torch.linalg.solve((I + lambda*L), X) 求解线性方程组，
    比直接求逆 (I + lambda*L)^{-1} @ X 更数值稳定。
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def tikhonov_smooth(X: torch.Tensor, L: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Tikhonov 图平滑：Z = (I + lam * L)^{-1} X

    参数：
        X   : [n, D] 节点特征矩阵（torch.Tensor, float32）
        L   : [n, n] 归一化拉普拉斯矩阵（torch.Tensor, float32）
        lam : 正则化参数（float），控制平滑程度

    返回：
        Z   : [n, D] 平滑后的节点特征矩阵（torch.Tensor, float32）

    实现细节：
        求解线性方程组 A_sys @ Z = X，其中 A_sys = I + lam * L
        A_sys 是对称正定矩阵（因为 L 半正定，lam > 0），
        torch.linalg.solve 使用 LU 分解，数值稳定。
    """
    assert lam >= 0, f"lambda 必须非负，当前值: {lam}"

    # lam=0 时 Z = X（不平滑），直接返回
    if lam == 0:
        return X.clone()

    n = X.shape[0]
    device = X.device
    dtype = X.dtype

    # 构造系数矩阵 A_sys = I + lam * L，形状 [n, n]
    I = torch.eye(n, device=device, dtype=dtype)
    A_sys = I + lam * L

    # 求解 A_sys @ Z = X，等价于 Z = A_sys^{-1} X
    # torch.linalg.solve(A, B) 求解 A @ X = B
    Z = torch.linalg.solve(A_sys, X)

    return Z


def tikhonov_smooth_batch(X: torch.Tensor, L: torch.Tensor, lam_list: list) -> dict:
    """
    对多个 lambda 值批量做 Tikhonov 平滑，复用 LU 分解。

    参数：
        X        : [n, D] 节点特征
        L        : [n, n] 归一化拉普拉斯
        lam_list : list of float，多个 lambda 值

    返回：
        dict: {lam: Z_lam}，每个 lambda 对应平滑后的特征
    """
    results = {}
    for lam in lam_list:
        results[lam] = tikhonov_smooth(X, L, lam)
    return results

