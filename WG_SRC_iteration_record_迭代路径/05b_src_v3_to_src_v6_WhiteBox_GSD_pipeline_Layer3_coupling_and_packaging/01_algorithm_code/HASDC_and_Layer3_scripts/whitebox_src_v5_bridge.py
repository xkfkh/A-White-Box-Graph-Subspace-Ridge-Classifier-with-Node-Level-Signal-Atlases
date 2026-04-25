# -*- coding: utf-8 -*-
"""
whitebox_src_v5_bridge.py

只负责把真实 src_v5 放到 sys.path，并调用 src_v5 中的 Layer1/Layer2/Layer3 函数。
本文件不会自动修改、patch 或覆盖你的算法源码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def setup_src_v5(src: Optional[str]) -> Path:
    if not src:
        src = os.environ.get("WHITEBOX_SRC_V5") or os.environ.get("WHITEBOX_SRC_DIR")
    if not src:
        raise FileNotFoundError("请传 --src 或设置环境变量 WHITEBOX_SRC_V5 指向 src_v5。")
    p = Path(src).expanduser().resolve()
    required = ["layer1_tikhonov.py", "layer2_subspace.py", "layer3_discriminative.py"]
    missing = [name for name in required if not (p / name).exists()]
    if missing:
        raise FileNotFoundError(f"src_v5 不完整: {p}; 缺少: {missing}")

    # 防止导入到别的同名模块。
    for name in ["layer1_tikhonov", "layer2_subspace", "layer3_discriminative", "layer3_hasdc", "data_loader", "framework"]:
        sys.modules.pop(name, None)
    sys.path = [s for s in sys.path if str(Path(s or '.').resolve()) != str(p)]
    sys.path.insert(0, str(p))
    os.environ["WHITEBOX_SRC_V5"] = str(p)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print(f"[src_v5] {p}")
    return p


def build_subspaces_src(Z_cpu: torch.Tensor, y_cpu: torch.Tensor, train_idx_cpu, num_classes: int, sub_dim: int) -> Dict[str, Any]:
    from layer2_subspace import build_class_subspaces
    return build_class_subspaces(Z_cpu, y_cpu, list(train_idx_cpu), num_classes, sub_dim=sub_dim)


def score_plain_residual_src(Z_cpu: torch.Tensor, sub: Dict[str, Any], num_classes: int) -> torch.Tensor:
    from layer2_subspace import classify_by_residual
    _pred, scores = classify_by_residual(Z_cpu, sub, num_classes)
    # classify_by_residual 返回的是 neg_resid，越大越好；这里统一成 R，越小越好。
    return -scores


def score_layer3_src(
    Z_cpu: torch.Tensor,
    sub: Dict[str, Any],
    num_classes: int,
    eta: float,
    activation_mode: str = "mahalanobis",
    activation_rho: float = 1e-6,
    max_activation_scale=None,
    aggregation_mode: str = "suppression",
) -> torch.Tensor:
    from layer3_discriminative import compute_discriminative_rperp
    R, _disc = compute_discriminative_rperp(
        Z_cpu, sub, num_classes,
        eta_react=eta,
        activation_mode=activation_mode,
        activation_rho=activation_rho,
        max_activation_scale=max_activation_scale,
        aggregation_mode=aggregation_mode,
    )
    return R


def tikhonov_smooth_src(X: torch.Tensor, L_dense: torch.Tensor, lam: float) -> torch.Tensor:
    from layer1_tikhonov import tikhonov_smooth
    return tikhonov_smooth(X, L_dense, lam)


def build_subspaces_raw_src(X_cpu: torch.Tensor, y_cpu: torch.Tensor, train_idx_cpu, num_classes: int, sub_dim: int) -> Dict[str, Any]:
    """Build class subspaces from raw features X (no smoothing)."""
    from layer2_subspace import build_class_subspaces
    return build_class_subspaces(X_cpu, y_cpu, list(train_idx_cpu), num_classes, sub_dim=sub_dim)


def score_hasdc_src(
    Z_cpu: torch.Tensor,
    X_cpu: torch.Tensor,
    Y_cpu: torch.Tensor,
    train_idx,
    val_idx,
    sub_s: Dict[str, Any],
    sub_r: Dict[str, Any],
    num_classes: int,
    scoring_mode: str = "plain_residual",
    eta: float = 0.10,
    tau: float = 5.0,
    activation_mode: str = "mahalanobis",
    activation_rho: float = 1e-6,
    max_activation_scale=None,
    aggregation_mode: str = "suppression",
    fusion_strategy: str = "channel_select",
    merged_dim: int = None,
) -> tuple:
    """Call HA-SDC dual-channel scoring from layer3_hasdc."""
    from layer3_hasdc import compute_hasdc_scores
    return compute_hasdc_scores(
        Z_cpu, X_cpu, Y_cpu,
        train_idx, val_idx,
        sub_s, sub_r, num_classes,
        scoring_mode=scoring_mode,
        eta=eta, tau=tau,
        activation_mode=activation_mode,
        activation_rho=activation_rho,
        max_activation_scale=max_activation_scale,
        aggregation_mode=aggregation_mode,
        fusion_strategy=fusion_strategy,
        merged_dim=merged_dim,
    )


def score_freq_decomposed_src(
    Z_cpu: torch.Tensor,
    X_cpu: torch.Tensor,
    Y_cpu: torch.Tensor,
    train_idx,
    val_idx,
    num_classes: int,
    sub_dim: int = 12,
    scoring_mode: str = "plain_residual",
    eta: float = 0.10,
    activation_mode: str = "mahalanobis",
    activation_rho: float = 1e-6,
) -> tuple:
    """Call frequency-decomposed scoring from layer3_hasdc."""
    from layer3_hasdc import compute_freq_decomposed_scores
    return compute_freq_decomposed_scores(
        Z_cpu, X_cpu, Y_cpu,
        train_idx, val_idx, num_classes,
        sub_dim=sub_dim, scoring_mode=scoring_mode, eta=eta,
        activation_mode=activation_mode, activation_rho=activation_rho,
    )
