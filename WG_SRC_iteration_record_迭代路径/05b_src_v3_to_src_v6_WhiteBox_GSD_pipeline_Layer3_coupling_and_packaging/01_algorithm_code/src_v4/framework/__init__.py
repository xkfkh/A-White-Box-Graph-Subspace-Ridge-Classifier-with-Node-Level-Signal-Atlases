"""
White-Box Graph Node Classification Framework v3
=================================================

统一框架: 三个数据集均使用相同的 5 层 WhiteBoxPipeline
    Layer 1: Tikhonov 图平滑
    Layer 2: 类内 PCA/SVD 子空间
    Layer 3: 反应版判别方向加权 R_perp（V3 整体交联 + 动态反应）
    Layer 4: (禁用)
    Layer 5: argmin R_disc 预测

用法:
    from framework import create_pipeline
    pipe = create_pipeline('cora')
    result = pipe.run()
"""

import os, sys

# 确保 src_v3 在 sys.path 中, 使 layer*.py 和 data_loader.py 可直接 import
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .pipeline import WhiteBoxPipeline
from .configs import DEFAULT_CONFIGS, get_config


def create_pipeline(dataset, config=None):
    """
    工厂函数: 所有数据集统一使用 WhiteBoxPipeline v3.

    参数:
        dataset:  'cora' | 'citeseer' | 'pubmed'
        config:   可选, 覆盖默认超参数（可传入 eta_pos/eta_neg 等 Layer3 参数）

    返回:
        WhiteBoxPipeline 实例
    """
    ds = dataset.lower()
    cfg = config if config is not None else get_config(ds)
    return WhiteBoxPipeline(ds, cfg)
