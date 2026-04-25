"""
White-Box Graph Node Classification Framework
==============================================

统一框架: 三个数据集均使用相同的 5 层 WhiteBoxPipeline
    Layer 1: Tikhonov 图平滑
    Layer 2: 类内 PCA/SVD 子空间
    Layer 3: 判别方向加权 R_perp
    Layer 4: (禁用)
    Layer 5: argmin R_disc 预测

用法:
    from framework import create_pipeline
    pipe = create_pipeline('cora')
    result = pipe.run()

历史备注:
    CiteSeer 曾使用 CiteSeerPlugin (Global PCA + Fisher-LDA),
    但修复 data_loader test_idx 映射 bug 后, 主框架 + lam=7
    在 CiteSeer 上的表现 (test=0.724) 远超 Fisher 插件 (test=0.654).
    因此统一使用 WhiteBoxPipeline.
"""

import os, sys

# 确保 src_v1 在 sys.path 中, 使 layer*.py 和 data_loader.py 可直接 import
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .pipeline import WhiteBoxPipeline
from .citeseer_plugin import CiteSeerPlugin  # 保留, 供消融实验使用
from .configs import DEFAULT_CONFIGS, get_config


def create_pipeline(dataset, config=None):
    """
    工厂函数: 所有数据集统一使用 WhiteBoxPipeline.

    参数:
        dataset:  'cora' | 'citeseer' | 'pubmed'
        config:   可选, 覆盖默认超参数

    返回:
        WhiteBoxPipeline 实例
    """
    ds = dataset.lower()
    cfg = config if config is not None else get_config(ds)
    return WhiteBoxPipeline(ds, cfg)

