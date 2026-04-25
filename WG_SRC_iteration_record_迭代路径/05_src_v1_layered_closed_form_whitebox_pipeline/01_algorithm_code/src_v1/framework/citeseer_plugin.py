"""
CiteSeer Plugin -- 插件式替换
===============================

继承 WhiteBoxPipeline, 仅替换以下层:
    Layer 1.5 (新增): Global PCA 降维    Y = (Z - mu_g) @ V[:,:r]
    Layer 2  (替换):  Fisher-LDA 判别子空间
    Layer 3  (替换):  Fisher 最近质心分类 (无需 R_perp)
    Layer 5  (替换):  argmax score (非 argmin residual)

不变的层:
    Layer 0:  数据加载 (继承)
    Layer 1:  Tikhonov 平滑 (继承, 仅 lam 不同)
    Layer 4:  禁用 (继承)

设计动机:
    CiteSeer 的难点: 120 个训练样本, 3703 维特征, 6 类
    - 样本远少于维度 (120 << 3703), 类内 PCA 在每类仅 20 样本下不稳定
    - 需要先 Global PCA 降到 r=25 维, 再用 Fisher-LDA 找判别方向

    与主框架的关键差异:
    ┌──────────┬──────────────────────────┬──────────────────────────────┐
    |   Layer  |   主框架 (Cora/PubMed)   |   CiteSeer 插件              |
    ├──────────┼──────────────────────────┼──────────────────────────────┤
    |   1      | Tikhonov (lam=10)        | Tikhonov (lam=0.1, 弱平滑)   |
    |   1.5    | (无)                     | Global PCA (r=25)            |
    |   2      | 类内 PCA/SVD             | Fisher-LDA                   |
    |   3      | disc R_perp (argmin)     | 最近质心 score (argmax)       |
    |   4      | 禁用                     | 禁用                          |
    |   5      | argmin R_disc            | argmax score                 |
    └──────────┴──────────────────────────┴──────────────────────────────┘

数学定义:
    Layer 1.5 - Global PCA:
        mu_g = mean(Z)
        Z_tilde = Z - mu_g
        U, S, V^T = SVD(Z_tilde)
        P = V[:, :r]            投影矩阵 [D, r]
        Y = Z_tilde @ P         降维表示 [n, r]

    Layer 2 - Fisher-LDA:
        S_W = sum_c sum_{x in c} (x-mu_c)(x-mu_c)^T   类内散度
        S_B = sum_c n_c (mu_c-mu)(mu_c-mu)^T            类间散度
        S_B w = lambda S_W w                             广义特征值
        W = top-d eigenvectors                           判别方向 [r, d]

    Layer 3 - 最近质心:
        score_c(z) = -||W^T(z - mu_g)@P - mu_c^{proj}||^2
        predict(z) = argmax_c score_c(z) = argmin_c ||...||^2

    几何意义:
        Fisher-LDA 直接最大化类间/类内散度比,
        在 PCA 降维后的低维空间中找到最优判别方向.
        最近质心在 Fisher 空间中寻找最近的类中心.
"""

import os, time, torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .pipeline import WhiteBoxPipeline
from layer2_fisher import build_fisher_subspace, classify_fisher_centroid


class CiteSeerPlugin(WhiteBoxPipeline):
    """
    CiteSeer 插件: 继承主框架, 替换 Layer 1.5/2/3/5.

    用法:
        pipe = CiteSeerPlugin('citeseer', {
            'lam': 0.1, 'r_dim': 25, 'sub_dim': 4, 'reg': 0.02
        })
        result = pipe.run()
    """

    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        # CiteSeer 专有中间结果
        self.Y_pca = None       # Layer 1.5: PCA 降维后表示
        self.P = None           # Layer 1.5: 投影矩阵
        self.mu_g = None        # Layer 1.5: 全局均值
        self.fisher = None      # Layer 2: Fisher-LDA 结果

    # ── Layer 1.5 (新增): Global PCA ─────────────────────
    def layer15_global_pca(self):
        """
        Y = (Z - mu_g) @ V[:, :r]

        降维: 从 D=3703 降到 r=25, 去除噪声维度,
        使后续 Fisher-LDA 在合理维度下工作.
        """
        r = self.config['r_dim']
        self.mu_g = self.Z.mean(dim=0)
        Z_tilde = self.Z - self.mu_g
        _, S, Vt = torch.linalg.svd(Z_tilde, full_matrices=False)
        self.P = Vt[:r].t()              # [D, r]
        self.Y_pca = Z_tilde @ self.P    # [n, r]

    # ── Layer 2 (替换): Fisher-LDA ───────────────────────
    def layer2_subspace(self):
        """Fisher-LDA 判别子空间, 替代类内 PCA."""
        sd  = self.config['sub_dim']
        reg = self.config['reg']
        self.fisher = build_fisher_subspace(
            self.Y_pca, self.Y, self.train_list,
            self.num_classes, sub_dim=sd, reg=reg
        )

    # ── Layer 3 (替换): Fisher 最近质心 ──────────────────
    def layer3_discriminative(self):
        """
        Fisher 最近质心分类器, 替代 disc R_perp.

        score_c(z) = -||W^T z_proj - mu_c^{proj}||^2
        """
        self.preds, self.scores = classify_fisher_centroid(
            self.Y_pca, self.fisher, self.num_classes
        )
        # scores 就是最终得分, 不需要 R_disc
        self.R_disc = -self.scores  # 兼容: 负 score = 残差

    # ── Layer 4: 禁用 (继承) ──────────────────────────────
    def layer4_correction(self):
        """禁用, 透传."""
        self.R_final = self.R_disc

    # ── Layer 5 (替换): argmax score ─────────────────────
    def layer5_predict(self):
        """
        predict = argmax score = argmin(-score) = argmin R_final

        注意: preds 已在 layer3 中计算, 这里仅保持一致性.
        实际上 layer3 中 classify_fisher_centroid 已直接给出 preds.
        """
        # preds 已在 layer3 设置, 此处用 R_final 验证一致性
        self.preds = self.R_final.argmin(dim=1)

    # ── 完整运行 (覆写: 插入 Layer 1.5) ──────────────────
    def run(self, verbose=True):
        """
        执行完整 pipeline, 含 Layer 1.5.

        流程: load -> L1 -> L1.5 -> L2 -> L3 -> L4 -> L5 -> evaluate
        """
        t0 = time.time()

        self.load_data()           # Layer 0: 数据加载 (继承)
        self.layer1_smooth()       # Layer 1: Tikhonov (继承, lam=0.1)
        self.layer15_global_pca()  # Layer 1.5: Global PCA (新增)
        self.layer2_subspace()     # Layer 2: Fisher-LDA (替换)
        self.layer3_discriminative()  # Layer 3: 最近质心 (替换)
        self.layer4_correction()   # Layer 4: 禁用 (继承)
        self.layer5_predict()      # Layer 5: argmax score (替换)

        train_acc, val_acc, test_acc = self.evaluate()
        elapsed = time.time() - t0

        self.result = {
            'dataset': self.dataset,
            'pipeline': 'CiteSeerPlugin',
            'config': self.config,
            'train': train_acc,
            'val': val_acc,
            'test': test_acc,
            'time': round(elapsed, 2),
        }

        if verbose:
            self._print_report(elapsed)

        return self.result

    def _print_report(self, elapsed):
        """打印运行报告."""
        cfg = self.config
        print(f"\n{'='*55}")
        print(f"  {self.dataset.upper()} -- CiteSeerPlugin")
        print(f"{'='*55}")
        print(f"  Layer 1:   Tikhonov smooth    lam={cfg['lam']}")
        print(f"  Layer 1.5: Global PCA         r={cfg['r_dim']}")
        print(f"  Layer 2:   Fisher-LDA         d={cfg['sub_dim']}  reg={cfg['reg']}")
        print(f"  Layer 3:   Nearest centroid")
        print(f"  Layer 4:   (disabled)")
        print(f"  Layer 5:   argmax score")
        print(f"{'-'*55}")
        r = self.result
        print(f"  train={r['train']:.4f}  val={r['val']:.4f}  test={r['test']:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*55}")

