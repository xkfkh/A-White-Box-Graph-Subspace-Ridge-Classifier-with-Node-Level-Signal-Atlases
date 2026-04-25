"""
White-Box Pipeline -- 主体框架
================================

适用数据集: Cora / PubMed (共享相同的 5 层结构)

Layer 1: Tikhonov 图平滑           Z = (I + lam*L)^{-1} X
Layer 2: 类内 PCA/SVD 子空间        B_c, mu_c, Sigma_c
Layer 3: 判别方向加权 R_perp        w_k = 1 - max_{c'!=c} ||B_{c'}^T b_k||^2
Layer 4: (禁用 -- 已验证无增益且有泄漏风险)
Layer 5: argmin R_disc 预测

数学总览:
    R_disc,c(z) = ||z - mu_c||^2 - sum_k w_k * (b_k^T (z - mu_c))^2
    predict(z) = argmin_c R_disc,c(z)

    其中:
        z      : 经 Tikhonov 平滑后的节点表示 [D]
        mu_c   : 类 c 训练样本在平滑空间的均值 [D]
        B_c    : 类 c 的 PCA 子空间基 [D, d], d = sub_dim
        b_k    : B_c 的第 k 列 (第 k 个主方向)
        w_k    : 第 k 方向的判别权重, 越大表示该方向越"独特"于类 c

    几何意义:
        R_disc,c(z) 度量 z 到类 c 子空间的"加权正交残差":
        - ||z - mu_c||^2 是到类中心的总距离
        - sum w_k (...)^2 是沿判别方向的投影能量 (被减掉)
        - 结果 = "子空间外残差" = z 不能被类 c 解释的部分
        - argmin 选择残差最小的类 = z 最能被解释的子空间
"""

import os, time, torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data_loader import load_dataset, to_torch
from layer1_tikhonov import tikhonov_smooth
from layer2_subspace import build_class_subspaces
from layer3_discriminative import compute_discriminative_rperp


class WhiteBoxPipeline:
    """
    主体白盒分类框架.

    用法:
        pipe = WhiteBoxPipeline('cora', {'lam': 10.0, 'sub_dim': 12})
        result = pipe.run()
    """

    def __init__(self, dataset, config):
        self.dataset = dataset.lower()
        self.config = config
        # 中间结果 (run 之后可查看)
        self.X = None           # 原始特征
        self.Y = None           # 标签
        self.Z = None           # Layer 1 输出
        self.sub = None         # Layer 2 输出
        self.R_disc = None      # Layer 3 输出
        self.preds = None       # 最终预测
        self.result = None      # 精度结果字典

    # ── 精度计算工具 ───────────────────────────────────────
    @staticmethod
    def _acc(preds, Y, idx):
        idx_t = idx if isinstance(idx, torch.Tensor) else torch.tensor(idx, dtype=torch.long)
        return (preds[idx_t] == Y[idx_t]).float().mean().item()

    # ── Layer 0: 数据加载 ─────────────────────────────────
    def load_data(self):
        """加载数据集, 转为 torch tensor."""
        feat, labels, adj_norm, lap, tr, va, te, nc = load_dataset(self.dataset)
        self.X, self.Y, self.A, self.L, self.tr_t, self.va_t, self.te_t = \
            to_torch(feat, labels, adj_norm, lap, tr, va, te)
        self.train_list = tr.tolist()
        self.num_classes = nc
        self.n, self.D = self.X.shape

    # ── Layer 1: Tikhonov 图平滑 ─────────────────────────
    def layer1_smooth(self):
        """Z = (I + lam * L)^{-1} X"""
        lam = self.config['lam']
        self.Z = tikhonov_smooth(self.X, self.L, lam)

    # ── Layer 2: 类内 PCA 子空间构造 ─────────────────────
    def layer2_subspace(self):
        """对每个类 c, SVD 提取前 d 个主方向 B_c."""
        sd = self.config['sub_dim']
        self.sub = build_class_subspaces(
            self.Z, self.Y, self.train_list, self.num_classes, sub_dim=sd
        )

    # ── Layer 3: 判别方向加权 R_perp ─────────────────────
    def layer3_discriminative(self):
        """计算判别加权残差 R_disc."""
        self.R_disc, self.disc_weights = compute_discriminative_rperp(
            self.Z, self.sub, self.num_classes
        )

    # ── Layer 4: 修正 (主框架中禁用) ─────────────────────
    def layer4_correction(self):
        """主框架中不做修正, 直接透传 R_disc."""
        self.R_final = self.R_disc

    # ── Layer 5: argmin 预测 ─────────────────────────────
    def layer5_predict(self):
        """predict(z) = argmin_c R_final,c(z)"""
        self.preds = self.R_final.argmin(dim=1)

    # ── 评估 ──────────────────────────────────────────────
    def evaluate(self):
        """计算 train / val / test 精度."""
        train_acc = self._acc(self.preds, self.Y, self.tr_t)
        val_acc   = self._acc(self.preds, self.Y, self.va_t)
        test_acc  = self._acc(self.preds, self.Y, self.te_t)
        return train_acc, val_acc, test_acc

    # ── 完整运行 ──────────────────────────────────────────
    def run(self, verbose=True):
        """
        执行完整 pipeline, 返回结果字典.

        流程: load -> L1 -> L2 -> L3 -> L4 -> L5 -> evaluate
        """
        t0 = time.time()

        self.load_data()
        self.layer1_smooth()
        self.layer2_subspace()
        self.layer3_discriminative()
        self.layer4_correction()
        self.layer5_predict()

        train_acc, val_acc, test_acc = self.evaluate()
        elapsed = time.time() - t0

        self.result = {
            'dataset': self.dataset,
            'pipeline': 'WhiteBoxPipeline',
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
        print(f"  {self.dataset.upper()} -- WhiteBoxPipeline (main framework)")
        print(f"{'='*55}")
        print(f"  Layer 1: Tikhonov smooth    lam={cfg['lam']}")
        print(f"  Layer 2: Class PCA          sub_dim={cfg['sub_dim']}")
        print(f"  Layer 3: Disc-weighted Rperp")
        print(f"  Layer 4: (disabled)")
        print(f"  Layer 5: argmin R_disc")
        print(f"{'-'*55}")
        r = self.result
        print(f"  train={r['train']:.4f}  val={r['val']:.4f}  test={r['test']:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*55}")

