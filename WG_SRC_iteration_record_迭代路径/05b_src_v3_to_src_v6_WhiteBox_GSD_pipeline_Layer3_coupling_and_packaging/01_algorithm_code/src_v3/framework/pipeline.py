"""
White-Box Pipeline v3 -- Main Framework
====================================================

Supported datasets: Cora / CiteSeer / PubMed

Layer 1: Tikhonov graph smoothing         Z = (I + lam * L)^{-1} X
Layer 2: Class-wise PCA/SVD subspaces     B_c, mu_c, Sigma_c
Layer 3: Reactive discriminative R_perp   V3 algorithm
Layer 4: (disabled -- no stable gain observed in the main pipeline)
Layer 5: Prediction by argmin R_disc

Evaluation protocol:
- Training labels are used only on the training split.
- The validation and test splits are not used for class-subspace construction.
- Train / validation / test metrics are computed only during evaluation.

Layer 3 V3 summary:
    K_{c,c'} = B_c^T B_{c'}                           # cross-subspace projection matrix
    w_base,k^(c) = 1 / (1 + sum_{c'!=c} ||K_{c,c'}[k,:]||^2)
    u_c(z) = B_c^T (z - mu_c)                         # coordinates in class-c subspace
    r_+,k^(c)(z) = sum_{c'!=c} sum_j M^(+)[k,j] * |u_{c',j}(z)|
    r_-,k^(c)(z) = sum_{c'!=c} sum_j M^(-)[k,j] * |u_{c',j}(z)|
    w_tilde,k^(c)(z) = clip(w_base + eta_pos*r_+ - eta_neg*r_-, wmin, wmax)

    R_disc,c(z) = ||z - mu_c||^2 - sum_k w_tilde,k^(c)(z) * u_{c,k}(z)^2
    predict(z)  = argmin_c R_disc,c(z)

Geometric interpretation:
    - w_base  : static discriminative weight based on global cross-class coupling
    - M^(+)   : orthogonal cross-class component; activation from other classes
                enhances the corresponding direction of the current class
    - M^(-)   : overlapping cross-class component; overlap activation suppresses
                the corresponding direction of the current class
    - The reactive mechanism makes the direction weights sample-adaptive
"""

import os, time, torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data_loader import load_dataset, to_torch
from layer1_tikhonov import tikhonov_smooth
from layer2_subspace import build_class_subspaces
from layer3_discriminative import compute_discriminative_rperp


class WhiteBoxPipeline:
    """
     Main white-box classification pipeline v3.

    Protocol note:
    - Training labels are used only for class-subspace construction on the training split.
    - The validation and test splits are not used for white-box representation construction
      or discriminative modeling.
    - Train / validation / test metrics are computed only in the final evaluation step.

    Usage:
        pipe = WhiteBoxPipeline('cora', config)
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
        self.disc_weights = None  # Layer 3 权重字典（含 base/dynamic/K）
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

    # ── Layer 3: 反应版判别方向加权 R_perp ───────────────
    def layer3_discriminative(self):
        """
        计算反应版判别加权残差 R_disc（Layer 3 V3）。

        从 config 中读取 eta_pos / eta_neg / weight_min / weight_max，
        若未配置则使用算法默认值。
        """
        eta_pos    = self.config.get('eta_pos',    0.10)
        eta_neg    = self.config.get('eta_neg',    0.30)
        weight_min = self.config.get('weight_min', 0.01)
        weight_max = self.config.get('weight_max', 10.0)

        self.R_disc, self.disc_weights = compute_discriminative_rperp(
            self.Z, self.sub, self.num_classes,
            eta_pos=eta_pos,
            eta_neg=eta_neg,
            weight_min=weight_min,
            weight_max=weight_max,
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

        流程: load -> L1 -> L2 -> L3(V3) -> L4 -> L5 -> evaluate
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
            'dataset':  self.dataset,
            'pipeline': 'WhiteBoxPipeline-v3',
            'config':   self.config,
            'train':    train_acc,
            'val':      val_acc,
            'test':     test_acc,
            'time':     round(elapsed, 2),
        }

        if verbose:
            self._print_report(elapsed)

        return self.result

    def _print_report(self, elapsed):
        """打印运行报告."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"  {self.dataset.upper()} -- WhiteBoxPipeline v3 (Layer3 反应版)")
        print(f"{'='*60}")
        print(f"  Layer 1: Tikhonov smooth    lam={cfg['lam']}")
        print(f"  Layer 2: Class PCA          sub_dim={cfg['sub_dim']}")
        print(f"  Layer 3: ReactiveDisc Rperp eta_pos={cfg.get('eta_pos', 0.10)}  "
              f"eta_neg={cfg.get('eta_neg', 0.30)}")
        print(f"  Layer 4: (disabled)")
        print(f"  Layer 5: argmin R_disc")
        print(f"{'-'*60}")
        r = self.result
        print(f"  train={r['train']:.4f}  val={r['val']:.4f}  test={r['test']:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*60}")
