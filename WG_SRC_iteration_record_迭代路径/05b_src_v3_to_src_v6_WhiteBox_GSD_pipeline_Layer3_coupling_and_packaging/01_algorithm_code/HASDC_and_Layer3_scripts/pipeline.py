"""
White-Box Pipeline v4
=====================

Layer 1: Tikhonov graph smoothing          Z = (I + lam*L)^(-1) X
Layer 2: class-wise PCA/SVD subspaces      B_c, mu_c, Sigma_c
Layer 3: corrected geometric discriminant  normalized activation + competition-aware suppression
Layer 4: disabled
Layer 5: argmin discriminant score
"""

import os
import time
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data_loader import load_dataset, to_torch
from layer1_tikhonov import tikhonov_smooth
from layer2_subspace import build_class_subspaces
from layer3_discriminative import compute_discriminative_rperp


class WhiteBoxPipeline:
    """Main white-box graph node classification pipeline."""

    def __init__(self, dataset, config):
        self.dataset = dataset.lower()
        self.config = config
        self.X = None
        self.Y = None
        self.Z = None
        self.sub = None
        self.R_disc = None
        self.disc_weights = None
        self.R_final = None
        self.preds = None
        self.result = None

    @staticmethod
    def _acc(preds, Y, idx):
        idx_t = idx if isinstance(idx, torch.Tensor) else torch.tensor(idx, dtype=torch.long)
        return (preds[idx_t] == Y[idx_t]).float().mean().item()

    def load_data(self):
        feat, labels, adj_norm, lap, tr, va, te, nc = load_dataset(self.dataset)
        self.X, self.Y, self.A, self.L, self.tr_t, self.va_t, self.te_t = to_torch(
            feat, labels, adj_norm, lap, tr, va, te
        )
        self.train_list = tr.tolist()
        self.num_classes = nc
        self.n, self.D = self.X.shape

    def layer1_smooth(self):
        self.Z = tikhonov_smooth(self.X, self.L, self.config['lam'])

    def layer2_subspace(self):
        self.sub = build_class_subspaces(
            self.Z,
            self.Y,
            self.train_list,
            self.num_classes,
            sub_dim=self.config['sub_dim'],
        )

    def layer3_discriminative(self):
        eta_react = self.config.get('eta_react', self.config.get('eta_pos', 0.10))
        self.R_disc, self.disc_weights = compute_discriminative_rperp(
            self.Z,
            self.sub,
            self.num_classes,
            eta_react=eta_react,
            max_activation_scale=self.config.get('max_activation_scale', None),
            activation_mode=self.config.get('activation_mode', 'mahalanobis'),
            activation_rho=self.config.get('activation_rho', 1e-6),
            aggregation_mode=self.config.get('aggregation_mode', 'suppression'),
        )

    def layer4_correction(self):
        self.R_final = self.R_disc

    def layer5_predict(self):
        self.preds = self.R_final.argmin(dim=1)

    def evaluate(self):
        return (
            self._acc(self.preds, self.Y, self.tr_t),
            self._acc(self.preds, self.Y, self.va_t),
            self._acc(self.preds, self.Y, self.te_t),
        )

    def run(self, verbose=True):
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
            'pipeline': 'WhiteBoxPipeline-v4',
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
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"  {self.dataset.upper()} -- WhiteBoxPipeline v4")
        print(f"{'='*60}")
        print(f"  Layer 1: Tikhonov smooth    lam={cfg['lam']}")
        print(f"  Layer 2: Class PCA          sub_dim={cfg['sub_dim']}")
        print(f"  Layer 3: Geometric Reactive  eta_react={cfg.get('eta_react', cfg.get('eta_pos', 0.10))}")
        print(f"           activation={cfg.get('activation_mode', 'mahalanobis')}  aggregation={cfg.get('aggregation_mode', 'suppression')}")
        print(f"  Layer 4: (disabled)")
        print(f"  Layer 5: argmin R_disc")
        print(f"{'-'*60}")
        r = self.result
        print(f"  train={r['train']:.4f}  val={r['val']:.4f}  test={r['test']:.4f}  ({elapsed:.1f}s)")
        print(f"{'='*60}")
