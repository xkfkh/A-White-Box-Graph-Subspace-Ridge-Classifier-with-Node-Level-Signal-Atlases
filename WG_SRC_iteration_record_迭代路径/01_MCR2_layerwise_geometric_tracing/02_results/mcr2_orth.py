#!/usr/bin/env python3
"""
mcr2_orth.py - MCR2 with explicit subspace orthogonality penalty.

Three-way comparison:
  ce          : CrossEntropy only
  mcr2        : CE + MCR2 loss
  mcr2_orth   : CE + MCR2 loss + orthogonality penalty

Orthogonality penalty (differentiable, no SVD):
  Sigma_k = Z_k^T Z_k / n_k   (class covariance, d x d)
  L_orth = sum_{i != j} ||Sigma_i * Sigma_j||_F^2 / (||Sigma_i||_F * ||Sigma_j||_F + eps)

Total loss:
  L = L_CE + lambda_mcr * L_mcr2 + lambda_orth * L_orth

Dataset: synthetic (10 classes, 100 samples/class, low-rank Gaussian subspaces)
Results saved to: results/orth_comparison/
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
# Synthetic dataset
# -------------------------------------------------------
def generate_synthetic_subspace_dataset(
    n_classes: int = 10,
    train_per_class: int = 100,
    test_per_class: int = 100,
    ambient_dim: int = 128,
    subspace_dim: int = 8,
    noise_std: float = 0.15,
    class_scale: float = 2.5,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(seed)
    total_per_class = train_per_class + test_per_class

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for k in range(n_classes):
        basis_raw = rng.normal(size=(ambient_dim, subspace_dim))
        basis, _ = np.linalg.qr(basis_raw)
        mean_dir = rng.normal(size=(ambient_dim,))
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        class_mean = class_scale * mean_dir
        coeffs = rng.normal(size=(total_per_class, subspace_dim))
        subspace_points = coeffs @ basis.T
        noise = noise_std * rng.normal(size=(total_per_class, ambient_dim))
        samples = subspace_points + noise + class_mean
        labels = np.full((total_per_class,), k, dtype=np.int64)
        x_train_list.append(samples[:train_per_class])
        y_train_list.append(labels[:train_per_class])
        x_test_list.append(samples[train_per_class:])
        y_test_list.append(labels[train_per_class:])

    x_train = np.concatenate(x_train_list, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train_list, axis=0).astype(np.int64)
    x_test = np.concatenate(x_test_list, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test_list, axis=0).astype(np.int64)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_ds, test_ds


# -------------------------------------------------------
# Model
# -------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.hidden_layers.append(nn.Linear(in_d, out_d))
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hidden_representations = [x]
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
            hidden_representations.append(h)
        logits = self.classifier(h)
        hidden_representations.append(logits)
        if return_hidden:
            return logits, hidden_representations
        return logits


# -------------------------------------------------------
# Loss functions
# -------------------------------------------------------
def coding_rate_torch(feat: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    m, d = feat.shape
    if d <= m:
        gram = (alpha / m) * (feat.T @ feat)
        mat = torch.eye(d, device=feat.device) + gram
    else:
        gram = (alpha / m) * (feat @ feat.T)
        mat = torch.eye(m, device=feat.device) + gram
    return 0.5 * torch.logdet(mat)


def mcr2_loss_torch(
    z: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 0.5
) -> torch.Tensor:
    """Negative delta-R: minimize -(R(Z) - mean_k R(Z_k))."""
    r_total = coding_rate_torch(z, alpha)
    r_classes = []
    for k in range(num_classes):
        mask = y == k
        if mask.sum() > 1:
            r_classes.append(coding_rate_torch(z[mask], alpha))
    if not r_classes:
        return torch.tensor(0.0, device=z.device)
    r_mean = torch.stack(r_classes).mean()
    return -(r_total - r_mean)


def orth_penalty_torch(
    z: torch.Tensor, y: torch.Tensor, num_classes: int, eps: float = 1e-8
) -> torch.Tensor:
    """
    Differentiable subspace orthogonality penalty.

    For each class k, compute the class covariance:
        Sigma_k = Z_k^T Z_k / n_k   (shape: d x d)

    Penalty:
        L_orth = sum_{i != j} ||Sigma_i @ Sigma_j||_F^2
                              / (||Sigma_i||_F * ||Sigma_j||_F + eps)

    This is fully differentiable through the covariance products.
    When class subspaces are orthogonal, Sigma_i @ Sigma_j ~ 0.
    """
    sigmas = []
    for k in range(num_classes):
        mask = y == k
        n_k = mask.sum()
        if n_k > 1:
            z_k = z[mask]
            sigma_k = (z_k.T @ z_k) / n_k.float()
        else:
            d = z.shape[1]
            sigma_k = torch.zeros(d, d, device=z.device)
        sigmas.append(sigma_k)

    penalty = torch.tensor(0.0, device=z.device)
    count = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            cross = sigmas[i] @ sigmas[j]          # d x d
            cross_norm_sq = (cross * cross).sum()   # ||Sigma_i Sigma_j||_F^2
            norm_i = (sigmas[i] * sigmas[i]).sum().sqrt()
            norm_j = (sigmas[j] * sigmas[j]).sum().sqrt()
            denom = norm_i * norm_j + eps
            penalty = penalty + cross_norm_sq / denom
            count += 1

    if count > 0:
        penalty = penalty / count
    return penalty


# -------------------------------------------------------
# Training helpers
# -------------------------------------------------------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.argmax(logits, dim=1) == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    mode: str,
    lambda_mcr: float = 1.0,
    lambda_orth: float = 0.1,
    alpha: float = 0.5,
) -> Tuple[float, float]:
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if mode == "ce":
            logits = model(x)
            loss = ce_criterion(logits, y)
        else:
            logits, hidden = model(x, return_hidden=True)
            z = hidden[-2]   # last hidden layer before classifier
            ce_loss = ce_criterion(logits, y)
            mcr_loss = mcr2_loss_torch(z, y, num_classes, alpha)
            loss = ce_loss + lambda_mcr * mcr_loss
            if mode == "mcr2_orth":
                orth_loss = orth_penalty_torch(z, y, num_classes)
                loss = loss + lambda_orth * orth_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        running_loss += criterion(logits, y).item()
        running_acc += accuracy_from_logits(logits, y)
        n_batches += 1
    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


# -------------------------------------------------------
# Representation extraction
# -------------------------------------------------------
@torch.no_grad()
def extract_all_representations(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[np.ndarray], np.ndarray]:
    model.eval()
    all_layers: List[List[np.ndarray]] = []
    all_labels: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        _, hidden = model(x, return_hidden=True)
        if not all_layers:
            all_layers = [[] for _ in range(len(hidden))]
        for idx, rep in enumerate(hidden):
            all_layers[idx].append(rep.detach().cpu().numpy())
        all_labels.append(y.numpy())
    layer_arrays = [np.concatenate(parts, axis=0) for parts in all_layers]
    labels = np.concatenate(all_labels, axis=0)
    return layer_arrays, labels


# -------------------------------------------------------
# Geometric metrics (numpy)
# -------------------------------------------------------
def coding_rate_np(z: np.ndarray, alpha: float = 0.5) -> float:
    n, d = z.shape
    if n < 2:
        return 0.0
    if d <= n:
        gram = (alpha / n) * (z.T @ z)
        mat = np.eye(d) + gram
    else:
        gram = (alpha / n) * (z @ z.T)
        mat = np.eye(n) + gram
    sign, logdet = np.linalg.slogdet(mat)
    return 0.5 * float(logdet) if sign > 0 else 0.0


def mcr2_delta_r_np(z: np.ndarray, y: np.ndarray, num_classes: int, alpha: float = 0.5) -> float:
    r_total = coding_rate_np(z, alpha)
    r_classes = [coding_rate_np(z[y == k], alpha) for k in range(num_classes) if (y == k).sum() > 1]
    return r_total - float(np.mean(r_classes)) if r_classes else 0.0


def principal_angles_between(a: np.ndarray, b: np.ndarray, n_components: int = 8) -> np.ndarray:
    def top_basis(x: np.ndarray, k: int) -> np.ndarray:
        x_c = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_c, full_matrices=False)
        return vt[:k].T
    k = min(n_components, a.shape[0] - 1, b.shape[0] - 1, a.shape[1])
    if k < 1:
        return np.array([0.0])
    qa = top_basis(a, k)
    qb = top_basis(b, k)
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.clip(sv, 0.0, 1.0)


def compute_mean_subspace_coherence(
    z: np.ndarray, y: np.ndarray, num_classes: int, n_components: int = 8
) -> float:
    cosines = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            zi, zj = z[y == i], z[y == j]
            if zi.shape[0] < 2 or zj.shape[0] < 2:
                continue
            pa = principal_angles_between(zi, zj, n_components)
            cosines.append(float(pa.max()))
    return float(np.mean(cosines)) if cosines else 0.0


def compute_layer_metrics(
    layer_arrays: List[np.ndarray], labels: np.ndarray, num_classes: int
) -> Dict[str, List[float]]:
    delta_r = []
    coherence = []
    for z in layer_arrays:
        delta_r.append(mcr2_delta_r_np(z, labels, num_classes))
        coherence.append(compute_mean_subspace_coherence(z, labels, num_classes))
    return {"mcr2_delta_r": delta_r, "subspace_coherence": coherence}


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------
def plot_3way_comparison(
    metrics: Dict[str, Dict[str, List[float]]], out_dir: Path
) -> None:
    """
    Two-panel figure:
      left:  delta-R across layers for all three modes
      right: subspace coherence across layers for all three modes
    """
    colors = {"ce": "steelblue", "mcr2": "tomato", "mcr2_orth": "seagreen"}
    markers = {"ce": "o", "mcr2": "s", "mcr2_orth": "^"}
    labels_map = {"ce": "CE", "mcr2": "CE+MCR2", "mcr2_orth": "CE+MCR2+Orth"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for mode, m in metrics.items():
        layers = list(range(len(m["mcr2_delta_r"])))
        axes[0].plot(layers, m["mcr2_delta_r"],
                     marker=markers[mode], color=colors[mode], label=labels_map[mode])
        axes[1].plot(layers, m["subspace_coherence"],
                     marker=markers[mode], color=colors[mode], label=labels_map[mode])

    axes[0].set_title("Delta-R across layers")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Delta-R(Z)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Subspace coherence (lower = more orthogonal)")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Mean max cos(principal angle)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("3-way comparison: CE vs CE+MCR2 vs CE+MCR2+Orth", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / "comparison_3way.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("3-way comparison plot saved:", out_path)


# -------------------------------------------------------
# Experiment log
# -------------------------------------------------------
def write_experiment_log(
    metrics: Dict[str, Dict[str, List[float]]],
    final_accs: Dict[str, float],
    out_dir: Path,
) -> None:
    log_path = out_dir / "experiment_log.md"
    lines = []
    lines.append("# Experiment Log: MCR2 + Orthogonality Penalty\n")
    lines.append("## Setup\n")
    lines.append("- Dataset: synthetic (10 classes, 100 train/100 test per class, low-rank Gaussian subspaces)\n")
    lines.append("- Architecture: MLP, hidden dims 128-64-32\n")
    lines.append("- Epochs: 15\n")
    lines.append("- Modes: ce, mcr2, mcr2_orth\n")
    lines.append("- lambda_mcr=1.0, lambda_orth=0.1\n\n")

    lines.append("## Final Test Accuracy\n")
    for mode, acc in final_accs.items():
        lines.append(f"- {mode}: {acc:.4f}\n")
    lines.append("\n")

    lines.append("## Layer-wise Delta-R\n")
    lines.append("| Layer | CE | MCR2 | MCR2+Orth |\n")
    lines.append("|-------|----|------|-----------|\n")
    n_layers = len(next(iter(metrics.values()))["mcr2_delta_r"])
    for l in range(n_layers):
        row = f"| {l} "
        for mode in ["ce", "mcr2", "mcr2_orth"]:
            val = metrics[mode]["mcr2_delta_r"][l]
            row += f"| {val:.4f} "
        row += "|\n"
        lines.append(row)
    lines.append("\n")

    lines.append("## Layer-wise Subspace Coherence\n")
    lines.append("| Layer | CE | MCR2 | MCR2+Orth |\n")
    lines.append("|-------|----|------|-----------|\n")
    for l in range(n_layers):
        row = f"| {l} "
        for mode in ["ce", "mcr2", "mcr2_orth"]:
            val = metrics[mode]["subspace_coherence"][l]
            row += f"| {val:.4f} "
        row += "|\n"
        lines.append(row)
    lines.append("\n")

    # Analysis
    last_layer = n_layers - 2  # last hidden layer (before logits)
    ce_coh = metrics["ce"]["subspace_coherence"][last_layer]
    mcr2_coh = metrics["mcr2"]["subspace_coherence"][last_layer]
    orth_coh = metrics["mcr2_orth"]["subspace_coherence"][last_layer]

    lines.append("## Analysis\n")
    lines.append(f"Last hidden layer (layer {last_layer}) subspace coherence:\n")
    lines.append(f"- CE:          {ce_coh:.4f}\n")
    lines.append(f"- MCR2:        {mcr2_coh:.4f}\n")
    lines.append(f"- MCR2+Orth:   {orth_coh:.4f}\n\n")

    if orth_coh < mcr2_coh - 0.01:
        lines.append("**Conclusion**: The orthogonality penalty successfully reduced subspace coherence "
                     "compared to plain MCR2. Explicit orthogonalization constraint is effective.\n")
    elif orth_coh < ce_coh - 0.01:
        lines.append("**Conclusion**: MCR2+Orth reduced coherence vs CE baseline, but improvement over "
                     "plain MCR2 is marginal. May need larger lambda_orth or more epochs.\n")
    else:
        lines.append("**Conclusion**: Orthogonality penalty did not significantly reduce coherence in this "
                     "run. Consider increasing lambda_orth or training longer.\n")

    lines.append("\n## Delta-R at Last Hidden Layer\n")
    for mode in ["ce", "mcr2", "mcr2_orth"]:
        val = metrics[mode]["mcr2_delta_r"][last_layer]
        lines.append(f"- {mode}: {val:.4f}\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Experiment log saved:", log_path)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main() -> None:
    SEED = 42
    EPOCHS = 15
    BATCH_SIZE = 128
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    HIDDEN_DIMS = [128, 64, 32]
    NUM_CLASSES = 10
    LAMBDA_MCR = 1.0
    LAMBDA_ORTH = 0.1
    OUTPUT_DIR = Path("results/orth_comparison")
    MODES = ["ce", "mcr2", "mcr2_orth"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    set_seed(SEED)
    train_ds, test_ds = generate_synthetic_subspace_dataset(
        n_classes=NUM_CLASSES,
        train_per_class=100,
        test_per_class=100,
        seed=SEED,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_metrics: Dict[str, Dict[str, List[float]]] = {}
    final_accs: Dict[str, float] = {}

    for mode in MODES:
        print(f"\n{'='*55}")
        print(f"  Training mode: {mode.upper()}")
        print(f"{'='*55}")
        set_seed(SEED)
        model = MLP(128, HIDDEN_DIMS, NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        mode_dir = OUTPUT_DIR / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                num_classes=NUM_CLASSES,
                mode=mode,
                lambda_mcr=LAMBDA_MCR,
                lambda_orth=LAMBDA_ORTH,
            )
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(
                f"[{mode}] Epoch {epoch:02d}/{EPOCHS} | "
                f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

        final_accs[mode] = test_acc
        torch.save(model.state_dict(), mode_dir / "model.pt")

        layer_arrays, labels = extract_all_representations(model, test_loader, device)
        metrics = compute_layer_metrics(layer_arrays, labels, NUM_CLASSES)
        all_metrics[mode] = metrics

        with open(mode_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{mode}] Delta-R:           ", [f"{v:.3f}" for v in metrics["mcr2_delta_r"]])
        print(f"[{mode}] Subspace coherence:", [f"{v:.3f}" for v in metrics["subspace_coherence"]])

    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE (last hidden layer)")
    print("="*70)
    n_layers = len(next(iter(all_metrics.values()))["mcr2_delta_r"])
    last_hidden = n_layers - 2
    print(f"{'Mode':<15} {'Delta-R':>10} {'Coherence':>12} {'Test Acc':>10}")
    print("-"*50)
    for mode in MODES:
        dr = all_metrics[mode]["mcr2_delta_r"][last_hidden]
        coh = all_metrics[mode]["subspace_coherence"][last_hidden]
        acc = final_accs[mode]
        print(f"{mode:<15} {dr:>10.4f} {coh:>12.4f} {acc:>10.4f}")

    # Plots and logs
    plot_3way_comparison(all_metrics, OUTPUT_DIR)
    write_experiment_log(all_metrics, final_accs, OUTPUT_DIR)

    # Save experiment_summary.txt
    result_path = OUTPUT_DIR / "experiment_summary.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("mcr2_orth.py experiment results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Script path: D:/desktop/MSR/mcr2_orth.py\n\n")
        f.write("Final Test Accuracy\n")
        f.write("-"*30 + "\n")
        for mode, acc in final_accs.items():
            f.write(f"  {mode}: {acc:.4f}\n")
        f.write("\n")
        f.write("Layer-wise Delta-R\n")
        f.write("-"*30 + "\n")
        for mode in MODES:
            vals = all_metrics[mode]["mcr2_delta_r"]
            f.write(f"  {mode}: {[round(v,4) for v in vals]}\n")
        f.write("\n")
        f.write("Layer-wise Subspace Coherence\n")
        f.write("-"*30 + "\n")
        for mode in MODES:
            vals = all_metrics[mode]["subspace_coherence"]
            f.write(f"  {mode}: {[round(v,4) for v in vals]}\n")
        f.write("\n")
        f.write("Last Hidden Layer Summary\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Mode':<15} {'Delta-R':>10} {'Coherence':>12} {'Test Acc':>10}\n")
        for mode in MODES:
            dr = all_metrics[mode]["mcr2_delta_r"][last_hidden]
            coh = all_metrics[mode]["subspace_coherence"][last_hidden]
            acc = final_accs[mode]
            f.write(f"{mode:<15} {dr:>10.4f} {coh:>12.4f} {acc:>10.4f}\n")
        f.write("\n")
        f.write("Output directory: results/orth_comparison/\n")
        f.write("Comparison plot:  results/orth_comparison/comparison_3way.png\n")
        f.write("Experiment log:   results/orth_comparison/experiment_log.md\n")

    print("\nDone. Results in:", OUTPUT_DIR)
    print("experiment_summary.txt saved:", result_path)


if __name__ == "__main__":
    main()



