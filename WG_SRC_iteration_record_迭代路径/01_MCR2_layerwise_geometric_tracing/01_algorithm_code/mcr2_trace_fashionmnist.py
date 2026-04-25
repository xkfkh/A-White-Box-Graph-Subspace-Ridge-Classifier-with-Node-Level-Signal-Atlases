#!/usr/bin/env python3
"""
MCR²-style layerwise geometric tracing for a simple MLP.

What this script does
---------------------
1. Loads a dataset:
   - FashionMNIST (auto-download via torchvision), or
   - a synthetic subspace dataset generated on the fly.
2. Trains a baseline MLP with cross-entropy.
3. Caches the representation of every test sample at every layer.
4. Computes layerwise geometric metrics:
   - within-class compression W_l
   - between-class separation B_l
   - average effective rank across classes
   - per-sample trajectory metrics (d_in, d_out, margin)
5. Saves plots, activations, and metrics to an output folder.

This is the "stage-1" experiment for testing whether a model's internal
representations evolve in a way consistent with MCR²-style geometry.

Example usage
-------------
python mcr2_trace_fashionmnist.py --dataset fashionmnist --epochs 15
python mcr2_trace_fashionmnist.py --dataset synthetic --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from torchvision import datasets, transforms
except Exception as e:
    raise RuntimeError(
        "Failed to import torchvision. Please install torchvision first."
    ) from e


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic dataset generator
# -----------------------------
def generate_synthetic_subspace_dataset(
    n_classes: int = 10,
    train_per_class: int = 600,
    test_per_class: int = 100,
    ambient_dim: int = 128,
    subspace_dim: int = 8,
    noise_std: float = 0.15,
    class_scale: float = 2.5,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Generate a labeled dataset where each class is concentrated near a low-dimensional
    linear subspace plus a class-dependent mean shift.

    This is useful for a clean first-pass theoretical experiment because it creates
    data whose geometry is already close to the assumptions behind subspace-based
    representation theories.
    """
    rng = np.random.default_rng(seed)
    total_per_class = train_per_class + test_per_class

    x_train_list: List[np.ndarray] = []
    y_train_list: List[np.ndarray] = []
    x_test_list: List[np.ndarray] = []
    y_test_list: List[np.ndarray] = []

    for k in range(n_classes):
        # Random orthonormal basis for a class-specific subspace.
        basis_raw = rng.normal(size=(ambient_dim, subspace_dim))
        basis, _ = np.linalg.qr(basis_raw)

        # Class-specific mean direction.
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

    # Normalize each feature using training statistics only.
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_ds, test_ds


# -----------------------------
# Dataset loading
# -----------------------------
@dataclass
class DatasetBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    input_dim: int
    num_classes: int
    class_names: List[str]


FASHION_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_dataset(name: str, data_root: Path, seed: int) -> DatasetBundle:
    if name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_dataset = datasets.FashionMNIST(
            root=str(data_root), train=True, transform=transform, download=True
        )
        test_dataset = datasets.FashionMNIST(
            root=str(data_root), train=False, transform=transform, download=True
        )
        return DatasetBundle(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            input_dim=28 * 28,
            num_classes=10,
            class_names=FASHION_CLASS_NAMES,
        )

    if name == "synthetic":
        train_ds, test_ds = generate_synthetic_subspace_dataset(seed=seed)
        return DatasetBundle(
            train_dataset=train_ds,
            test_dataset=test_ds,
            input_dim=128,
            num_classes=10,
            class_names=[f"class_{i}" for i in range(10)],
        )

    raise ValueError(f"Unsupported dataset: {name}")


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        dims = [input_dim, *hidden_dims]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.hidden_layers.append(nn.Linear(in_dim, out_dim))
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


# -----------------------------
# Training / evaluation
# -----------------------------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


def mcr2_loss_torch(z: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 0.5) -> torch.Tensor:
    """
    Differentiable MCR² loss: minimize -ΔR = -(R(Z) - mean_k R(Z_k))
    z: [n, d] feature tensor (last hidden layer, before classifier)
    """
    n, d = z.shape

    def coding_rate_torch(feat: torch.Tensor) -> torch.Tensor:
        m = feat.shape[0]
        if d <= m:
            gram = (alpha / m) * (feat.T @ feat)
            mat = torch.eye(d, device=feat.device) + gram
        else:
            gram = (alpha / m) * (feat @ feat.T)
            mat = torch.eye(m, device=feat.device) + gram
        return 0.5 * torch.logdet(mat)

    r_total = coding_rate_torch(z)
    r_classes = []
    for k in range(num_classes):
        mask = y == k
        if mask.sum() > 1:
            r_classes.append(coding_rate_torch(z[mask]))
    r_mean = torch.stack(r_classes).mean()
    return -(r_total - r_mean)   # minimise negative ΔR


def train_one_epoch_mcr2(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    alpha: float = 0.5,
) -> Tuple[float, float]:
    """Train with MCR² loss on the last hidden layer + CE on logits (weighted sum)."""
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, hidden = model(x, return_hidden=True)
        z = hidden[-2]   # last hidden layer before classifier

        ce_loss = ce_criterion(logits, y)
        mcr_loss = mcr2_loss_torch(z, y, num_classes, alpha)
        # CE keeps accuracy; MCR² shapes the geometry. Equal weight.
        loss = ce_loss + mcr_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


# -----------------------------
# Representation extraction
# -----------------------------
@torch.no_grad()
def extract_all_representations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Returns:
        layer_arrays: list of arrays, one per layer representation
                      layer_arrays[l].shape = [N, D_l]
        labels: shape [N]

    Layer convention:
        0 = input
        1..L = hidden layers after ReLU
        L+1 = logits
    """
    model.eval()
    all_layers: List[List[np.ndarray]] = []
    all_labels: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        logits, hidden = model(x, return_hidden=True)
        if not all_layers:
            all_layers = [[] for _ in range(len(hidden))]
        for idx, rep in enumerate(hidden):
            all_layers[idx].append(rep.detach().cpu().numpy())
        all_labels.append(y.numpy())

    layer_arrays = [np.concatenate(parts, axis=0) for parts in all_layers]
    labels = np.concatenate(all_labels, axis=0)
    return layer_arrays, labels


# -----------------------------
# Geometric metrics
# -----------------------------
def compute_class_means(z: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    means = []
    for k in range(num_classes):
        cls = z[y == k]
        means.append(cls.mean(axis=0))
    return np.stack(means, axis=0)



def compute_within_class_scatter(z: np.ndarray, y: np.ndarray, means: np.ndarray) -> float:
    diffs = z - means[y]
    return float(np.mean(np.sum(diffs * diffs, axis=1)))



def compute_between_class_scatter(means: np.ndarray) -> float:
    k = means.shape[0]
    pairwise_sq = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            diff = means[i] - means[j]
            pairwise_sq.append(float(np.dot(diff, diff)))
    return float(np.mean(pairwise_sq)) if pairwise_sq else 0.0



def effective_rank_from_cov(x: np.ndarray) -> float:
    """
    Participation ratio:
        r_eff = (sum lambda)^2 / sum lambda^2
    where lambda are eigenvalues of the covariance matrix.

    Fix: pass raw x to np.cov (rowvar=False), which centers internally.
    The previous version manually centered first, causing double-centering.
    """
    if x.shape[0] < 2:
        return 1.0
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    denom = float(np.sum(eigvals ** 2))
    numer = float(np.sum(eigvals) ** 2)
    if denom <= 1e-12:
        return 0.0
    return numer / denom



def compute_average_effective_rank(z: np.ndarray, y: np.ndarray, num_classes: int) -> float:
    ranks = []
    for k in range(num_classes):
        cls = z[y == k]
        ranks.append(effective_rank_from_cov(cls))
    return float(np.mean(ranks))



# -----------------------------
# MCR² rate function
# -----------------------------
def coding_rate(z: np.ndarray, alpha: float) -> float:
    """
    R(Z) = 1/2 * log det(I + alpha/n * Z Z^T)

    z: shape [n, d]
    Uses the identity  log det(I_n + alpha/n * Z Z^T) = log det(I_d + alpha/n * Z^T Z)
    when d < n (cheaper), otherwise uses the n×n form.
    """
    n, d = z.shape
    if n < 2:
        return 0.0
    if d <= n:
        gram = (alpha / n) * (z.T @ z)          # d×d
        mat = np.eye(d) + gram
    else:
        gram = (alpha / n) * (z @ z.T)          # n×n
        mat = np.eye(n) + gram
    sign, logdet = np.linalg.slogdet(mat)
    if sign <= 0:
        return 0.0
    return 0.5 * float(logdet)


def mcr2_delta_r(z: np.ndarray, y: np.ndarray, num_classes: int, alpha: float = 0.5) -> float:
    """
    ΔR(Z) = R(Z) - (1/K) * sum_k R(Z_k)

    Maximising this simultaneously compresses within-class and separates between-class.
    """
    r_total = coding_rate(z, alpha)
    r_class_mean = float(np.mean([
        coding_rate(z[y == k], alpha)
        for k in range(num_classes)
        if np.sum(y == k) > 1
    ]))
    return r_total - r_class_mean


# -----------------------------
# Principal angles between class subspaces
# -----------------------------
def principal_angles_between(a: np.ndarray, b: np.ndarray, n_components: int = 8) -> np.ndarray:
    """
    Compute principal angles between the subspaces spanned by class a and class b.
    Uses SVD of the cross-covariance of their centered, whitened representations.
    Returns cosines of principal angles (1 = parallel, 0 = orthogonal).
    """
    def top_basis(x: np.ndarray, k: int) -> np.ndarray:
        x_c = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_c, full_matrices=False)
        return vt[:k].T  # shape [d, k]

    k = min(n_components, a.shape[0] - 1, b.shape[0] - 1, a.shape[1])
    if k < 1:
        return np.array([0.0])
    qa = top_basis(a, k)   # [d, k]
    qb = top_basis(b, k)   # [d, k]
    cross = qa.T @ qb      # [k, k]
    sv = np.linalg.svd(cross, compute_uv=False)
    return np.clip(sv, 0.0, 1.0)


def compute_mean_subspace_coherence(z: np.ndarray, y: np.ndarray, num_classes: int, n_components: int = 8) -> float:
    """
    Average max principal-angle cosine across all class pairs.
    Low value → subspaces are nearly orthogonal (MCR² ideal).
    High value → subspaces overlap (poor separation).
    """
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
    layer_arrays: List[np.ndarray],
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, List[float]]:
    within = []
    between = []
    eff_rank = []
    layer_dims = []
    delta_r = []
    subspace_coherence = []

    for z in layer_arrays:
        means = compute_class_means(z, labels, num_classes)
        within.append(compute_within_class_scatter(z, labels, means))
        between.append(compute_between_class_scatter(means))
        eff_rank.append(compute_average_effective_rank(z, labels, num_classes))
        layer_dims.append(int(z.shape[1]))
        delta_r.append(mcr2_delta_r(z, labels, num_classes))
        subspace_coherence.append(compute_mean_subspace_coherence(z, labels, num_classes))

    return {
        "within_class_scatter": within,
        "between_class_scatter": between,
        "average_effective_rank": eff_rank,
        "layer_dims": layer_dims,
        "mcr2_delta_r": delta_r,
        "subspace_coherence": subspace_coherence,
    }



def compute_single_sample_trajectories(
    layer_arrays: List[np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    samples_per_class: int = 1,
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    rng = np.random.default_rng(seed)
    chosen_indices: List[int] = []
    for k in range(num_classes):
        cls_idx = np.where(labels == k)[0]
        take = min(samples_per_class, len(cls_idx))
        picked = rng.choice(cls_idx, size=take, replace=False)
        chosen_indices.extend(picked.tolist())

    trajectories: Dict[str, Dict[str, List[float]]] = {}
    for idx in chosen_indices:
        label = int(labels[idx])
        key = f"sample_{idx}_class_{label}"
        trajectories[key] = {
            "label": label,
            "d_in": [],
            "d_out": [],
            "margin": [],
        }
        for z in layer_arrays:
            means = compute_class_means(z, labels, num_classes)
            point = z[idx]
            d_in = float(np.linalg.norm(point - means[label]))
            other_centers = np.delete(means, label, axis=0)
            d_out = float(np.min(np.linalg.norm(other_centers - point[None, :], axis=1)))
            margin = d_out - d_in
            trajectories[key]["d_in"].append(d_in)
            trajectories[key]["d_out"].append(d_out)
            trajectories[key]["margin"].append(margin)

    return trajectories


# -----------------------------
# Plotting
# -----------------------------
def plot_metric_curve(values: Sequence[float], ylabel: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    layers = list(range(len(values)))
    plt.plot(layers, values, marker="o")
    plt.xlabel("Layer index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_single_sample_trajectories(
    trajectories: Dict[str, Dict[str, List[float]]], out_dir: Path
) -> None:
    for key, values in trajectories.items():
        layers = list(range(len(values["margin"])))

        plt.figure(figsize=(7, 4.5))
        plt.plot(layers, values["d_in"], marker="o", label="d_in")
        plt.plot(layers, values["d_out"], marker="o", label="d_out")
        plt.xlabel("Layer index")
        plt.ylabel("Distance")
        plt.title(f"Trajectory distances: {key}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}_distances.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.plot(layers, values["margin"], marker="o")
        plt.xlabel("Layer index")
        plt.ylabel("Margin = d_out - d_in")
        plt.title(f"Trajectory margin: {key}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}_margin.png", dpi=180)
        plt.close()



def plot_mcr2_summary(metrics: Dict[str, List[float]], out_dir: Path) -> None:
    """Single figure with 4 subplots: ΔR, subspace coherence, within scatter, between scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    layers = list(range(len(metrics["mcr2_delta_r"])))

    axes[0, 0].plot(layers, metrics["mcr2_delta_r"], marker="o", color="steelblue")
    axes[0, 0].set_title("MCR² ΔR across layers")
    axes[0, 0].set_xlabel("Layer index")
    axes[0, 0].set_ylabel("ΔR(Z)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(layers, metrics["subspace_coherence"], marker="o", color="tomato")
    axes[0, 1].set_title("Subspace coherence (lower = more orthogonal)")
    axes[0, 1].set_xlabel("Layer index")
    axes[0, 1].set_ylabel("Mean max cos(principal angle)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(layers, metrics["within_class_scatter"], marker="o", color="seagreen")
    axes[1, 0].set_title("Within-class scatter W_l")
    axes[1, 0].set_xlabel("Layer index")
    axes[1, 0].set_ylabel("W_l")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(layers, metrics["between_class_scatter"], marker="o", color="darkorange")
    axes[1, 1].set_title("Between-class scatter B_l")
    axes[1, 1].set_xlabel("Layer index")
    axes[1, 1].set_ylabel("B_l")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "mcr2_summary.png", dpi=180)
    plt.close()


def _plot_comparison(ce_metrics: Dict, mcr2_metrics: Dict, out_dir: Path) -> None:
    """Side-by-side comparison of CE vs MCR2 training on key geometric metrics."""
    keys = [
        ("subspace_coherence",   "Subspace coherence (lower=more orthogonal)"),
        ("mcr2_delta_r",         "MCR2 Delta_R"),
        ("within_class_scatter", "Within-class scatter W_l"),
        ("between_class_scatter","Between-class scatter B_l"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (key, title) in zip(axes.flat, keys):
        layers = list(range(len(ce_metrics[key])))
        ax.plot(layers, ce_metrics[key],   marker="o", label="CE",   color="steelblue")
        ax.plot(layers, mcr2_metrics[key], marker="s", label="CE+MCR2", color="tomato")
        ax.set_title(title)
        ax.set_xlabel("Layer index")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("CE vs CE+MCR2 training: geometric comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=180)
    plt.close()
    print("Comparison plot saved:", out_dir / "comparison.png")


def plot_training_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    for metric_name in ["train_loss", "train_acc", "test_loss", "test_acc"]:
        plt.figure(figsize=(7, 4.5))
        epochs = np.arange(1, len(history[metric_name]) + 1)
        plt.plot(epochs, history[metric_name], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(metric_name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_name}.png", dpi=180)
        plt.close()


# -----------------------------
# Main experiment
# -----------------------------
def parse_hidden_dims(hidden_dims_str: str) -> List[int]:
    return [int(x.strip()) for x in hidden_dims_str.split(",") if x.strip()]



def save_numpy_representations(layer_arrays: List[np.ndarray], out_dir: Path) -> None:
    rep_dir = out_dir / "representations"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for idx, arr in enumerate(layer_arrays):
        np.save(rep_dir / f"layer_{idx}.npy", arr)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashionmnist", choices=["fashionmnist", "synthetic"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/mcr2_trace")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=str, default="256,128,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)  # 0 = main process, safe on Windows
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--loss-mode", type=str, default="both", choices=["ce", "mcr2", "both"],
                        help="ce=CrossEntropy only, mcr2=CE+MCR2, both=run both and compare")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(args.dataset, Path(args.data_root), seed=args.seed)
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    modes = ["ce", "mcr2"] if args.loss_mode == "both" else [args.loss_mode]
    all_metrics: Dict[str, Dict] = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Training mode: {mode.upper()}")
        print(f"{'='*50}")
        set_seed(args.seed)
        model = MLP(bundle.input_dim, hidden_dims, bundle.num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        for epoch in range(1, args.epochs + 1):
            if mode == "ce":
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            else:
                train_loss, train_acc = train_one_epoch_mcr2(
                    model, train_loader, optimizer, device, bundle.num_classes)
            test_loss, test_acc = evaluate(model, test_loader, device)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            print(f"[{mode}] Epoch {epoch:03d}/{args.epochs} | "
                  f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        torch.save(model.state_dict(), mode_dir / "model.pt")
        layer_arrays, labels = extract_all_representations(model, test_loader, device)
        save_numpy_representations(layer_arrays, mode_dir)

        metrics = compute_layer_metrics(layer_arrays, labels, bundle.num_classes)
        trajectories = compute_single_sample_trajectories(
            layer_arrays, labels, bundle.num_classes,
            samples_per_class=args.samples_per_class, seed=args.seed)

        with open(mode_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(mode_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump({**vars(args), "mode": mode}, f, indent=2)

        plot_training_curves(history, mode_dir)
        plot_mcr2_summary(metrics, mode_dir)
        plot_single_sample_trajectories(trajectories, mode_dir)

        print(f"[{mode}] Delta_R:          ", [f"{v:.3f}" for v in metrics["mcr2_delta_r"]])
        print(f"[{mode}] Subspace coherence:", [f"{v:.3f}" for v in metrics["subspace_coherence"]])
        all_metrics[mode] = metrics

    # --- comparison plot when both modes were run ---
    if args.loss_mode == "both":
        _plot_comparison(all_metrics["ce"], all_metrics["mcr2"], output_dir)

    print("\nDone. Results in:", output_dir)


if __name__ == "__main__":
    main()

