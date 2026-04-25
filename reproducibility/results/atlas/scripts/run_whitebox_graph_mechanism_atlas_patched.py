#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
White-box Graph Mechanism Atlas for SRC-v16c.

This script does NOT depend on E08.
It reruns SRC-v16c selection, then performs side-channel white-box analysis
on the selected best config.

Outputs per dataset:
A_graph_signal_barcode.png/csv
B_class_signal_glyph.png/csv
C_geometry_boundary_plane.png/csv
D_subspace_constellation.png/csv
E_error_signal_shift.png/csv
F_split_mechanism_stability_strip.png/csv
whitebox_dataset_summary.csv
WHITEBOX_MECHANISM_REPORT.txt
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from paperexp import core


EPS = 1e-12

BLOCK_ORDER = [
    "raw",
    "row_1hop",
    "row_2hop",
    "row_3hop",
    "highpass_row_1",
    "highpass_row_2",
    "sym_1hop",
    "sym_2hop",
    "highpass_sym_1",
]

BLOCK_LABEL = {
    "raw": "X",
    "row_1hop": "P_r X",
    "row_2hop": "P_r^2 X",
    "row_3hop": "P_r^3 X",
    "highpass_row_1": "X-P_r X",
    "highpass_row_2": "P_r X-P_r^2 X",
    "sym_1hop": "P_s X",
    "sym_2hop": "P_s^2 X",
    "highpass_sym_1": "X-P_s X",
}

I_NAME = {
    "raw": "I_X",
    "row_1hop": "I_PrX",
    "row_2hop": "I_Pr2X",
    "row_3hop": "I_Pr3X",
    "highpass_row_1": "I_XminusPrX",
    "highpass_row_2": "I_PrXminusPr2X",
    "sym_1hop": "I_PsX",
    "sym_2hop": "I_Ps2X",
    "highpass_sym_1": "I_XminusPsX",
}

RAW_BLOCKS = ["raw"]
LOW_BLOCKS = ["row_1hop", "row_2hop", "row_3hop", "sym_1hop", "sym_2hop"]
HIGH_BLOCKS = ["highpass_row_1", "highpass_row_2", "highpass_sym_1"]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m]
    bb = b[m]
    if np.std(aa) < EPS or np.std(bb) < EPS:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def normalize_scores_by_train(S: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    return S / (np.std(S[train_idx]) + EPS)


def wrong_min_margin(score: np.ndarray, y: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute margin = wrong_min_score - true_score.

    Robust case:
    Some tiny datasets / random splits may have a class label appearing in y
    but absent from train_idx, so that class is not present in `classes` and
    the score matrix has no column for it. In that case, the true-class score
    is undefined. We set margin / true_score / wrong_min to NaN for those nodes
    instead of crashing.
    """
    class_to_pos = {int(c): i for i, c in enumerate(classes)}

    n = len(y)
    margin = np.full(n, np.nan, dtype=float)
    true_score = np.full(n, np.nan, dtype=float)
    wrong_min = np.full(n, np.nan, dtype=float)

    for i, yy in enumerate(y):
        yy = int(yy)
        if yy not in class_to_pos:
            continue

        pos = class_to_pos[yy]
        true_score[i] = score[i, pos]

        tmp = score[i].copy()
        tmp[pos] = np.inf
        wrong_min[i] = np.min(tmp)
        margin[i] = wrong_min[i] - true_score[i]

    return margin, true_score, wrong_min


def infer_node_type(P_block: np.ndarray, raw_share: np.ndarray, low_share: np.ndarray, high_share: np.ndarray) -> List[str]:
    out = []
    for k, row in enumerate(P_block):
        R = raw_share[k]
        L = low_share[k]
        H = high_share[k]
        two = row[2] + row[7]
        one = row[1] + row[6]
        sym = row[6] + row[7] + row[8]
        row_based = row[1] + row[2] + row[3] + row[4] + row[5]

        if two > one + 1e-9:
            label = "two-hop-node"
        elif sym > row_based + 1e-9:
            label = "sym-sensitive-node"
        elif R >= L and R >= H:
            label = "raw-node"
        elif L >= R and L >= H:
            label = "low-pass-node"
        elif H >= R and H >= L:
            label = "high-pass-node"
        else:
            label = "mixed-node"
        out.append(label)
    return out


def compute_family_adjusted_shares_from_evidence(P_evidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = {b: i for i, b in enumerate(BLOCK_ORDER)}

    def fam_mean(blocks: List[str]) -> np.ndarray:
        cols = [idx[b] for b in blocks]
        if len(cols) == 0:
            return np.zeros(P_evidence.shape[0], dtype=float)
        return np.mean(P_evidence[:, cols], axis=1)

    raw_ev = fam_mean(RAW_BLOCKS)
    low_ev = fam_mean(LOW_BLOCKS)
    high_ev = fam_mean(HIGH_BLOCKS)
    denom = raw_ev + low_ev + high_ev + EPS
    raw_share = raw_ev / denom
    low_share = low_ev / denom
    high_share = high_ev / denom
    return raw_share, low_share, high_share, np.stack([raw_ev, low_ev, high_ev], axis=1)


def plot_A_barcode(ds: str, node_df: pd.DataFrame, out_dir: Path, max_nodes: int = 900):
    cols = [I_NAME[b].replace("I_", "P_") for b in BLOCK_ORDER]
    df = node_df[node_df["split"] == "test"].copy()
    if df.empty:
        df = node_df.copy()

    sort_cols = [c for c in ["HHI_i", "M_pca", "final_correct"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True, True, False][:len(sort_cols)])

    if len(df) > max_nodes:
        idx = np.linspace(0, len(df) - 1, max_nodes).astype(int)
        df_plot = df.iloc[idx].copy()
    else:
        df_plot = df.copy()

    M = df_plot[cols].to_numpy(dtype=float)

    plt.figure(figsize=(10, 7))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.colorbar(label="P_b(i)")
    plt.xticks(range(len(cols)), [BLOCK_LABEL[b] for b in BLOCK_ORDER], rotation=45, ha="right", fontsize=8)
    plt.ylabel("test nodes sorted by HHI_i")
    plt.title(f"A. Graph Signal Barcode - {ds}")
    plt.tight_layout()
    plt.savefig(out_dir / "A_graph_signal_barcode.png", dpi=220)
    plt.close()


def plot_B_class_glyph(ds: str, class_df: pd.DataFrame, out_dir: Path):
    df = class_df.copy()
    if df.empty:
        return

    classes = df["class"].astype(str).tolist()
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
    ax.set_title(f"B. Class Signal Glyph - {ds}")
    ax.set_xlim(-0.8, len(classes) - 0.2)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("mechanism strength")

    for i, (_, r) in enumerate(df.iterrows()):
        R = safe_float(r.get("R_c"))
        L = safe_float(r.get("L_c"))
        H = safe_float(r.get("H_c"))
        T = safe_float(r.get("T_c"))
        S = safe_float(r.get("S_c"))

        ax.add_patch(Circle((i, 0.30), 0.10 + 0.28 * R, alpha=0.35))
        ax.add_patch(Circle((i, 0.58), 0.10 + 0.28 * L, alpha=0.35))
        ax.add_patch(Circle((i, 0.86), 0.10 + 0.28 * H, alpha=0.35))

        ax.text(i, 0.30, "R", ha="center", va="center", fontsize=8)
        ax.text(i, 0.58, "L", ha="center", va="center", fontsize=8)
        ax.text(i, 0.86, "H", ha="center", va="center", fontsize=8)
        ax.text(i, 1.05, f"T={T:.2f}\nS={S:.2f}", ha="center", va="top", fontsize=7)

    ax.text(-0.7, 0.30, "Raw", va="center")
    ax.text(-0.7, 0.58, "Low", va="center")
    ax.text(-0.7, 0.86, "High", va="center")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "B_class_signal_glyph.png", dpi=220)
    plt.close()


def plot_C_geometry_boundary(ds: str, node_df: pd.DataFrame, out_dir: Path, max_nodes: int = 2500):
    df = node_df[node_df["split"] == "test"].copy()
    if df.empty:
        return
    if len(df) > max_nodes:
        df = df.sample(max_nodes, random_state=0)

    x = df["M_pca"].to_numpy(float)
    y = df["M_ridge"].to_numpy(float)
    correct = df["final_correct"].to_numpy(int)
    size = 15 + 55 * np.clip(df["HHI_i"].to_numpy(float), 0, 3) / 3

    plt.figure(figsize=(7, 6))
    plt.axvline(0, lw=1)
    plt.axhline(0, lw=1)
    plt.scatter(x[correct == 1], y[correct == 1], s=size[correct == 1], alpha=0.55, label="final correct")
    plt.scatter(x[correct == 0], y[correct == 0], s=size[correct == 0], alpha=0.85, marker="x", label="final wrong")
    plt.xlabel("M_pca(i) = R_pca_wrong_min - R_pca_true")
    plt.ylabel("M_ridge(i) = R_ridge_wrong_min - R_ridge_true")
    plt.title(f"C. Geometry-Boundary Mechanism Plane - {ds}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "C_geometry_boundary_mechanism_plane.png", dpi=220)
    plt.close()


def classical_mds_from_overlap(pair_df: pd.DataFrame, classes: List[int]):
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    O = np.eye(k)
    for _, r in pair_df.iterrows():
        a = int(r["class_a"])
        b = int(r["class_b"])
        val = safe_float(r["O_ab"], 0.0)
        if a in idx and b in idx:
            O[idx[a], idx[b]] = val
            O[idx[b], idx[a]] = val
    D = np.maximum(0.0, 1.0 - O)
    J = np.eye(k) - np.ones((k, k)) / k
    B = -0.5 * J @ (D ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    coords = np.zeros((k, 2), dtype=float)
    for j in range(min(2, k)):
        if evals[j] > 0:
            coords[:, j] = evecs[:, j] * np.sqrt(evals[j])
    return coords


def plot_D_constellation(ds: str, class_geom_df: pd.DataFrame, pair_df: pd.DataFrame, out_dir: Path):
    if class_geom_df.empty:
        return

    classes = [int(c) for c in class_geom_df["class"].tolist()]
    coords = classical_mds_from_overlap(pair_df, classes)
    pos = {c: coords[i] for i, c in enumerate(classes)}
    dim_map = {int(r["class"]): safe_float(r["C_c"], 1.0) for _, r in class_geom_df.iterrows()}

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.set_title(f"D. Subspace Constellation - {ds}")

    if not pair_df.empty:
        for _, r in pair_df.iterrows():
            a = int(r["class_a"])
            b = int(r["class_b"])
            if a not in pos or b not in pos:
                continue
            O = safe_float(r.get("O_ab"), 0.0)
            conf = safe_float(r.get("ConfPair_ab"), 0.0)
            if O <= 0 and conf <= 0:
                continue
            xa, ya = pos[a]
            xb, yb = pos[b]
            lw = 0.5 + 5.0 * min(1.0, O)
            alpha = min(0.85, 0.15 + 0.7 * max(O, conf))
            ax.plot([xa, xb], [ya, yb], lw=lw, alpha=alpha)

    for c in classes:
        x, y = pos[c]
        size = 150 + 45 * dim_map.get(c, 1.0)
        ax.scatter([x], [y], s=size, alpha=0.75)
        ax.text(x, y, str(c), ha="center", va="center", fontsize=9)

    ax.set_xlabel("subspace geometry axis 1")
    ax.set_ylabel("subspace geometry axis 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "D_subspace_constellation.png", dpi=220)
    plt.close()


def plot_E_error_shift(ds: str, shift_df: pd.DataFrame, out_dir: Path):
    if shift_df.empty:
        return

    df = shift_df.copy()
    if "level" in df.columns and np.any(df["level"] == "family"):
        df = df[df["level"] == "family"].copy()
        label_col = "family_label"
        xlabel = "Delta_f = share_wrong - share_correct"
    else:
        label_col = "block_label"
        xlabel = "Delta_b = P_b_wrong - P_b_correct"

    y = np.arange(len(df))
    delta = df["Delta_wrong_minus_correct"].to_numpy(float)

    plt.figure(figsize=(8, 5))
    plt.axvline(0, lw=1)
    plt.hlines(y, 0, delta, lw=2)
    plt.scatter(delta, y, s=45)
    plt.yticks(y, df[label_col].tolist())
    plt.xlabel(xlabel)
    plt.title(f"E. Error Signal Shift - {ds}")
    plt.tight_layout()
    plt.savefig(out_dir / "E_error_signal_shift.png", dpi=220)
    plt.close()


def plot_F_stability(ds: str, stability_df: pd.DataFrame, out_dir: Path):
    if stability_df.empty:
        return

    metrics = ["top_k", "dim", "energy", "w", "HHI", "T", "S", "C_mean", "D", "test_acc"]
    existing = [m for m in metrics if m in stability_df.columns]
    M_raw = stability_df[existing].to_numpy(dtype=float).T

    M = M_raw.copy()
    for i in range(M.shape[0]):
        row = M[i]
        mn = np.nanmin(row)
        mx = np.nanmax(row)
        if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < EPS:
            M[i] = 0.5
        else:
            M[i] = (row - mn) / (mx - mn + EPS)

    plt.figure(figsize=(max(5, 0.7 * len(stability_df)), 6))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.colorbar(label="row-normalized value")
    plt.yticks(range(len(existing)), existing)
    plt.xticks(range(len(stability_df)), stability_df["split"].astype(str).tolist(), rotation=45)
    plt.title(f"F. Split Mechanism Stability Strip - {ds}")
    for i in range(M_raw.shape[0]):
        for j in range(M_raw.shape[1]):
            val = M_raw[i, j]
            txt = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "F_split_mechanism_stability_strip.png", dpi=220)
    plt.close()


def analyze_one_split(graph, dataset: str, split_id: int, seed: int, grid_name: str, out_dir: Path, max_barcode_nodes: int):
    y = graph.y
    train_idx, val_idx, test_idx, split_policy = core.random_split_matching_protocol(graph, seed, prefer_fixed_counts=True)
    classes = np.unique(y[train_idx])
    class_to_pos = {int(c): i for i, c in enumerate(classes)}

    grid = core.grid_from_name(grid_name)
    trace = []

    cfg, pred_selected, info_selected, metrics = core.select_by_validation(
        graph,
        train_idx,
        val_idx,
        test_idx,
        grid,
        feature_variant="full",
        objective="val_acc",
        trace=trace,
    )

    blocks = core.get_feature_variant_blocks(cfg.feature_variant)
    F_full, block_meta = core.build_features(graph.x, graph.adj, blocks)

    sel, fish = core.fisher_select(F_full, y, train_idx, classes, cfg.top_k)
    F = F_full[:, sel]

    subspaces = {}
    pca_dims = {}
    retained = {}
    basis_by_class = {}

    for ci, c in enumerate(classes):
        idx_c = train_idx[y[train_idx] == c]
        mu, B, dk, rv = core.fit_class_subspace(F[idx_c], cfg.dim, cfg.energy)
        subspaces[ci] = (mu, B)
        pca_dims[int(c)] = int(dk)
        retained[int(c)] = float(rv)
        basis_by_class[int(c)] = B

    R_pca = core.pca_residuals(F, subspaces)
    R_pca_n = normalize_scores_by_train(R_pca, train_idx)

    R_ridge = core.multi_ridge_classify(F, y, train_idx, classes, cfg.alphas)
    R_ridge_n = normalize_scores_by_train(R_ridge, train_idx)

    S_final = float(cfg.w) * R_pca_n + (1.0 - float(cfg.w)) * R_ridge_n

    pred_pca = classes[np.argmin(R_pca_n, axis=1)]
    pred_ridge = classes[np.argmin(R_ridge_n, axis=1)]
    pred_final = classes[np.argmin(S_final, axis=1)]

    train_acc = float(np.mean(pred_final[train_idx] == y[train_idx]))
    val_acc = float(np.mean(pred_final[val_idx] == y[val_idx]))
    test_acc = float(np.mean(pred_final[test_idx] == y[test_idx]))

    # ------------------------------------------------------------
    # Fisher block contribution I_b
    # ------------------------------------------------------------
    selected_total = max(1, len(sel))
    block_rows = []
    P_evidence = np.zeros((graph.num_nodes, len(BLOCK_ORDER)), dtype=float)

    meta_map = {name: (start, end) for name, start, end in block_meta}

    for bi, block in enumerate(BLOCK_ORDER):
        if block not in meta_map:
            count = 0
            start, end = 0, 0
            idx_block = np.asarray([], dtype=int)
        else:
            start, end = meta_map[block]
            idx_block = sel[(sel >= start) & (sel < end)]
            count = int(len(idx_block))

        I = count / selected_total

        if len(idx_block) > 0:
            weights = np.maximum(fish[idx_block], 0.0)
            E = np.mean(np.abs(F_full[:, idx_block]) * weights[None, :], axis=1)
        else:
            E = np.zeros(graph.num_nodes, dtype=float)

        P_evidence[:, bi] = E

        block_rows.append({
            "dataset": dataset,
            "split": split_id,
            "block": block,
            "block_label": BLOCK_LABEL[block],
            "I_name": I_NAME[block],
            "selected_count": count,
            "top_k_effective": selected_total,
            "I_b": I,
        })

    P_sum = P_evidence.sum(axis=1, keepdims=True)
    P = P_evidence / (P_sum + EPS)

    I = {I_NAME[BLOCK_ORDER[i]]: block_rows[i]["I_b"] for i in range(len(BLOCK_ORDER))}

    T = (I["I_Pr2X"] + I["I_Ps2X"]) / (I["I_PrX"] + I["I_PsX"] + EPS)
    O = I["I_Pr3X"] / (I["I_PrX"] + I["I_Pr2X"] + EPS)
    S_hub = (I["I_PsX"] + I["I_Ps2X"] + I["I_XminusPsX"]) / (
        I["I_PrX"] + I["I_Pr2X"] + I["I_Pr3X"] + I["I_XminusPrX"] + I["I_PrXminusPr2X"] + EPS
    )

    # Family-size-adjusted signal statistics from Fisher-weighted block evidence
    Raw_i, Low_i, High_i, FamE = compute_family_adjusted_shares_from_evidence(P_evidence)
    R = safe_mean(Raw_i)
    L = safe_mean(Low_i)
    H = safe_mean(High_i)
    HHI = H / (L + EPS)

    # Node-level signal statistics
    P_cols = [I_NAME[b].replace("I_", "P_") for b in BLOCK_ORDER]
    node_rows = []
    deg = np.asarray(graph.adj.sum(axis=1)).ravel()

    HHI_i = High_i / (Low_i + EPS)
    T_i = (P[:, 2] + P[:, 7]) / (P[:, 1] + P[:, 6] + EPS)
    SymRatio_i = (P[:, 6] + P[:, 7] + P[:, 8]) / (P[:, 1] + P[:, 2] + P[:, 3] + P[:, 4] + P[:, 5] + EPS)
    entropy_i = -np.sum(P * np.log(P + EPS), axis=1)
    purity_i = np.max(P, axis=1)
    mechanism_type = infer_node_type(P, Raw_i, Low_i, High_i)

    M_pca, R_pca_true, R_pca_wrong = wrong_min_margin(R_pca_n, y, classes)
    M_ridge, R_ridge_true, R_ridge_wrong = wrong_min_margin(R_ridge_n, y, classes)

    split_name = np.full(graph.num_nodes, "none", dtype=object)
    split_name[train_idx] = "train"
    split_name[val_idx] = "val"
    split_name[test_idx] = "test"

    for i in range(graph.num_nodes):
        row = {
            "dataset": dataset,
            "split_id": split_id,
            "node_index": int(i),
            "original_node_id": int(graph.original_node_ids[i]) if len(graph.original_node_ids) == graph.num_nodes else int(i),
            "split": split_name[i],
            "true_label": int(y[i]),
            "pred_final": int(pred_final[i]),
            "pred_pca": int(pred_pca[i]),
            "pred_ridge": int(pred_ridge[i]),
            "final_correct": int(pred_final[i] == y[i]),
            "pca_correct": int(pred_pca[i] == y[i]),
            "ridge_correct": int(pred_ridge[i] == y[i]),
            "M_pca": float(M_pca[i]),
            "M_ridge": float(M_ridge[i]),
            "R_raw_i": float(Raw_i[i]),
            "L_low_i": float(Low_i[i]),
            "H_high_i": float(High_i[i]),
            "HHI_i": float(HHI_i[i]),
            "T_i": float(T_i[i]),
            "SymRatio_i": float(SymRatio_i[i]),
            "SignalEntropy_i": float(entropy_i[i]),
            "Purity_i": float(purity_i[i]),
            "degree": float(deg[i]),
            "mechanism_type": mechanism_type[i],
        }
        for j, c in enumerate(P_cols):
            row[c] = float(P[i, j])
        node_rows.append(row)

    node_df = pd.DataFrame(node_rows)

    # Class-level signal glyph
    class_rows = []
    for c in classes:
        m = (y == c)
        m_test = m & np.isin(np.arange(graph.num_nodes), test_idx)
        use = m_test if np.any(m_test) else m

        Pc = P[use]
        R_c = safe_mean(Raw_i[use])
        L_c = safe_mean(Low_i[use])
        H_c = safe_mean(High_i[use])
        T_c = safe_mean((Pc[:, 2] + Pc[:, 7]) / (Pc[:, 1] + Pc[:, 6] + EPS))
        S_c = safe_mean((Pc[:, 6] + Pc[:, 7] + Pc[:, 8]) / (Pc[:, 1] + Pc[:, 2] + Pc[:, 3] + Pc[:, 4] + Pc[:, 5] + EPS))
        P_c = np.mean(Pc, axis=0)
        VarSignal_c = safe_mean(np.sum((Pc - P_c[None, :]) ** 2, axis=1))

        idx_test_c = test_idx[y[test_idx] == c]
        acc_c = float(np.mean(pred_final[idx_test_c] == y[idx_test_c])) if len(idx_test_c) else np.nan

        class_rows.append({
            "dataset": dataset,
            "split": split_id,
            "class": int(c),
            "n_nodes_all": int(np.sum(m)),
            "n_nodes_test": int(len(idx_test_c)),
            "R_c": R_c,
            "L_c": L_c,
            "H_c": H_c,
            "T_c": T_c,
            "S_c": S_c,
            "VarSignal_c": VarSignal_c,
            "test_acc_c": acc_c,
        })

    class_df = pd.DataFrame(class_rows)

    # PCA geometry by class
    class_geom_rows = []
    dims = []
    for c in classes:
        dim_c = int(pca_dims[int(c)])
        dims.append(dim_c)
        class_geom_rows.append({
            "dataset": dataset,
            "split": split_id,
            "class": int(c),
            "C_c": dim_c,
            "retained_variance": retained[int(c)],
        })

    class_geom_df = pd.DataFrame(class_geom_rows)
    C_mean = safe_mean(dims)
    C_std = safe_std(dims)

    # Subspace overlap and confusion
    pair_rows = []
    test_true = y[test_idx]
    test_pred = pred_final[test_idx]

    for ia, a in enumerate(classes):
        for ib, b in enumerate(classes):
            if int(a) == int(b):
                continue

            Ba = basis_by_class[int(a)]
            Bb = basis_by_class[int(b)]
            da = Ba.shape[1]
            db = Bb.shape[1]
            if min(da, db) <= 0:
                O_ab = np.nan
            else:
                O_ab = float(np.sum((Ba.T @ Bb) ** 2) / max(1, min(da, db)))

            denom_a = max(1, int(np.sum(test_true == a)))
            Conf_ab = float(np.sum((test_true == a) & (test_pred == b)) / denom_a)

            denom_b = max(1, int(np.sum(test_true == b)))
            Conf_ba = float(np.sum((test_true == b) & (test_pred == a)) / denom_b)

            pair_rows.append({
                "dataset": dataset,
                "split": split_id,
                "class_a": int(a),
                "class_b": int(b),
                "O_ab": O_ab,
                "Conf_ab": Conf_ab,
                "Conf_ba": Conf_ba,
                "ConfPair_ab": Conf_ab + Conf_ba,
            })

    pair_df = pd.DataFrame(pair_rows)

    # PCA-Ridge agreement and quadrants on test
    pca_ok = pred_pca[test_idx] == y[test_idx]
    ridge_ok = pred_ridge[test_idx] == y[test_idx]
    final_ok = pred_final[test_idx] == y[test_idx]

    A_PR = float(np.mean(pred_pca[test_idx] == pred_ridge[test_idx]))
    D_conflict = 1.0 - A_PR

    Q_easy = float(np.mean(pca_ok & ridge_ok))
    Q_geo = float(np.mean(pca_ok & (~ridge_ok)))
    Q_boundary = float(np.mean((~pca_ok) & ridge_ok))
    Q_hard = float(np.mean((~pca_ok) & (~ridge_ok)))

    # Error signal shift on test
    test_node_df = node_df[node_df["split"] == "test"].copy()
    correct_nodes = test_node_df[test_node_df["final_correct"] == 1]
    wrong_nodes = test_node_df[test_node_df["final_correct"] == 0]

    shift_rows = []
    for b in BLOCK_ORDER:
        col = I_NAME[b].replace("I_", "P_")
        pc = safe_mean(correct_nodes[col]) if len(correct_nodes) else np.nan
        pw = safe_mean(wrong_nodes[col]) if len(wrong_nodes) else np.nan
        shift_rows.append({
            "dataset": dataset,
            "split": split_id,
            "level": "block",
            "block": b,
            "block_label": BLOCK_LABEL[b],
            "P_b_correct": pc,
            "P_b_wrong": pw,
            "Delta_wrong_minus_correct": pw - pc if np.isfinite(pc) and np.isfinite(pw) else np.nan,
        })

    family_specs = [
        ("raw", "Raw family", "R_raw_i"),
        ("low", "Low-pass family", "L_low_i"),
        ("high", "High-pass family", "H_high_i"),
    ]
    for fam_key, fam_label, fam_col in family_specs:
        pc = safe_mean(correct_nodes[fam_col]) if len(correct_nodes) else np.nan
        pw = safe_mean(wrong_nodes[fam_col]) if len(wrong_nodes) else np.nan
        shift_rows.append({
            "dataset": dataset,
            "split": split_id,
            "level": "family",
            "family": fam_key,
            "family_label": fam_label,
            "P_b_correct": pc,
            "P_b_wrong": pw,
            "Delta_wrong_minus_correct": pw - pc if np.isfinite(pc) and np.isfinite(pw) else np.nan,
        })
    shift_df = pd.DataFrame(shift_rows)

    # Node type accuracy
    type_rows = []
    for t, g in test_node_df.groupby("mechanism_type"):
        type_rows.append({
            "dataset": dataset,
            "split": split_id,
            "mechanism_type": t,
            "n_test": int(len(g)),
            "test_acc": float(g["final_correct"].mean()),
            "mean_HHI_i": safe_mean(g["HHI_i"]),
            "mean_entropy": safe_mean(g["SignalEntropy_i"]),
            "mean_purity": safe_mean(g["Purity_i"]),
        })
    type_df = pd.DataFrame(type_rows)

    # Hub / error coupling and high-pass geometry coupling
    test_mask_df = node_df["split"] == "test"
    corr_HHI_Mpca = safe_corr(node_df.loc[test_mask_df, "HHI_i"], node_df.loc[test_mask_df, "M_pca"])
    corr_deg_SymRatio = safe_corr(node_df.loc[test_mask_df, "degree"], node_df.loc[test_mask_df, "SymRatio_i"])
    corr_deg_error = safe_corr(node_df.loc[test_mask_df, "degree"], 1 - node_df.loc[test_mask_df, "final_correct"])

    B_boundary = 1.0 - float(cfg.w)

    if H > L and H > R:
        graph_signal_identity = "high-pass-dominant"
    elif L > H and L > R:
        graph_signal_identity = "low-pass-dominant"
    elif R >= L and R >= H:
        graph_signal_identity = "raw-dominant"
    else:
        graph_signal_identity = "mixed"

    if B_boundary > 0.6:
        decision_identity = "boundary-dominant"
    elif B_boundary < 0.4:
        decision_identity = "geometry-dominant"
    else:
        decision_identity = "balanced-geometry-boundary"

    summary = {
        "dataset": dataset,
        "split": split_id,
        "seed": seed,
        "split_policy": split_policy,
        "grid": grid_name,
        "method": "src_v16c",
        "top_k": int(cfg.top_k),
        "dim": int(cfg.dim),
        "energy": float(cfg.energy),
        "alphas_json": json.dumps(list(cfg.alphas)),
        "w": float(cfg.w),
        "B_boundary_dominance": float(B_boundary),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,

        **I,

        "R_raw_dominance": float(R),
        "L_lowpass_dominance": float(L),
        "H_highpass_dominance": float(H),
        "HHI_highpass_heterophily_index": float(HHI),
        "T_twohop_return_index": float(T),
        "O_deephop_utility": float(O),
        "S_hub_sensitivity": float(S_hub),

        "C_mean_class_geometry_complexity": float(C_mean),
        "C_std_class_geometry_imbalance": float(C_std),
        "C_min": float(np.min(dims)) if len(dims) else np.nan,
        "C_max": float(np.max(dims)) if len(dims) else np.nan,

        "A_PR_pca_ridge_agreement": float(A_PR),
        "D_geometry_discriminant_conflict": float(D_conflict),
        "Q_easy": float(Q_easy),
        "Q_geo_only": float(Q_geo),
        "Q_boundary_only": float(Q_boundary),
        "Q_hard": float(Q_hard),

        "SignalEntropy_test_mean": safe_mean(test_node_df["SignalEntropy_i"]),
        "Purity_test_mean": safe_mean(test_node_df["Purity_i"]),
        "SignalEntropy_correct_mean": safe_mean(correct_nodes["SignalEntropy_i"]) if len(correct_nodes) else np.nan,
        "SignalEntropy_wrong_mean": safe_mean(wrong_nodes["SignalEntropy_i"]) if len(wrong_nodes) else np.nan,
        "Purity_correct_mean": safe_mean(correct_nodes["Purity_i"]) if len(correct_nodes) else np.nan,
        "Purity_wrong_mean": safe_mean(wrong_nodes["Purity_i"]) if len(wrong_nodes) else np.nan,

        "Corr_HHI_Mpca": corr_HHI_Mpca,
        "Corr_deg_SymRatio": corr_deg_SymRatio,
        "Corr_deg_error": corr_deg_error,

        "graph_signal_identity": graph_signal_identity,
        "decision_mechanism_identity": decision_identity,
    }

    stability_row = {
        "dataset": dataset,
        "split": split_id,
        "seed": seed,
        "top_k": int(cfg.top_k),
        "dim": int(cfg.dim),
        "energy": float(cfg.energy),
        "w": float(cfg.w),
        "HHI": float(HHI),
        "T": float(T),
        "S": float(S_hub),
        "C_mean": float(C_mean),
        "D": float(D_conflict),
        "test_acc": float(test_acc),
    }

    # Save split-level files
    split_dir = ensure_dir(out_dir / f"split_{split_id}")
    pd.DataFrame(trace).to_csv(split_dir / "validation_trace.csv", index=False)
    pd.DataFrame([summary]).to_csv(split_dir / "whitebox_summary_split.csv", index=False)
    pd.DataFrame(block_rows).to_csv(split_dir / "graph_signal_block_fingerprint.csv", index=False)
    node_df.to_csv(split_dir / "node_graph_signal_barcode.csv", index=False)
    class_df.to_csv(split_dir / "class_signal_glyph.csv", index=False)
    class_geom_df.to_csv(split_dir / "class_geometry_complexity.csv", index=False)
    pair_df.to_csv(split_dir / "class_subspace_overlap_and_confusion.csv", index=False)
    shift_df.to_csv(split_dir / "error_signal_shift.csv", index=False)
    type_df.to_csv(split_dir / "node_mechanism_type_accuracy.csv", index=False)

    # For split 0, draw A-E
    if split_id == 0:
        plot_A_barcode(dataset, node_df, out_dir, max_nodes=max_barcode_nodes)
        plot_B_class_glyph(dataset, class_df, out_dir)
        plot_C_geometry_boundary(dataset, node_df, out_dir)
        plot_D_constellation(dataset, class_geom_df, pair_df, out_dir)
        plot_E_error_shift(dataset, shift_df, out_dir)

        # Also copy key CSVs to top-level for convenience
        pd.DataFrame(block_rows).to_csv(out_dir / "A_graph_signal_block_fingerprint.csv", index=False)
        node_df.to_csv(out_dir / "A_graph_signal_barcode_nodes.csv", index=False)
        class_df.to_csv(out_dir / "B_class_signal_glyph.csv", index=False)
        pd.concat([class_geom_df], ignore_index=True).to_csv(out_dir / "D_class_geometry_complexity.csv", index=False)
        pair_df.to_csv(out_dir / "D_subspace_overlap_constellation.csv", index=False)
        shift_df.to_csv(out_dir / "E_error_signal_shift.csv", index=False)
        type_df.to_csv(out_dir / "node_mechanism_type_accuracy.csv", index=False)

    return summary, stability_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--grid", default="default", choices=["default", "fast"])
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=20260419)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-barcode-nodes", type=int, default=900)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    graph = core.load_dataset(Path(args.data_root), args.dataset)
    dataset = core.canonical_dataset_name(args.dataset)

    summaries = []
    stability = []

    for split in range(args.splits):
        seed = int(args.seed0) + split
        print(f"========== {dataset} split={split} seed={seed} grid={args.grid} ==========", flush=True)
        s, st = analyze_one_split(
            graph=graph,
            dataset=dataset,
            split_id=split,
            seed=seed,
            grid_name=args.grid,
            out_dir=out_dir,
            max_barcode_nodes=args.max_barcode_nodes,
        )
        summaries.append(s)
        stability.append(st)

    summary_df = pd.DataFrame(summaries)
    stability_df = pd.DataFrame(stability)

    summary_df.to_csv(out_dir / "whitebox_dataset_summary_by_split.csv", index=False)
    stability_df.to_csv(out_dir / "F_split_mechanism_stability_strip.csv", index=False)

    # Aggregate over splits. With splits=1, std values are 0.
    agg_rows = []
    numeric_cols = [c for c in summary_df.columns if pd.api.types.is_numeric_dtype(summary_df[c])]
    row = {
        "dataset": dataset,
        "grid": args.grid,
        "splits": int(args.splits),
    }
    for c in numeric_cols:
        row[c + "_mean"] = safe_mean(summary_df[c])
        row[c + "_std"] = safe_std(summary_df[c])
    for c in ["graph_signal_identity", "decision_mechanism_identity"]:
        if c in summary_df.columns:
            row[c] = summary_df[c].iloc[0]
    agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "whitebox_dataset_summary_aggregated.csv", index=False)

    plot_F_stability(dataset, stability_df, out_dir)

    # Text report
    r0 = summary_df.iloc[0].to_dict()
    report = []
    report.append(f"White-box Graph Mechanism Atlas Report")
    report.append(f"Dataset: {dataset}")
    report.append(f"Grid: {args.grid}")
    report.append(f"Splits: {args.splits}")
    report.append("")
    report.append("1. Graph Signal Identity")
    report.append(f"R = {r0.get('R_raw_dominance', np.nan):.4f}")
    report.append(f"L = {r0.get('L_lowpass_dominance', np.nan):.4f}")
    report.append(f"H = {r0.get('H_highpass_dominance', np.nan):.4f}")
    report.append(f"HHI = {r0.get('HHI_highpass_heterophily_index', np.nan):.4f}")
    report.append(f"T = {r0.get('T_twohop_return_index', np.nan):.4f}")
    report.append(f"O = {r0.get('O_deephop_utility', np.nan):.4f}")
    report.append(f"S = {r0.get('S_hub_sensitivity', np.nan):.4f}")
    report.append(f"Graph signal identity: {r0.get('graph_signal_identity')}")
    report.append("")
    report.append("2. Class Geometry Identity")
    report.append(f"C_mean = {r0.get('C_mean_class_geometry_complexity', np.nan):.4f}")
    report.append(f"C_std = {r0.get('C_std_class_geometry_imbalance', np.nan):.4f}")
    report.append("")
    report.append("3. Decision Mechanism Identity")
    report.append(f"A_PR = {r0.get('A_PR_pca_ridge_agreement', np.nan):.4f}")
    report.append(f"D = {r0.get('D_geometry_discriminant_conflict', np.nan):.4f}")
    report.append(f"Q_easy = {r0.get('Q_easy', np.nan):.4f}")
    report.append(f"Q_geo = {r0.get('Q_geo_only', np.nan):.4f}")
    report.append(f"Q_boundary = {r0.get('Q_boundary_only', np.nan):.4f}")
    report.append(f"Q_hard = {r0.get('Q_hard', np.nan):.4f}")
    report.append(f"B = {r0.get('B_boundary_dominance', np.nan):.4f}")
    report.append(f"Decision identity: {r0.get('decision_mechanism_identity')}")
    report.append("")
    report.append("4. Output Figures")
    report.append("A_graph_signal_barcode.png")
    report.append("B_class_signal_glyph.png")
    report.append("C_geometry_boundary_mechanism_plane.png")
    report.append("D_subspace_constellation.png")
    report.append("E_error_signal_shift.png")
    report.append("F_split_mechanism_stability_strip.png")
    report.append("")
    report.append("Note:")
    report.append("When splits=1, F is a single-split mechanism strip. True split stability requires splits > 1.")

    (out_dir / "WHITEBOX_MECHANISM_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    print("===== DONE =====")
    print("Output:", out_dir)
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
