#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for paper experiments of src_v16c.

Protocol guarantees
-------------------
1. Feature selection uses train nodes only.
2. PCA subspaces use train nodes only.
3. Ridge closed-form classifier uses train nodes only.
4. Hyper-parameters are selected by validation accuracy/objective only.
5. Test nodes are evaluated exactly once after selection; test accuracy is never
   used to select a configuration.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import pickle
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


# ============================================================
# IO helpers
# ============================================================

def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path | str, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Dataset discovery / loading
# ============================================================

ALIASES = {
    "amazon-photo": ["Photo", "photo", "amazon-photo", "amazon_photo", "AmazonPhoto"],
    "photo": ["Photo", "photo", "amazon-photo", "amazon_photo", "AmazonPhoto"],
    "amazon-computers": ["Computers", "computers", "amazon-computers", "amazon_computers", "AmazonComputers"],
    "computers": ["Computers", "computers", "amazon-computers", "amazon_computers", "AmazonComputers"],
    "cs": ["CS", "cs", "CoauthorCS"],
    "physics": ["Physics", "physics", "CoauthorPhysics"],
    "chameleon": ["chameleon"],
    "squirrel": ["squirrel"],
    "cornell": ["cornell"],
    "texas": ["texas"],
    "wisconsin": ["wisconsin"],
    "actor": ["actor", "Actor"],
    "cora": ["cora"],
    "citeseer": ["citeseer"],
    "pubmed": ["pubmed"],
}


def canonical_dataset_name(name: str) -> str:
    s = name.lower().replace("_", "-")
    if s in ("amazon-photo", "photo"):
        return "amazon-photo"
    if s in ("amazon-computers", "computers"):
        return "amazon-computers"
    return s


def discover_data_root(start: Optional[Path | str] = None) -> Path:
    candidates: List[Path] = []
    if start is not None:
        st = Path(start).resolve()
        candidates += [st, st / "planetoid" / "data", st.parent / "planetoid" / "data", st.parent.parent / "planetoid" / "data"]
        candidates += [p / "planetoid" / "data" for p in [st] + list(st.parents)]
    candidates += [
        Path("D:/桌面/MSR实验复现与创新/planetoid/data"),
        Path("D:/MSR实验复现与创新/planetoid/data"),
        Path.cwd() / "planetoid" / "data",
        Path.cwd().parent / "planetoid" / "data",
    ]
    for c in candidates:
        try:
            if c.exists() and (any(c.glob("ind.cora.x")) or any(c.glob("*/raw")) or any(c.glob("*"))):
                return c.resolve()
        except Exception:
            pass
    raise FileNotFoundError("Cannot locate planetoid/data root. Pass --data-root explicitly.")


def find_dataset_dir(data_root: Path | str, dataset: str) -> Optional[Path]:
    root = Path(data_root)
    aliases = ALIASES.get(canonical_dataset_name(dataset), [dataset])
    for a in aliases:
        p = root / a
        if p.exists():
            return p
    # fallback case-insensitive
    low_aliases = {a.lower() for a in aliases}
    for p in root.iterdir():
        if p.name.lower() in low_aliases:
            return p
    return None


@dataclass
class GraphData:
    dataset: str
    x: np.ndarray
    y: np.ndarray
    adj: sparse.csr_matrix
    original_node_ids: np.ndarray
    raw_root: str

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(len(np.unique(self.y)))


def _make_undirected_adj(n: int, rows: np.ndarray, cols: np.ndarray) -> sparse.csr_matrix:
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n), dtype=np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def load_geom_gcn_raw(raw_root: Path, dataset: str) -> GraphData:
    node_file = raw_root / "out1_node_feature_label.txt"
    edge_file = raw_root / "out1_graph_edges.txt"
    if not node_file.exists() or not edge_file.exists():
        raise FileNotFoundError(f"Missing Geom-GCN raw files in {raw_root}")
    ids, feats, labels = [], [], []
    with open(node_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            nid, feat, lab = line.rstrip("\n").split("\t")
            ids.append(int(nid))
            feats.append(np.asarray(feat.split(","), dtype=np.float64))
            labels.append(int(lab))
    order = np.argsort(ids)
    x = np.vstack(feats)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    original = np.asarray(ids, dtype=np.int64)[order]
    src, dst = [], []
    with open(edge_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b = line.strip().split("\t")
            src.append(int(a)); dst.append(int(b))
    adj = _make_undirected_adj(x.shape[0], np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64))
    return GraphData(dataset, x, y, adj, original, str(raw_root))


def load_npz_graph(npz_path: Path, dataset: str) -> GraphData:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)
    # Common PyG saved raw npz formats.
    if "features" in keys:
        x = z["features"]
    elif "x" in keys:
        x = z["x"]
    elif "node_features" in keys:
        x = z["node_features"]
    elif "attr_data" in keys and "attr_indices" in keys and "attr_indptr" in keys:
        x = sparse.csr_matrix((z["attr_data"], z["attr_indices"], z["attr_indptr"]), shape=tuple(z["attr_shape"])).toarray()
    else:
        # choose the largest 2D array as features
        arrays = [(k, z[k]) for k in z.files if isinstance(z[k], np.ndarray) and z[k].ndim == 2]
        if not arrays:
            raise ValueError(f"Cannot infer features in {npz_path}; keys={z.files}")
        k, x = max(arrays, key=lambda kv: kv[1].size)
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)

    if "labels" in keys:
        y = z["labels"]
    elif "y" in keys:
        y = z["y"]
    elif "node_labels" in keys:
        y = z["node_labels"]
    else:
        cand = [(k, z[k]) for k in z.files if isinstance(z[k], np.ndarray) and z[k].ndim == 1 and len(z[k]) == x.shape[0]]
        if not cand:
            raise ValueError(f"Cannot infer labels in {npz_path}; keys={z.files}")
        k, y = cand[-1]
    y = np.asarray(y).reshape(-1)
    if y.dtype.kind not in "iu":
        _, y = np.unique(y, return_inverse=True)
    y = y.astype(np.int64)

    n = x.shape[0]
    if "edge_index" in keys:
        ei = z["edge_index"]
        if ei.shape[0] == 2:
            rows, cols = ei[0], ei[1]
        else:
            rows, cols = ei[:, 0], ei[:, 1]
        adj = _make_undirected_adj(n, rows.astype(np.int64), cols.astype(np.int64))
    elif "adj_data" in keys and "adj_indices" in keys and "adj_indptr" in keys:
        adj = sparse.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"])).astype(np.float64)
        adj = ((adj + adj.T) > 0).astype(np.float64).tocsr()
        adj.setdiag(0); adj.eliminate_zeros()
    elif "edges" in keys:
        e = z["edges"]
        adj = _make_undirected_adj(n, e[:, 0].astype(np.int64), e[:, 1].astype(np.int64))
    else:
        raise ValueError(f"Cannot infer graph edges in {npz_path}; keys={z.files}")
    return GraphData(dataset, x, y, adj, np.arange(n, dtype=np.int64), str(npz_path.parent))


def parse_index_file(path: Path) -> List[int]:
    return [int(line.strip()) for line in open(path, "r", encoding="utf-8")]


def load_planetoid_ind(data_root: Path, dataset: str) -> GraphData:
    # Standard Planetoid ind.dataset.* files.
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objs = []
    for name in names:
        with open(data_root / f"ind.{dataset}.{name}", "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx = parse_index_file(data_root / f"ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx)
    if dataset == "citeseer":
        # Fix isolated nodes in Citeseer, standard Planetoid handling.
        test_idx_range_full = np.arange(min(test_idx), max(test_idx) + 1)
        tx_ext = sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_ext[test_idx_range - min(test_idx_range), :] = tx
        ty_ext = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_ext[test_idx_range - min(test_idx_range), :] = ty
        tx, ty = tx_ext, ty_ext
    features = sparse.vstack((allx, tx)).tolil()
    features[test_idx_range, :] = features[test_idx, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_range, :] = labels[test_idx, :]
    x_arr = features.toarray().astype(np.float64)
    y_arr = np.argmax(labels, axis=1).astype(np.int64)
    rows, cols = [], []
    for i, neigh in graph.items():
        for j in neigh:
            rows.append(i); cols.append(j)
    adj = _make_undirected_adj(x_arr.shape[0], np.asarray(rows), np.asarray(cols))
    return GraphData(dataset, x_arr, y_arr, adj, np.arange(x_arr.shape[0], dtype=np.int64), str(data_root))


def load_dataset(data_root: Path | str, dataset: str) -> GraphData:
    root = Path(data_root)
    canon = canonical_dataset_name(dataset)
    if canon in ["cora", "citeseer", "pubmed"] and (root / f"ind.{canon}.x").exists():
        return load_planetoid_ind(root, canon)
    ddir = find_dataset_dir(root, canon)
    if ddir is None:
        raise FileNotFoundError(f"Cannot find dataset {dataset} under {root}")
    raw = ddir / "geom_gcn" / "raw"
    if raw.exists() and (raw / "out1_node_feature_label.txt").exists():
        return load_geom_gcn_raw(raw, canon)
    raw2 = ddir / "raw"
    if raw2.exists() and (raw2 / "out1_node_feature_label.txt").exists():
        return load_geom_gcn_raw(raw2, canon)
    npzs = list((ddir / "raw").glob("*.npz")) if (ddir / "raw").exists() else list(ddir.glob("*.npz"))
    if npzs:
        return load_npz_graph(npzs[0], canon)
    raise FileNotFoundError(f"No supported raw files for {dataset} in {ddir}")


# ============================================================
# Splits
# ============================================================

def find_geom_split_file(raw_root: str | Path, dataset: str, split_id: int) -> Optional[Path]:
    raw = Path(raw_root)
    candidates = [
        raw / f"{dataset}_split_0.6_0.2_{split_id}.npz",
        raw / f"{dataset.replace('-', '_')}_split_0.6_0.2_{split_id}.npz",
        raw / f"{dataset.replace('-', '')}_split_0.6_0.2_{split_id}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(raw.glob(f"*_split_0.6_0.2_{split_id}.npz"))
    return hits[0] if hits else None


def load_fixed_split(graph: GraphData, split_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = find_geom_split_file(graph.raw_root, graph.dataset, split_id)
    if p is None:
        raise FileNotFoundError(f"No fixed split {split_id} for {graph.dataset} in {graph.raw_root}")
    z = np.load(p)
    return (np.where(z["train_mask"].astype(bool))[0], np.where(z["val_mask"].astype(bool))[0], np.where(z["test_mask"].astype(bool))[0])


def has_fixed_splits(graph: GraphData) -> bool:
    return find_geom_split_file(graph.raw_root, graph.dataset, 0) is not None


def stratified_split_by_counts(y: np.ndarray, seed: int, train_counts: Dict[int, int], val_counts: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        tr = min(int(train_counts.get(int(c), 0)), max(0, n - 2))
        va = min(int(val_counts.get(int(c), 0)), max(0, n - tr - 1))
        train.extend(idx[:tr]); val.extend(idx[tr:tr + va]); test.extend(idx[tr + va:])
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64), np.asarray(test, dtype=np.int64)


def counts_from_fixed_split(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    train_counts = {int(c): int(np.sum(y[train_idx] == c)) for c in np.unique(y)}
    val_counts = {int(c): int(np.sum(y[val_idx] == c)) for c in np.unique(y)}
    return train_counts, val_counts


def class_balanced_split(y: np.ndarray, seed: int, train_per_class: int = 20, val_per_class: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    train_counts = {}
    val_counts = {}
    for c in classes:
        n = int(np.sum(y == c))
        tr = min(train_per_class, max(1, n - 2))
        va = min(val_per_class, max(1, n - tr - 1))
        train_counts[int(c)] = tr
        val_counts[int(c)] = va
    return stratified_split_by_counts(y, seed, train_counts, val_counts)


def random_split_matching_protocol(graph: GraphData, seed: int, prefer_fixed_counts: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if prefer_fixed_counts and has_fixed_splits(graph):
        tr0, va0, _ = load_fixed_split(graph, 0)
        trc, vac = counts_from_fixed_split(graph.y, tr0, va0)
        tr, va, te = stratified_split_by_counts(graph.y, seed, trc, vac)
        return tr, va, te, "random_stratified_matching_fixed_split0_class_counts"
    tr, va, te = class_balanced_split(graph.y, seed, 20, 30)
    return tr, va, te, "class_balanced_random_20_train_30_val_per_class"


def fewshot_split(y: np.ndarray, train_per_class: int, seed: int, val_per_class: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    train_counts = {}
    val_counts = {}
    for c in classes:
        n = int(np.sum(y == c))
        tr = min(int(train_per_class), max(1, n - 2))
        # Keep val fixed if possible, but never consume all remaining samples.
        va = min(int(val_per_class), max(1, n - tr - 1))
        train_counts[int(c)] = tr
        val_counts[int(c)] = va
    return stratified_split_by_counts(y, seed, train_counts, val_counts)


# ============================================================
# Feature engineering and src_v16c model
# ============================================================

def row_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X / np.maximum(norm, eps)


def row_normalize(A: sparse.csr_matrix) -> sparse.csr_matrix:
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    return sparse.diags(inv) @ A


def sym_normalize(A: sparse.csr_matrix) -> sparse.csr_matrix:
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D = sparse.diags(inv)
    return D @ A @ D


FEATURE_BLOCKS = [
    "raw", "row_1hop", "row_2hop", "row_3hop", "highpass_row_1", "highpass_row_2", "sym_1hop", "sym_2hop", "highpass_sym_1"
]


def build_feature_blocks(x: np.ndarray, adj: sparse.csr_matrix) -> Dict[str, np.ndarray]:
    P_row = row_normalize(adj)
    P_sym = sym_normalize(adj)
    PXr = np.asarray(P_row @ x)
    P2Xr = np.asarray(P_row @ PXr)
    P3Xr = np.asarray(P_row @ P2Xr)
    PXs = np.asarray(P_sym @ x)
    P2Xs = np.asarray(P_sym @ PXs)
    return {
        "raw": row_l2_normalize(x),
        "row_1hop": row_l2_normalize(PXr),
        "row_2hop": row_l2_normalize(P2Xr),
        "row_3hop": row_l2_normalize(P3Xr),
        "highpass_row_1": row_l2_normalize(x - PXr),
        "highpass_row_2": row_l2_normalize(PXr - P2Xr),
        "sym_1hop": row_l2_normalize(PXs),
        "sym_2hop": row_l2_normalize(P2Xs),
        "highpass_sym_1": row_l2_normalize(x - PXs),
    }


def build_features(x: np.ndarray, adj: sparse.csr_matrix, block_names: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    blocks = build_feature_blocks(x, adj)
    if block_names is None:
        block_names = FEATURE_BLOCKS
    parts = []
    meta = []
    start = 0
    for name in block_names:
        arr = blocks[name]
        parts.append(arr)
        end = start + arr.shape[1]
        meta.append((name, start, end))
        start = end
    return np.hstack(parts), meta


def fisher_scores(F: np.ndarray, y: np.ndarray, train_idx: np.ndarray, classes: np.ndarray) -> np.ndarray:
    gm = np.mean(F[train_idx], axis=0)
    vb = np.zeros(F.shape[1], dtype=np.float64)
    vw = np.zeros(F.shape[1], dtype=np.float64)
    for c in classes:
        idx = train_idx[y[train_idx] == c]
        Xc = F[idx]
        mu = np.mean(Xc, axis=0)
        vb += len(idx) * (mu - gm) ** 2
        vw += np.sum((Xc - mu) ** 2, axis=0)
    return vb / (vw + 1e-12)


def fisher_select(F: np.ndarray, y: np.ndarray, train_idx: np.ndarray, classes: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    score = fisher_scores(F, y, train_idx, classes)
    top_k = min(int(top_k), F.shape[1])
    sel = np.argsort(score)[::-1][:top_k]
    return sel, score


def fit_class_subspace(Fc: np.ndarray, max_dim: int, energy_thresh: float) -> Tuple[np.ndarray, np.ndarray, int, float]:
    mu = np.mean(Fc, axis=0)
    C = Fc - mu
    d_max = min(int(max_dim), max(0, Fc.shape[0] - 1), Fc.shape[1])
    if d_max <= 0:
        return mu, np.empty((Fc.shape[1], 0)), 0, 0.0
    G = C @ C.T
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    pos = evals[order] > 1e-12
    evals_pos = evals[order][pos]
    if evals_pos.size == 0:
        return mu, np.empty((Fc.shape[1], 0)), 0, 0.0
    cum = np.cumsum(evals_pos) / np.sum(evals_pos)
    n_keep = min(d_max, max(1, int(np.searchsorted(cum, float(energy_thresh)) + 1)))
    keep = order[pos][:n_keep]
    B = C.T @ (evecs[:, keep] / np.sqrt(evals[keep])[None, :])
    Q, _ = np.linalg.qr(B)
    retained = float(np.sum(evals[keep]) / np.sum(evals_pos))
    return mu, np.asarray(Q[:, :n_keep], dtype=np.float64), int(n_keep), retained


def pca_residuals(F: np.ndarray, subspaces: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    R = np.zeros((F.shape[0], len(subspaces)), dtype=np.float64)
    for ci in range(len(subspaces)):
        mu, B = subspaces[ci]
        C = F - mu
        total = np.sum(C * C, axis=1)
        if B.shape[1] > 0:
            proj = np.sum((C @ B) ** 2, axis=1)
        else:
            proj = np.zeros(F.shape[0])
        R[:, ci] = np.maximum(total - proj, 0.0)
    return R


def multi_ridge_classify(F: np.ndarray, y: np.ndarray, train_idx: np.ndarray, classes: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    Ftr = F[train_idx]
    ntr = Ftr.shape[0]
    Y = np.zeros((ntr, len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Y[y[train_idx] == c, i] = 1.0
    GG = Ftr @ Ftr.T
    scores_sum = np.zeros((F.shape[0], len(classes)), dtype=np.float64)
    for alpha in alphas:
        beta = np.linalg.solve(GG + float(alpha) * np.eye(ntr), Y)
        s = F @ (Ftr.T @ beta)
        scores_sum += s / (np.std(s[train_idx]) + 1e-12)
    # Return residual-like score, lower better.
    return -(scores_sum / len(alphas))


@dataclass
class Src16cConfig:
    top_k: int = 6000
    dim: int = 64
    energy: float = 0.95
    alphas: Tuple[float, ...] = (0.01, 0.1, 1.0)
    w: float = 0.5
    feature_variant: str = "full"


def get_feature_variant_blocks(variant: str) -> List[str]:
    if variant == "full":
        return list(FEATURE_BLOCKS)
    if variant == "no_p3":
        return [b for b in FEATURE_BLOCKS if b != "row_3hop"]
    if variant == "no_sym":
        return [b for b in FEATURE_BLOCKS if not b.startswith("sym") and b != "highpass_sym_1"]
    if variant == "no_highpass":
        return [b for b in FEATURE_BLOCKS if not b.startswith("highpass")]
    if variant == "raw_only":
        return ["raw"]
    if variant == "row_lowpass_only":
        return ["raw", "row_1hop", "row_2hop", "row_3hop"]
    raise ValueError(f"Unknown feature variant {variant}")


def fit_predict_src16c(
    F_full: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    classes: np.ndarray,
    cfg: Src16cConfig,
    preselected: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if preselected is None:
        sel, fish = fisher_select(F_full, y, train_idx, classes, cfg.top_k)
    else:
        sel, fish = preselected
    F = F_full[:, sel]
    subspaces: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    pca_dims = {}
    retained = {}
    for ci, c in enumerate(classes):
        idx_c = train_idx[y[train_idx] == c]
        mu, B, dk, rv = fit_class_subspace(F[idx_c], cfg.dim, cfg.energy)
        subspaces[ci] = (mu, B)
        pca_dims[int(c)] = int(dk)
        retained[int(c)] = float(rv)
    R_pca = pca_residuals(F, subspaces)
    R_pca_n = R_pca / (np.std(R_pca[train_idx]) + 1e-12)
    R_ridge = multi_ridge_classify(F, y, train_idx, classes, cfg.alphas)
    R_ridge_n = R_ridge / (np.std(R_ridge[train_idx]) + 1e-12)
    S = float(cfg.w) * R_pca_n + (1.0 - float(cfg.w)) * R_ridge_n
    pred = classes[np.argmin(S, axis=1)]
    info = {
        "scores": S,
        "selector": sel,
        "fisher_scores": fish,
        "pca_dims": pca_dims,
        "retained_variance": retained,
        "stored_coefficients_proxy": int(len(sel) * sum(pca_dims.values()) + len(train_idx) * len(classes) * len(cfg.alphas)),
        "epoch_trainable_params": 0,
    }
    return pred, info


def grid_from_name(name: str = "default") -> Dict[str, Any]:
    if name == "fast":
        return {
            "top_k": [4000, 6000],
            "dim": [64, 96],
            "energy": [0.95, 0.99],
            "alphas": [(0.01, 0.1, 1.0), (0.05, 0.5, 5.0)],
            "w": [0.4, 0.5, 0.6],
        }
    return {
        "top_k": [4000, 5000, 6000, 8000],
        "dim": [32, 48, 64, 96],
        "energy": [0.90, 0.95, 0.99],
        "alphas": [(0.01, 0.1, 1.0), (0.05, 0.5, 5.0), (0.1, 1.0, 10.0)],
        "w": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    }


def iter_grid(grid: Dict[str, Sequence[Any]], feature_variant: str = "full") -> Iterable[Src16cConfig]:
    for top_k in grid["top_k"]:
        for dim in grid["dim"]:
            for energy in grid["energy"]:
                for alphas in grid["alphas"]:
                    for w in grid["w"]:
                        yield Src16cConfig(int(top_k), int(dim), float(energy), tuple(float(a) for a in alphas), float(w), feature_variant)


def select_by_validation(
    graph: GraphData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    grid: Dict[str, Sequence[Any]],
    feature_variant: str = "full",
    objective: str = "val_acc",
    trace: Optional[List[Dict[str, Any]]] = None,
    fixed_cfg: Optional[Src16cConfig] = None,
) -> Tuple[Src16cConfig, np.ndarray, Dict[str, Any], Dict[str, float]]:
    classes = np.unique(graph.y[train_idx])
    blocks = get_feature_variant_blocks(feature_variant)
    F_full, block_meta = build_features(graph.x, graph.adj, blocks)
    best = None
    configs = [fixed_cfg] if fixed_cfg is not None else list(iter_grid(grid, feature_variant))
    for cfg in configs:
        assert cfg is not None
        pred, info = fit_predict_src16c(F_full, graph.y, train_idx, classes, cfg)
        val_acc = float(np.mean(pred[val_idx] == graph.y[val_idx]))
        test_acc = float(np.mean(pred[test_idx] == graph.y[test_idx]))
        train_acc = float(np.mean(pred[train_idx] == graph.y[train_idx]))
        bal = balanced_acc(graph.y[val_idx], pred[val_idx], classes)
        score = val_acc if objective == "val_acc" else (val_acc + 0.1 * bal)
        row = {**asdict(cfg), "alphas_json": json.dumps(list(cfg.alphas)), "train_acc": train_acc, "val_acc": val_acc, "test_acc_shadow_not_for_selection": test_acc, "val_bal_acc": bal, "objective": score, "num_selected_features": len(info["selector"])}
        if trace is not None:
            trace.append(row)
        if best is None or score > best[0]:
            best = (score, cfg, pred, info, {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "val_bal_acc": bal, "num_features_after_variant": F_full.shape[1]})
    assert best is not None
    _, cfg, pred, info, metrics = best
    info["feature_blocks"] = block_meta
    return cfg, pred, info, metrics


def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    vals = []
    for c in classes:
        m = y_true == c
        if np.any(m):
            vals.append(np.mean(y_pred[m] == c))
    return float(np.mean(vals)) if vals else 0.0


# ============================================================
# Simple non-epoch baselines for sample-efficiency / sanity
# ============================================================

def fit_predict_ridge_raw(graph: GraphData, train_idx: np.ndarray, classes: np.ndarray, alpha: float = 1.0, use_multihop: bool = False) -> np.ndarray:
    if use_multihop:
        F, _ = build_features(graph.x, graph.adj, ["raw", "row_1hop", "row_2hop", "highpass_row_1", "highpass_row_2"])
    else:
        F = row_l2_normalize(graph.x)
    scores = multi_ridge_classify(F, graph.y, train_idx, classes, [alpha])
    return classes[np.argmin(scores, axis=1)]


def simple_baseline_grid(graph: GraphData, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> List[Dict[str, Any]]:
    classes = np.unique(graph.y[train_idx])
    rows = []
    for method, use_multihop in [("ridge_raw", False), ("ridge_multihop5", True)]:
        best = None
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            pred = fit_predict_ridge_raw(graph, train_idx, classes, alpha, use_multihop)
            va = float(np.mean(pred[val_idx] == graph.y[val_idx]))
            te = float(np.mean(pred[test_idx] == graph.y[test_idx]))
            if best is None or va > best[0]:
                best = (va, te, alpha)
        rows.append({"method": method, "val_acc": best[0], "test_acc": best[1], "alpha": best[2], "selection_rule": "alpha_selected_by_val_only_no_test"})
    return rows


# ============================================================
# Efficiency measurement
# ============================================================

def measure_run(func, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    proc = psutil.Process(os.getpid()) if psutil is not None else None
    rss_before = proc.memory_info().rss if proc is not None else None
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss if proc is not None else None
    return result, {
        "time_sec": float(elapsed),
        "python_peak_memory_mb": float(peak / 1024 / 1024),
        "rss_before_mb": float(rss_before / 1024 / 1024) if rss_before is not None else "",
        "rss_after_mb": float(rss_after / 1024 / 1024) if rss_after is not None else "",
    }

