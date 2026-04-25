#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src_v16c: Enhanced Multihop Features + Multi-Ridge Ensemble
============================================================
Key changes vs v16/v16b:
  1. Extended multihop: add P^3X, symmetric-normalized features (D^{-1/2}AD^{-1/2})
  2. Multi-alpha Ridge ensemble: average Ridge predictions across multiple alphas
  3. Per-class adaptive dim with energy threshold (keep dims explaining 95% variance)
  4. Simplified fusion: just search w in [0.1..0.9], no reliability map overhead

This targets the feature quality bottleneck rather than the fusion mechanism.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy import sparse

VERSION = "src_v16c"

def discover_project_root(start):
    start = Path(start).resolve()
    for p in [start] + list(start.parents):
        if any((p/d).exists() for d in ["src_v10","src_v14","src_v15","src_v16","scripts"]):
            return p
    raise FileNotFoundError(f"Cannot locate project root from {start}")

def discover_data_base(start, dataset="chameleon"):
    for c in [start/"planetoid"/"data"/dataset, start.parent/"planetoid"/"data"/dataset,
              start.parent.parent/"planetoid"/"data"/dataset,
              Path("D:/桌面/MSR实验复现与创新/planetoid/data")/dataset,
              Path("D:/MSR实验复现与创新/planetoid/data")/dataset]:
        if c.resolve().exists(): return c.resolve()
    raise FileNotFoundError(f"Cannot find data for {dataset}")

def find_raw_root(base):
    hits = list(Path(base).rglob("out1_node_feature_label.txt"))
    if not hits: raise FileNotFoundError(f"No raw data under {base}")
    return hits[0].parent

def load_chameleon_raw(raw_root):
    raw_root = Path(raw_root)
    ids, feats, labels = [], [], []
    with open(raw_root/"out1_node_feature_label.txt","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            nid, feat, label = line.rstrip("\n").split("\t")
            ids.append(int(nid))
            feats.append(np.array(feat.split(","), dtype=np.float64))
            labels.append(int(label))
    order = np.argsort(ids)
    X = np.vstack(feats)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    src, dst = [], []
    with open(raw_root/"out1_graph_edges.txt","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            a, b = line.strip().split("\t")
            src.append(int(a)); dst.append(int(b))
    n = X.shape[0]
    A = sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(dst))),
                           shape=(n,n), dtype=np.float64)
    A.setdiag(0); A.eliminate_zeros()
    A_sym = ((A + A.T) > 0).astype(np.float64).tocsr()
    A_sym.setdiag(0); A_sym.eliminate_zeros()
    return X, y, A_sym

def load_split(raw_root, repeat):
    z = np.load(Path(raw_root)/f"chameleon_split_0.6_0.2_{repeat}.npz")
    return (np.where(z["train_mask"].astype(bool))[0],
            np.where(z["val_mask"].astype(bool))[0],
            np.where(z["test_mask"].astype(bool))[0])

# ============================================================
# Feature engineering - ENHANCED
# ============================================================
def row_l2_normalize(X, eps=1e-12):
    norm = np.sqrt(np.sum(X*X, axis=1, keepdims=True))
    return X / np.maximum(norm, eps)

def row_normalize(A):
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.zeros_like(deg); inv[deg>0] = 1.0/deg[deg>0]
    return sparse.diags(inv) @ A

def sym_normalize(A):
    """D^{-1/2} A D^{-1/2}"""
    A = A.tocsr().astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg>0] = 1.0 / np.sqrt(deg[deg>0])
    D = sparse.diags(inv_sqrt)
    return D @ A @ D

def build_enhanced_features(X, A_sym):
    """Extended multihop with both row-norm and sym-norm propagation."""
    P_row = row_normalize(A_sym)
    P_sym = sym_normalize(A_sym)

    # Row-normalized hops
    PX_r = np.asarray(P_row @ X)
    P2X_r = np.asarray(P_row @ PX_r)
    P3X_r = np.asarray(P_row @ P2X_r)

    # Sym-normalized hops
    PX_s = np.asarray(P_sym @ X)
    P2X_s = np.asarray(P_sym @ PX_s)

    blocks = [
        row_l2_normalize(X),           # raw
        row_l2_normalize(PX_r),        # 1-hop row
        row_l2_normalize(P2X_r),       # 2-hop row
        row_l2_normalize(P3X_r),       # 3-hop row
        row_l2_normalize(X - PX_r),    # high-pass 1
        row_l2_normalize(PX_r - P2X_r),# high-pass 2
        row_l2_normalize(PX_s),        # 1-hop sym
        row_l2_normalize(P2X_s),       # 2-hop sym
        row_l2_normalize(X - PX_s),    # high-pass sym
    ]
    return np.hstack(blocks)

def fisher_select(F, y, train_idx, classes, top_k):
    d = F.shape[1]
    gm = np.mean(F[train_idx], axis=0)
    vb = np.zeros(d); vw = np.zeros(d)
    for c in classes:
        idx = train_idx[y[train_idx]==c]; Xc = F[idx]; mu = np.mean(Xc,0)
        vb += len(idx)*(mu-gm)**2; vw += np.sum((Xc-mu)**2, axis=0)
    return np.argsort(vb/(vw+1e-12))[::-1][:top_k]

# ============================================================
# PCA subspace with energy-based dim
# ============================================================
def fit_class_subspace(F_c, max_dim, energy_thresh=0.95):
    """PCA with energy-based dim selection: keep dims explaining energy_thresh of variance."""
    mu = np.mean(F_c, axis=0)
    C = F_c - mu
    d_max = min(max_dim, max(0, len(F_c)-1), F_c.shape[1])
    if d_max == 0:
        return mu, np.empty((F_c.shape[1],0))
    G = C @ C.T
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals_sorted = evals[order]
    pos_mask = evals_sorted > 1e-12
    evals_pos = evals_sorted[pos_mask]
    if len(evals_pos) == 0:
        return mu, np.empty((F_c.shape[1],0))
    cumvar = np.cumsum(evals_pos) / np.sum(evals_pos)
    n_keep = min(d_max, max(1, int(np.searchsorted(cumvar, energy_thresh) + 1)))
    keep = order[pos_mask][:n_keep]
    B = C.T @ (evecs[:,keep] / np.sqrt(evals[keep])[None,:])
    return mu, B

def pca_residuals(F, subspaces, nc):
    n = F.shape[0]
    R = np.zeros((n, nc))
    for ci in range(nc):
        mu, B = subspaces[ci]
        C = F - mu
        total = np.sum(C*C, axis=1)
        if B.shape[1] > 0:
            proj = np.sum((C@B)**2, axis=1)
            R[:, ci] = np.maximum(total - proj, 0.0)
        else:
            R[:, ci] = total
    return R

# ============================================================
# Multi-alpha Ridge ensemble
# ============================================================
def multi_ridge_classify(F, y, train_idx, classes, alphas):
    """Average Ridge predictions across multiple alpha values."""
    Ftr = F[train_idx]; ntr = Ftr.shape[0]; nc = len(classes)
    Y = np.zeros((ntr, nc))
    for i,c in enumerate(classes): Y[y[train_idx]==c, i] = 1.0
    GG = Ftr @ Ftr.T  # shared Gram matrix

    scores_sum = np.zeros((F.shape[0], nc))
    for alpha in alphas:
        G = GG + alpha * np.eye(ntr)
        beta = np.linalg.solve(G, Y)
        scores = F @ (Ftr.T @ beta)
        # Normalize each alpha's scores before averaging
        scores_n = scores / (np.std(scores[train_idx]) + 1e-12)
        scores_sum += scores_n
    return -(scores_sum / len(alphas))

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chameleon")
    parser.add_argument("--num_splits", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== {VERSION}: Enhanced Multihop + Multi-Ridge ===")

    proj_root = discover_project_root(Path(__file__).parent)
    data_base = discover_data_base(proj_root, args.dataset)
    raw_root = find_raw_root(data_base)
    out_dir = proj_root / "scripts" / f"results_{VERSION}_audit_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y, A_sym = load_chameleon_raw(raw_root)
    n = X_raw.shape[0]
    classes = np.unique(y)
    nc = len(classes)

    # Build enhanced features
    F_full = build_enhanced_features(X_raw, A_sym)
    print(f"Nodes={n}, Enhanced features={F_full.shape[1]}, Classes={nc}")

    # Hyperparameter grid
    topk_grid = [4000, 5000, 6000, 8000]
    dim_grid = [32, 48, 64, 96]
    energy_grid = [0.90, 0.95, 0.99]
    ridge_alpha_sets = [
        [0.01, 0.1, 1.0],
        [0.05, 0.5, 5.0],
        [0.1, 1.0, 10.0],
    ]
    w_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    all_results = []
    for split in range(args.num_splits):
        train_idx, val_idx, test_idx = load_split(raw_root, split)
        print(f"\n--- Split {split} (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}) ---")

        best_val = -1; best_test = -1; best_cfg = None
        n_tried = 0

        for top_k in topk_grid:
            sel = fisher_select(F_full, y, train_idx, classes, top_k)
            F = F_full[:, sel]

            for dim in dim_grid:
                for energy in energy_grid:
                    # Build PCA subspaces
                    subspaces = {}
                    for ci,c in enumerate(classes):
                        idx_c = train_idx[y[train_idx]==c]
                        mu, B = fit_class_subspace(F[idx_c], dim, energy)
                        subspaces[ci] = (mu, B)
                    R_pca = pca_residuals(F, subspaces, nc)
                    R_pca_n = R_pca / (np.std(R_pca[train_idx]) + 1e-12)

                    for alphas in ridge_alpha_sets:
                        R_ridge = multi_ridge_classify(F, y, train_idx, classes, alphas)
                        R_ridge_n = R_ridge / (np.std(R_ridge[train_idx]) + 1e-12)

                        for w in w_grid:
                            S = w * R_pca_n + (1-w) * R_ridge_n
                            pred = classes[np.argmin(S, axis=1)]
                            va = float(np.mean(pred[val_idx]==y[val_idx]))
                            n_tried += 1
                            if va > best_val:
                                ta = float(np.mean(pred[test_idx]==y[test_idx]))
                                best_val = va; best_test = ta
                                best_cfg = {"top_k": top_k, "dim": dim, "energy": energy,
                                            "alphas": alphas, "w": w}

        all_results.append({"split": split, "val": best_val, "test": best_test, "cfg": best_cfg})
        print(f"  val={best_val:.4f}  test={best_test:.4f}  tried={n_tried}")
        print(f"  cfg: top_k={best_cfg['top_k']} dim={best_cfg['dim']} energy={best_cfg['energy']} "
              f"alphas={best_cfg['alphas']} w={best_cfg['w']}")

    vals = [r["val"] for r in all_results]
    tests = [r["test"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"val_mean  = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(f"test_mean = {np.mean(tests):.4f} +/- {np.std(tests):.4f}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    summary = {"version": VERSION, "dataset": args.dataset,
               "val_mean": float(np.mean(vals)), "val_std": float(np.std(vals)),
               "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
               "elapsed": time.time()-t0, "splits": all_results}
    with open(out_dir/f"run_summary_{VERSION}.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    with open(out_dir/f"split_summary_{VERSION}.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","val","test","top_k","dim","energy","w"])
        for r in all_results:
            w.writerow([r["split"], f'{r["val"]:.6f}', f'{r["test"]:.6f}',
                        r["cfg"]["top_k"], r["cfg"]["dim"], r["cfg"]["energy"], r["cfg"]["w"]])
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()


