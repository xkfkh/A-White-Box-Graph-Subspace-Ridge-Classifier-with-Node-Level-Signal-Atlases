# -*- coding: utf-8 -*-
"""
run_whitebox_hasdc.py

HA-SDC experiment: dual-channel (smooth + raw) gated scoring for heterophily graphs.

5 variants:
  1. smooth_plain_residual   - smooth channel, plain residual (existing baseline)
  2. smooth_dynamic_layer3   - smooth channel, suppression (existing dynamic_layer3)
  3. raw_plain_residual      - raw channel, plain residual (new)
  4. raw_dynamic_layer3      - raw channel, suppression (new)
  5. hasdc_gated             - dual-channel gated fusion (core new method)

All hyperparameter selection uses val_acc only, never test_acc.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from fair_utils import (
    DEFAULT_DATASETS, HETEROPHILY_DATASETS, accuracy_from_pred,
    build_dense_laplacian, get_device, load_extended_dataset,
    normalize_dataset_name, parse_csv_list, parse_float_grid,
    parse_int_grid, parse_int_list, set_seed, summarize_csv,
    tikhonov_smooth_sparse_cg, write_csv_rows,
)
from whitebox_src_v5_bridge import (
    build_subspaces_src, build_subspaces_raw_src,
    score_layer3_src, score_plain_residual_src,
    score_hasdc_src, score_freq_decomposed_src,
    setup_src_v5, tikhonov_smooth_src,
)

HASDC_VARIANTS = [
    "smooth_plain_residual",
    "smooth_dynamic_layer3",
    "raw_plain_residual",
    "raw_dynamic_layer3",
    "hasdc_channel_select",
    "hasdc_merged",
    "hasdc_dual_rank",
    "freq_decomposed",
]

# Variants that use the hasdc dual-channel system
HASDC_DUAL_VARIANTS = {"hasdc_channel_select", "hasdc_merged", "hasdc_dual_rank"}

# Map variant -> fusion_strategy
FUSION_MAP = {
    "hasdc_channel_select": "channel_select",
    "hasdc_merged": "merged_subspace",
    "hasdc_dual_rank": "dual_rank",
}

# Default heterophily datasets for quick testing
HETERO_DATASETS = ["cornell", "texas", "wisconsin", "actor", "chameleon", "squirrel"]


def smooth_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    lam: float,
    backend: str,
    dense_node_limit: int,
    cg_max_iter: int,
    cg_tol: float,
) -> Tuple[torch.Tensor, str, dict]:
    if lam <= 0:
        return x.clone(), "none_lam0", {"cg_iters": 0, "cg_final_max_rel_resid": 0.0, "cg_converged": True}
    n = x.shape[0]
    if backend == "src" or (backend == "auto" and n <= dense_node_limit):
        L = build_dense_laplacian(edge_index, n, x.device)
        return tikhonov_smooth_src(x, L, lam), "src_v5_dense_layer1", {"cg_iters": 0, "cg_final_max_rel_resid": 0.0, "cg_converged": True}
    z, info = tikhonov_smooth_sparse_cg(
        x, edge_index=edge_index, lam=lam, max_iter=cg_max_iter, tol=cg_tol, return_info=True,
    )
    return z, "sparse_cg_equivalent_layer1", info


def select_best_by_val_only(rows: List[dict]) -> dict:
    """Val-only model selection; tie-break by smaller sub_dim, lam, eta."""
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise ValueError("No successful rows to select from.")

    def key(r: dict):
        val = r.get("val_acc", float("nan"))
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = -float("inf")
        return (float(val), -int(r.get("sub_dim", 10**9)), -float(r.get("lam", 10**9)), -float(r.get("eta", 10**9)))

    best = max(ok, key=key)
    out = dict(best)
    out["selection_rule"] = "val_acc_only"
    return out


def effective_repeats_for_dataset(ds: str, repeats: List[int]) -> List[int]:
    return list(repeats)


def run_one(
    dataset_name: str,
    data_root: str,
    repeat: int,
    explicit_split_id,
    lam: float,
    sub_dim: int,
    eta: float,
    tau: float,
    variant: str,
    device: torch.device,
    feature_norm: str,
    train_per_class: int,
    val_per_class: int,
    smooth_backend: str,
    dense_node_limit: int,
    cg_max_iter: int,
    cg_tol: float,
) -> dict:
    if variant not in HASDC_VARIANTS:
        raise ValueError(f"Unknown variant={variant}; valid: {HASDC_VARIANTS}")

    t0 = time.time()
    g = load_extended_dataset(
        dataset_name, data_root=data_root, repeat=repeat,
        explicit_split_id=explicit_split_id,
        train_per_class=train_per_class, val_per_class=val_per_class,
        feature_norm=feature_norm, make_undirected=True,
    )

    x = g.x.to(device)
    y = g.y.to(device)
    edge_index = g.edge_index.to(device)

    # Raw features (always available)
    X_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    train_list = g.train_idx.cpu().tolist()
    val_list = g.val_idx.cpu().tolist()

    # Smooth features
    needs_smooth = variant in ("smooth_plain_residual", "smooth_dynamic_layer3", "freq_decomposed") or variant in HASDC_DUAL_VARIANTS
    if needs_smooth and lam > 0:
        Z, smooth_used, cg_info = smooth_features(
            x, edge_index, lam, smooth_backend, dense_node_limit, cg_max_iter, cg_tol,
        )
        Z_cpu = Z.detach().cpu()
        actual_lam = lam
    else:
        Z_cpu = X_cpu.clone()
        actual_lam = 0.0
        smooth_used = "none"
        cg_info = {"cg_iters": 0, "cg_final_max_rel_resid": 0.0, "cg_converged": True}

    # Build subspaces as needed
    sub_s = None
    sub_r = None
    needs_sub_s = variant in ("smooth_plain_residual", "smooth_dynamic_layer3") or variant in HASDC_DUAL_VARIANTS
    needs_sub_r = variant in ("raw_plain_residual", "raw_dynamic_layer3") or variant in HASDC_DUAL_VARIANTS

    if needs_sub_s:
        sub_s = build_subspaces_src(Z_cpu, y_cpu, train_list, g.num_classes, sub_dim=sub_dim)
    if needs_sub_r:
        sub_r = build_subspaces_raw_src(X_cpu, y_cpu, train_list, g.num_classes, sub_dim=sub_dim)

    # Scoring
    scoring_used = variant
    g_c_info = {}

    if variant == "smooth_plain_residual":
        R_cpu = score_plain_residual_src(Z_cpu, sub_s, g.num_classes)
        pred = R_cpu.argmin(dim=1).to(device)

    elif variant == "smooth_dynamic_layer3":
        R_cpu = score_layer3_src(Z_cpu, sub_s, g.num_classes, eta=eta)
        pred = R_cpu.argmin(dim=1).to(device)

    elif variant == "raw_plain_residual":
        R_cpu = score_plain_residual_src(X_cpu, sub_r, g.num_classes)
        pred = R_cpu.argmin(dim=1).to(device)

    elif variant == "raw_dynamic_layer3":
        R_cpu = score_layer3_src(X_cpu, sub_r, g.num_classes, eta=eta)
        pred = R_cpu.argmin(dim=1).to(device)

    elif variant in HASDC_DUAL_VARIANTS:
        inner_mode = "suppression" if eta > 0 else "plain_residual"
        fusion = FUSION_MAP[variant]
        R_cpu, diag = score_hasdc_src(
            Z_cpu, X_cpu, y_cpu, train_list, val_list,
            sub_s, sub_r, g.num_classes,
            scoring_mode=inner_mode, eta=eta, tau=tau,
            fusion_strategy=fusion, merged_dim=sub_dim,
        )
        pred = R_cpu.argmin(dim=1).to(device)
        g_c_info = diag.get("g_c", {})
        scoring_used = f"{variant}_{inner_mode}"

    elif variant == "freq_decomposed":
        inner_mode = "suppression" if eta > 0 else "plain_residual"
        R_cpu, diag = score_freq_decomposed_src(
            Z_cpu, X_cpu, y_cpu, train_list, val_list,
            g.num_classes, sub_dim=sub_dim,
            scoring_mode=inner_mode, eta=eta,
        )
        pred = R_cpu.argmin(dim=1).to(device)
        scoring_used = f"freq_decomposed_{inner_mode}_{diag.get('selected_channel', '?')}"

    elapsed = time.time() - t0
    result = {
        "dataset": g.name,
        "repeat": repeat,
        "split_id": g.split_id,
        "seed": g.seed,
        "split_policy": g.split_policy,
        "method": "WhiteBox-HASDC",
        "variant": variant,
        "lam": actual_lam,
        "sub_dim": sub_dim,
        "eta": eta,
        "tau": tau,
        "smooth_backend": smooth_used,
        "scoring_backend": scoring_used,
        "num_nodes": g.num_nodes,
        "num_features": g.num_features,
        "num_classes": g.num_classes,
        "train_size": int(g.train_idx.numel()),
        "val_size": int(g.val_idx.numel()),
        "test_size": int(g.test_idx.numel()),
        "train_acc": accuracy_from_pred(pred, y, g.train_idx.to(device)),
        "val_acc": accuracy_from_pred(pred, y, g.val_idx.to(device)),
        "test_acc": accuracy_from_pred(pred, y, g.test_idx.to(device)),
        "cg_iters": cg_info.get("cg_iters", 0),
        "cg_converged": cg_info.get("cg_converged", True),
        "time_sec": round(elapsed, 3),
    }
    # Add gating info for hasdc
    if g_c_info:
        for c, gc in g_c_info.items():
            result[f"g_{c}"] = round(gc, 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="HA-SDC experiment")
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--datasets", type=str, default=",".join(HETERO_DATASETS))
    ap.add_argument("--out_dir", type=str, default="results_hasdc")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--repeats", type=str, default="0,1,2,3,4")
    ap.add_argument("--explicit_split_id", type=int, default=None)
    ap.add_argument("--variants", type=str, default=",".join(HASDC_VARIANTS))
    ap.add_argument("--lam_grid", type=str, default="0.5,1,3,5,7,10")
    ap.add_argument("--sub_dim_grid", type=str, default="12")
    ap.add_argument("--eta_grid", type=str, default="0.05,0.10,0.20")
    ap.add_argument("--tau_grid", type=str, default="3.0,5.0,10.0")
    ap.add_argument("--feature_norm", type=str, default="row_l1")
    ap.add_argument("--train_per_class", type=int, default=20)
    ap.add_argument("--val_per_class", type=int, default=30)
    ap.add_argument("--smooth_backend", type=str, default="auto")
    ap.add_argument("--dense_node_limit", type=int, default=6000)
    ap.add_argument("--cg_max_iter", type=int, default=500)
    ap.add_argument("--cg_tol", type=float, default=1e-5)
    args = ap.parse_args()

    setup_src_v5(args.src)
    datasets = [normalize_dataset_name(x) for x in parse_csv_list(args.datasets, HETERO_DATASETS)]
    repeats = parse_int_list(args.repeats, [0, 1, 2, 3, 4])
    variants = parse_csv_list(args.variants, HASDC_VARIANTS)
    lam_grid = parse_float_grid(args.lam_grid, [0.5, 1, 3, 5, 7, 10])
    sub_dim_grid = parse_int_grid(args.sub_dim_grid, [12])
    eta_grid = parse_float_grid(args.eta_grid, [0.05, 0.10, 0.20])
    tau_grid = parse_float_grid(args.tau_grid, [3.0, 5.0, 10.0])
    device = get_device(args.device)
    out = Path(args.out_dir)

    rows_all: List[dict] = []
    best_rows: List[dict] = []

    for ds in datasets:
        for repeat in effective_repeats_for_dataset(ds, repeats):
            set_seed(repeat)
            for variant in variants:
                grid_rows: List[dict] = []

                # Determine grids per variant
                if variant in ("raw_plain_residual", "raw_dynamic_layer3"):
                    local_lams = [0.0]  # no smoothing needed
                else:
                    local_lams = list(lam_grid)

                if variant in ("smooth_plain_residual", "raw_plain_residual"):
                    local_etas = [0.0]  # no suppression
                else:
                    local_etas = list(eta_grid)

                if variant in HASDC_DUAL_VARIANTS or variant == "freq_decomposed":
                    local_taus = list(tau_grid)
                else:
                    local_taus = [5.0]  # tau irrelevant for single-channel

                for sub_dim in sub_dim_grid:
                    for lam in local_lams:
                        for eta in local_etas:
                            for tau in local_taus:
                                tag = f"ds={ds} r={repeat} v={variant} lam={lam} sd={sub_dim} eta={eta} tau={tau}"
                                print(f"\n[HASDC] {tag}")
                                try:
                                    row = run_one(
                                        dataset_name=ds, data_root=args.data_root,
                                        repeat=repeat, explicit_split_id=args.explicit_split_id,
                                        lam=lam, sub_dim=sub_dim, eta=eta, tau=tau,
                                        variant=variant, device=device,
                                        feature_norm=args.feature_norm,
                                        train_per_class=args.train_per_class,
                                        val_per_class=args.val_per_class,
                                        smooth_backend=args.smooth_backend,
                                        dense_node_limit=args.dense_node_limit,
                                        cg_max_iter=args.cg_max_iter,
                                        cg_tol=args.cg_tol,
                                    )
                                    print(f"  val={row['val_acc']:.4f} test={row['test_acc']:.4f} t={row['time_sec']}s")
                                except Exception as e:
                                    row = {
                                        "dataset": ds, "repeat": repeat,
                                        "method": "WhiteBox-HASDC", "variant": variant,
                                        "lam": lam, "sub_dim": sub_dim, "eta": eta, "tau": tau,
                                        "error": repr(e), "val_acc": float("nan"), "test_acc": float("nan"),
                                    }
                                    print(f"  ERROR: {e!r}")
                                rows_all.append(row)
                                grid_rows.append(row)
                                write_csv_rows(out / "hasdc_all_grid_partial.csv", rows_all)

                try:
                    best_rows.append(select_best_by_val_only(grid_rows))
                except Exception as e:
                    best_rows.append({
                        "dataset": ds, "repeat": repeat, "variant": variant,
                        "error": f"selection_failed: {e!r}",
                    })

    all_path = out / "hasdc_all_grid.csv"
    best_path = out / "hasdc_best_by_val.csv"
    write_csv_rows(all_path, rows_all)
    write_csv_rows(best_path, best_rows)
    summarize_csv(best_path, out / "hasdc_summary.csv", group_cols=["dataset", "variant"])
    print(f"\nDone. Results: {best_path}")


if __name__ == "__main__":
    main()

