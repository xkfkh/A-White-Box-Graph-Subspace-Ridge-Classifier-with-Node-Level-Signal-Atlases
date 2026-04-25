#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E05 ablation study for SRC-v16c under the SAME split protocol as the final main SRC-v16c table.

Key differences from the old scripts/run_E05_ablation.py:
1. The 'full' line is intentionally excluded by default because it has already been run.
2. Splits are aligned to the main SRC-v16c final protocol:
   - chameleon: fixed Geom-GCN split_id = repeat, repeat = 0..9.
   - cornell/texas/wisconsin: random stratified split matching fixed split0 class counts,
     seed = 20260419 + repeat.
   - amazon-photo/amazon-computers: class-balanced random split, 20 train + 30 val per class,
     seed = 20260419 + repeat.
3. Hyper-parameter/configuration selection is validation-only.
4. Test accuracy is computed exactly once for the validation-selected configuration.
5. Split indices are saved for auditability.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sys

# This file is intended to live in <repo_root>/scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from paperexp.core import (  # noqa: E402
    GraphData,
    Src16cConfig,
    balanced_acc,
    build_features,
    canonical_dataset_name,
    discover_data_root,
    ensure_dir,
    fit_predict_src16c,
    get_feature_variant_blocks,
    grid_from_name,
    has_fixed_splits,
    iter_grid,
    load_dataset,
    load_fixed_split,
    random_split_matching_protocol,
    write_csv,
)

DEFAULT_DATASETS = [
    "chameleon",
    "cornell",
    "texas",
    "wisconsin",
    "amazon-photo",
    "amazon-computers",
]

# Original run_E05_ablation.py default ablations minus 'full'.
DEFAULT_ABLATIONS_NO_FULL = [
    "no_p3",
    "no_sym",
    "no_highpass",
    "pca_only",
    "ridge_only",
    "single_alpha",
    "fixed_dim64_energy095",
    "raw_only",
    "row_lowpass_only",
]

# Ablations implemented in the old ablation_grid() but not included in the old default list.
EXTRA_ABLATIONS_NO_FULL = [
    "no_energy_fixed_dim",
    "small_grid_v16c_core",
]


def parse_repeat_list(text: str) -> List[int]:
    """Parse '0,1,2' or '0 1 2' into [0, 1, 2]."""
    items: List[int] = []
    for part in str(text).replace(",", " ").split():
        if part.strip():
            items.append(int(part.strip()))
    if not items:
        raise ValueError("repeat list is empty")
    return items


def ablation_grid(base_grid: Dict[str, Sequence[Any]], ablation: str) -> Tuple[Dict[str, List[Any]], str]:
    """Same ablation definitions as the old E05 script, except callers exclude full by default."""
    g: Dict[str, List[Any]] = {k: list(v) for k, v in base_grid.items()}
    variant = "full"
    if ablation == "full":
        pass
    elif ablation in ["no_p3", "no_sym", "no_highpass", "raw_only", "row_lowpass_only"]:
        variant = ablation
    elif ablation == "pca_only":
        g["w"] = [1.0]
    elif ablation == "ridge_only":
        g["w"] = [0.0]
    elif ablation == "single_alpha":
        g["alphas"] = [(1.0,)]
    elif ablation == "fixed_dim64_energy095":
        g["dim"] = [64]
        g["energy"] = [0.95]
    elif ablation == "no_energy_fixed_dim":
        # energy=1.0 approximates using max dim without early truncation.
        g["energy"] = [1.0]
    elif ablation == "small_grid_v16c_core":
        g["top_k"] = [6000]
        g["dim"] = [64]
        g["energy"] = [0.95]
        g["alphas"] = [(0.01, 0.1, 1.0)]
        g["w"] = [0.5]
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    return g, variant


def validate_split(graph: GraphData, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Fail fast if a split overlaps or does not cover all nodes exactly once."""
    n = graph.num_nodes
    train = set(map(int, train_idx))
    val = set(map(int, val_idx))
    test = set(map(int, test_idx))

    if train & val:
        raise RuntimeError(f"train/val overlap: {len(train & val)} nodes")
    if train & test:
        raise RuntimeError(f"train/test overlap: {len(train & test)} nodes")
    if val & test:
        raise RuntimeError(f"val/test overlap: {len(val & test)} nodes")

    union = train | val | test
    if len(union) != n:
        raise RuntimeError(f"split does not cover all nodes: covered={len(union)} total={n}")

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("empty train/val/test split")

    train_classes = set(map(int, np.unique(graph.y[train_idx])))
    all_classes = set(map(int, np.unique(graph.y)))
    if train_classes != all_classes:
        raise RuntimeError(f"train split misses classes: train={sorted(train_classes)} all={sorted(all_classes)}")


def main_protocol_split(
    graph: GraphData,
    dataset: str,
    repeat: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return the exact split protocol used by the final SRC-v16c main table."""
    ds = canonical_dataset_name(dataset)

    if ds == "squirrel":
        raise ValueError("squirrel is intentionally excluded")

    if ds == "chameleon":
        # Main table uses fixed Geom-GCN split_id = repeat for chameleon.
        train_idx, val_idx, test_idx = load_fixed_split(graph, int(repeat))
        meta = {
            "repeat": int(repeat),
            "split_id": int(repeat),
            "split_seed": int(repeat),
            "split_policy": f"fixed_geom_gcn_split_id_{int(repeat)}",
        }
    else:
        # Main table random split seed: 20260419 + repeat.
        # For WebKB with fixed splits, this matches fixed split0 class counts.
        # For Amazon without fixed splits, this falls back to 20 train + 30 val per class.
        seed = int(base_seed) + int(repeat)
        train_idx, val_idx, test_idx, policy = random_split_matching_protocol(
            graph,
            seed,
            prefer_fixed_counts=True,
        )
        meta = {
            "repeat": int(repeat),
            "split_id": "",
            "split_seed": int(seed),
            "split_policy": str(policy),
        }

    validate_split(graph, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx, meta


def select_by_validation_strict(
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
    """
    Validation-only selector.

    Unlike the old core.select_by_validation(), this function does NOT compute test accuracy
    for every candidate. It computes test accuracy only once, after the best validation
    configuration has already been selected.
    """
    classes = np.unique(graph.y[train_idx])
    blocks = get_feature_variant_blocks(feature_variant)
    F_full, block_meta = build_features(graph.x, graph.adj, blocks)
    configs: Iterable[Src16cConfig]
    configs = [fixed_cfg] if fixed_cfg is not None else iter_grid(grid, feature_variant)

    best: Optional[Tuple[float, Src16cConfig, np.ndarray, Dict[str, Any], Dict[str, float]]] = None

    for cfg in configs:
        assert cfg is not None
        pred, info = fit_predict_src16c(F_full, graph.y, train_idx, classes, cfg)
        train_acc = float(np.mean(pred[train_idx] == graph.y[train_idx]))
        val_acc = float(np.mean(pred[val_idx] == graph.y[val_idx]))
        val_bal = balanced_acc(graph.y[val_idx], pred[val_idx], classes)
        score = val_acc if objective == "val_acc" else (val_acc + 0.1 * val_bal)

        row = {
            **asdict(cfg),
            "alphas_json": json.dumps(list(cfg.alphas)),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_bal_acc": val_bal,
            "objective": score,
            "num_selected_features": len(info["selector"]),
            "candidate_note": "test_not_evaluated_for_candidates",
        }
        if trace is not None:
            trace.append(row)

        if best is None or score > best[0]:
            best = (
                float(score),
                cfg,
                pred,
                info,
                {
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_bal_acc": val_bal,
                    "num_features_after_variant": int(F_full.shape[1]),
                },
            )

    assert best is not None
    _, cfg, pred, info, metrics = best

    # Test is evaluated exactly once after the validation-selected configuration is fixed.
    metrics["test_acc"] = float(np.mean(pred[test_idx] == graph.y[test_idx]))
    info["feature_blocks"] = block_meta
    return cfg, pred, info, metrics


def split_index_rows(
    dataset: str,
    repeat: int,
    split_meta: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        for node_idx in idx:
            rows.append({
                "dataset": dataset,
                "repeat": int(repeat),
                "split_id": split_meta["split_id"],
                "split_seed": split_meta["split_seed"],
                "split_policy": split_meta["split_policy"],
                "split_part": part,
                "node_idx": int(node_idx),
            })
    return rows


def summarize_rows(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        write_csv(out_dir / "E05_ablation_summary.csv", [])
        return
    summary = (
        df.groupby(["dataset", "ablation"])
        .agg(
            n=("test_acc", "size"),
            train_mean=("train_acc", "mean"),
            train_std=("train_acc", "std"),
            val_mean=("val_acc", "mean"),
            val_std=("val_acc", "std"),
            test_mean=("test_acc", "mean"),
            test_std=("test_acc", "std"),
            train_size=("train_size", "first"),
            val_size=("val_size", "first"),
            test_size=("test_size", "first"),
            split_policy=("split_policy", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    for col in ["train_mean", "train_std", "val_mean", "val_std", "test_mean", "test_std"]:
        summary[col + "_%"] = summary[col] * 100.0
    summary.to_csv(out_dir / "E05_ablation_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--repeats", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--base-seed", type=int, default=20260419)
    p.add_argument("--grid", choices=["default", "fast"], default="fast")
    p.add_argument("--ablations", nargs="+", default=DEFAULT_ABLATIONS_NO_FULL)
    p.add_argument("--include-extra-ablations", action="store_true", help="Also run no_energy_fixed_dim and small_grid_v16c_core.")
    p.add_argument("--out-dir", default="results/E05_ablation_same_splits_no_full")
    p.add_argument("--objective", choices=["val_acc", "val_acc_plus_bal"], default="val_acc")
    p.add_argument("--write-split-indices", action="store_true", default=True)
    p.add_argument("--no-write-split-indices", dest="write_split_indices", action="store_false")
    args = p.parse_args()

    root = discover_data_root(args.data_root)
    out = ensure_dir(args.out_dir)
    repeats = parse_repeat_list(args.repeats)
    base_grid = grid_from_name(args.grid)

    ablations = [a for a in args.ablations if a != "full"]
    if args.include_extra_ablations:
        for a in EXTRA_ABLATIONS_NO_FULL:
            if a not in ablations:
                ablations.append(a)
    if not ablations:
        raise ValueError("No ablations to run after removing 'full'.")

    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        ds = canonical_dataset_name(dataset)
        if ds == "squirrel":
            print("[skip] squirrel is excluded")
            continue

        print(f"\n=== E05 same-split no-full dataset={ds} ===")
        graph = load_dataset(root, ds)

        for repeat in repeats:
            train_idx, val_idx, test_idx, split_meta = main_protocol_split(
                graph=graph,
                dataset=ds,
                repeat=int(repeat),
                base_seed=int(args.base_seed),
            )
            print(
                f"[split] dataset={ds} repeat={repeat} policy={split_meta['split_policy']} "
                f"split_id={split_meta['split_id']} seed={split_meta['split_seed']} "
                f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
            )

            if args.write_split_indices:
                split_rows.extend(split_index_rows(ds, int(repeat), split_meta, train_idx, val_idx, test_idx))

            for ab in ablations:
                t0 = time.perf_counter()
                g, variant = ablation_grid(base_grid, ab)
                trace: List[Dict[str, Any]] = []
                cfg, pred, info, metrics = select_by_validation_strict(
                    graph,
                    train_idx,
                    val_idx,
                    test_idx,
                    g,
                    feature_variant=variant,
                    objective=("val_acc" if args.objective == "val_acc" else "val_acc_plus_bal"),
                    trace=trace,
                )
                elapsed = time.perf_counter() - t0

                row = {
                    "experiment": "E05",
                    "dataset": ds,
                    "repeat": int(repeat),
                    "split": int(repeat),
                    "split_id": split_meta["split_id"],
                    "split_seed": split_meta["split_seed"],
                    "split_policy": split_meta["split_policy"],
                    "ablation": ab,
                    "method": "src_v16c_ablation",
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "test_size": int(len(test_idx)),
                    **asdict(cfg),
                    "alphas_json": json.dumps(list(cfg.alphas)),
                    **metrics,
                    "pca_dims_json": json.dumps(info["pca_dims"], ensure_ascii=False),
                    "retained_variance_json": json.dumps(info.get("retained_variance", {}), ensure_ascii=False),
                    "time_sec": float(elapsed),
                    "selection_rule": "ablation_grid_selected_by_val_only_test_once_after_selection_no_full",
                }
                rows.append(row)

                for cand in trace:
                    cand.update({
                        "dataset": ds,
                        "repeat": int(repeat),
                        "split": int(repeat),
                        "split_id": split_meta["split_id"],
                        "split_seed": split_meta["split_seed"],
                        "split_policy": split_meta["split_policy"],
                        "ablation": ab,
                    })
                traces.extend(trace)

                print(
                    f"repeat={repeat} {ab:24s} "
                    f"val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f} "
                    f"cfg=(top_k={cfg.top_k}, dim={cfg.dim}, energy={cfg.energy}, "
                    f"alphas={cfg.alphas}, w={cfg.w}) time={elapsed:.2f}s"
                )

                # Partial writes after every ablation, so interrupted runs are auditable.
                write_csv(out / "E05_ablation_rows_partial.csv", rows)
                write_csv(out / "E05_ablation_candidate_trace_partial.csv", traces)
                if args.write_split_indices:
                    write_csv(out / "E05_split_indices_partial.csv", split_rows)

    write_csv(out / "E05_ablation_rows.csv", rows)
    write_csv(out / "E05_ablation_candidate_trace.csv", traces)
    if args.write_split_indices:
        write_csv(out / "E05_split_indices.csv", split_rows)
    summarize_rows(rows, out)

    print(f"\nSaved E05 same-split no-full ablation to {out}")


if __name__ == "__main__":
    main()
