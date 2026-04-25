#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill src_v16c main-table results to match baseline repeats/splits.

Place this file inside src_v16c_paper_experiments/ and run from that directory.

Protocol:
  - no test leakage
  - Fisher selection uses train only
  - PCA/Ridge fit train only
  - hyper-parameters selected by validation only
  - test accuracy is logged only after selection
  - each run is appended immediately to CSV
  - completed dataset/split/repeat rows are skipped on resume

Typical commands:
  python run_src_v16c_fill_main_table_10repeats.py --data-root "D:\\桌面\\MSR实验复现与创新\\planetoid\\data"
  python run_src_v16c_fill_main_table_10repeats.py --include-actor --data-root "D:\\桌面\\MSR实验复现与创新\\planetoid\\data"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from paperexp.core import (  # noqa: E402
    discover_data_root,
    load_dataset,
    has_fixed_splits,
    load_fixed_split,
    random_split_matching_protocol,
    class_balanced_split,
    stratified_split_by_counts,
    counts_from_fixed_split,
    grid_from_name,
    select_by_validation,
)

DEFAULT_DATASETS_7 = [
    "chameleon", "squirrel", "cornell", "texas", "wisconsin", "amazon-photo", "amazon-computers"
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_csv(path: Path, row: Dict):
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    # Use a stable broad header. Extra fields are added if new row has them by rewriting safely.
    if not exists:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        return

    old = pd.read_csv(path)
    cols = list(old.columns)
    for k in row.keys():
        if k not in cols:
            cols.append(k)
            old[k] = ""
    new = pd.concat([old, pd.DataFrame([{k: row.get(k, "") for k in cols}])], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def save_summary(rows_path: Path, summary_path: Path):
    if not rows_path.exists() or rows_path.stat().st_size == 0:
        return
    df = pd.read_csv(rows_path)
    if df.empty:
        return
    group_cols = ["dataset", "method"]
    s = df.groupby(group_cols).agg(
        n=("test_acc", "size"),
        val_mean=("val_acc", "mean"),
        val_std=("val_acc", "std"),
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", "std"),
        train_size_mean=("train_size", "mean"),
        val_size_mean=("val_size", "mean"),
        test_size_mean=("test_size", "mean"),
    ).reset_index()
    s.to_csv(summary_path, index=False, encoding="utf-8-sig")


def load_existing_keys(rows_path: Path) -> set:
    if not rows_path.exists() or rows_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(rows_path)
    keys = set()
    for _, r in df.iterrows():
        keys.add((str(r["dataset"]), int(r["target_id"])))
    return keys


def find_baseline_csv(path_arg: Optional[str]) -> Optional[Path]:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg))
    candidates += [
        ROOT / "baselines" / "pyg_baselines_fair_best_by_val_partial(1).csv",
        ROOT / "baselines" / "pyg_baselines_fair_best_by_val_partial.csv",
        ROOT / "pyg_baselines_fair_best_by_val_partial(1).csv",
        ROOT / "pyg_baselines_fair_best_by_val_partial.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def baseline_targets(baseline_csv: Optional[Path], datasets: List[str], max_repeats: int) -> Dict[str, List[Dict]]:
    """Return target split descriptors.

    If baseline CSV is present, use its unique split_id/repeat/seed records per dataset.
    Otherwise fall back to target_id=0..max_repeats-1.
    """
    out: Dict[str, List[Dict]] = {}
    if baseline_csv is None:
        for ds in datasets:
            out[ds] = [dict(target_id=i, split_id=i, repeat=i, seed=None, baseline_policy="fallback_0_to_nminus1") for i in range(max_repeats)]
        return out

    b = pd.read_csv(baseline_csv)
    for ds in datasets:
        sub = b[b["dataset"].astype(str) == ds].copy()
        if sub.empty:
            print(f"[WARN] baseline CSV has no rows for {ds}; using 0..{max_repeats-1}")
            out[ds] = [dict(target_id=i, split_id=i, repeat=i, seed=None, baseline_policy="fallback_no_baseline_rows") for i in range(max_repeats)]
            continue

        # Prefer split_id if present; it is the closest alignment unit for fixed Geom-GCN masks.
        records = []
        if "split_id" in sub.columns:
            # Keep one row per split_id, sorted. Do not depend on method count.
            for sid in sorted(pd.unique(sub["split_id"].dropna())):
                if len(records) >= max_repeats:
                    break
                ss = sub[sub["split_id"] == sid].iloc[0]
                rep = int(ss["repeat"]) if "repeat" in sub.columns and pd.notna(ss.get("repeat", np.nan)) else int(sid)
                seed = int(ss["seed"]) if "seed" in sub.columns and pd.notna(ss.get("seed", np.nan)) else None
                policy = str(ss["split_policy"]) if "split_policy" in sub.columns else "baseline_split_id"
                records.append(dict(target_id=int(sid), split_id=int(sid), repeat=rep, seed=seed, baseline_policy=policy))
        elif "repeat" in sub.columns:
            for rep in sorted(pd.unique(sub["repeat"].dropna())):
                if len(records) >= max_repeats:
                    break
                ss = sub[sub["repeat"] == rep].iloc[0]
                seed = int(ss["seed"]) if "seed" in sub.columns and pd.notna(ss.get("seed", np.nan)) else None
                policy = str(ss["split_policy"]) if "split_policy" in sub.columns else "baseline_repeat"
                records.append(dict(target_id=int(rep), split_id=int(rep), repeat=int(rep), seed=seed, baseline_policy=policy))
        else:
            records = [dict(target_id=i, split_id=i, repeat=i, seed=None, baseline_policy="fallback_no_split_columns") for i in range(max_repeats)]

        # If baseline gives fewer than max_repeats, fill deterministically.
        used = {int(r["target_id"]) for r in records}
        i = 0
        while len(records) < max_repeats:
            if i not in used:
                records.append(dict(target_id=i, split_id=i, repeat=i, seed=None, baseline_policy="filled_missing_target"))
            i += 1
        out[ds] = records[:max_repeats]
    return out


def make_split(graph, target: Dict, fallback_seed0: int = 20260419):
    ds = graph.dataset
    sid = int(target.get("split_id", target.get("target_id", 0)))
    rep = int(target.get("repeat", target.get("target_id", sid)))
    seed = target.get("seed", None)
    if seed is None or (isinstance(seed, float) and np.isnan(seed)):
        seed = fallback_seed0 + rep
    seed = int(seed)

    if has_fixed_splits(graph):
        try:
            tr, va, te = load_fixed_split(graph, sid)
            return tr, va, te, f"fixed_geom_gcn_split_id_{sid}"
        except Exception as e:
            print(f"[WARN] fixed split_id={sid} failed for {ds}: {e}; fallback to random matching counts")

    # If no fixed mask exists, use class-balanced random protocol. This is aligned
    # by seed when baseline CSV includes seed; otherwise deterministic by repeat.
    tr, va, te, policy = random_split_matching_protocol(graph, seed, prefer_fixed_counts=True)
    return tr, va, te, policy + f"_seed_{seed}"


def run_one_dataset(graph, dataset: str, targets: List[Dict], grid, out_dir: Path, rows_path: Path, trace_path: Path, completed: set):
    print(f"\n=== SRC-v16c fill main table: {dataset} ===")
    for target in targets:
        target_id = int(target["target_id"])
        key = (dataset, target_id)
        if key in completed:
            print(f"skip completed {dataset} target_id={target_id}")
            continue

        train_idx, val_idx, test_idx, split_policy = make_split(graph, target)
        trace = []
        t0 = time.perf_counter()
        cfg, pred, info, metrics = select_by_validation(graph, train_idx, val_idx, test_idx, grid, trace=trace)
        elapsed = time.perf_counter() - t0

        cfg_dict = asdict(cfg)
        row = dict(
            experiment="MAIN_SRC_V16C_10REPEAT_FILL",
            dataset=dataset,
            method="src_v16c",
            target_id=target_id,
            split_id=int(target.get("split_id", target_id)),
            repeat=int(target.get("repeat", target_id)),
            seed=target.get("seed", ""),
            baseline_policy=target.get("baseline_policy", ""),
            split_policy=split_policy,
            num_nodes=graph.num_nodes,
            num_features=graph.num_features,
            num_classes=graph.num_classes,
            train_size=len(train_idx),
            val_size=len(val_idx),
            test_size=len(test_idx),
            selection_rule="grid_selected_by_val_only_no_test",
            time_sec=float(elapsed),
            alphas_json=json.dumps(list(cfg.alphas), ensure_ascii=False),
            pca_dims_json=json.dumps(info.get("pca_dims", {}), ensure_ascii=False),
            retained_variance_json=json.dumps(info.get("retained_variance", {}), ensure_ascii=False),
            stored_coefficients_proxy=info.get("stored_coefficients_proxy", ""),
            **cfg_dict,
            **metrics,
        )
        append_csv(rows_path, row)
        completed.add(key)

        for t in trace:
            trow = dict(
                experiment="MAIN_SRC_V16C_10REPEAT_FILL_TRACE",
                dataset=dataset,
                target_id=target_id,
                split_id=int(target.get("split_id", target_id)),
                repeat=int(target.get("repeat", target_id)),
                **t,
            )
            append_csv(trace_path, trow)

        save_summary(rows_path, out_dir / "src_v16c_main_table_10repeat_summary.csv")
        print(
            f"{dataset} target={target_id} split={row['split_id']} repeat={row['repeat']} "
            f"val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f} "
            f"time={elapsed:.1f}s cfg={cfg_dict}"
        )


def main():
    ap = argparse.ArgumentParser(description="Fill src_v16c main table to 10 repeats aligned with baseline splits.")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--baseline-csv", default=None)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS_7)
    ap.add_argument("--include-actor", action="store_true")
    ap.add_argument("--only-actor", action="store_true")
    ap.add_argument("--max-repeats", type=int, default=10)
    ap.add_argument("--grid", choices=["default", "fast"], default="default")
    ap.add_argument("--out-dir", default="results/src_v16c_main_table_10repeat_fill")
    args = ap.parse_args()

    data_root = discover_data_root(args.data_root)
    baseline_csv = find_baseline_csv(args.baseline_csv)
    if baseline_csv:
        print(f"Using baseline CSV: {baseline_csv}")
    else:
        print("[WARN] baseline CSV not found; using deterministic split ids 0..max_repeats-1")

    datasets = ["actor"] if args.only_actor else list(args.datasets)
    if args.include_actor and "actor" not in datasets:
        datasets.append("actor")

    out_dir = ensure_dir(Path(args.out_dir))
    rows_path = out_dir / "src_v16c_main_table_10repeat_rows.csv"
    trace_path = out_dir / "src_v16c_main_table_10repeat_candidate_trace.csv"
    completed = load_existing_keys(rows_path)
    grid = grid_from_name(args.grid)
    targets_by_dataset = baseline_targets(baseline_csv, datasets, int(args.max_repeats))

    meta = dict(
        data_root=str(data_root),
        baseline_csv=str(baseline_csv) if baseline_csv else "",
        datasets=datasets,
        max_repeats=int(args.max_repeats),
        grid=args.grid,
        protocol=[
            "feature_selection_train_only",
            "pca_fit_train_only",
            "ridge_fit_train_only",
            "hyperparameter_selection_validation_only",
            "test_used_only_for_final_reporting",
            "row_appended_immediately_for_resume",
        ],
    )
    (out_dir / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for ds in datasets:
        graph = load_dataset(data_root, ds)
        run_one_dataset(graph, ds, targets_by_dataset[ds], grid, out_dir, rows_path, trace_path, completed)

    save_summary(rows_path, out_dir / "src_v16c_main_table_10repeat_summary.csv")
    print(f"\nDONE. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
