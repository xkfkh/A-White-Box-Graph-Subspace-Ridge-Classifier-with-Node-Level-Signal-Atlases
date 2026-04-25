#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge per-run E05 same-split no-full ablation outputs."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def read_many(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df["source_file"] = f
                frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {f}: {e!r}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/E05_ablation_same_splits_no_full")
    ap.add_argument("--expected-repeats", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out_dir)
    rows = read_many(str(out / "single_runs" / "*" / "E05_ablation_rows.csv"))
    traces = read_many(str(out / "single_runs" / "*" / "E05_ablation_candidate_trace.csv"))
    splits = read_many(str(out / "single_runs" / "*" / "E05_split_indices.csv"))

    if rows.empty:
        raise RuntimeError(f"No rows found under {out / 'single_runs'}")

    rows.to_csv(out / "E05_ablation_rows.csv", index=False, encoding="utf-8-sig")
    if not traces.empty:
        traces.to_csv(out / "E05_ablation_candidate_trace.csv", index=False, encoding="utf-8-sig")
    if not splits.empty:
        splits.to_csv(out / "E05_split_indices.csv", index=False, encoding="utf-8-sig")

    summary = (
        rows.groupby(["dataset", "ablation"])
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
            selection_rule=("selection_rule", "first"),
        )
        .reset_index()
    )
    for col in ["train_mean", "train_std", "val_mean", "val_std", "test_mean", "test_std"]:
        summary[col + "_%"] = summary[col] * 100.0

    summary.to_csv(out / "E05_ablation_summary.csv", index=False, encoding="utf-8-sig")

    # Audit table: every dataset × ablation should have 10 runs.
    expected = int(args.expected_repeats)
    bad = summary[summary["n"] != expected].copy()

    print(f"merged rows: {len(rows)}")
    print(f"datasets: {sorted(rows['dataset'].unique())}")
    print(f"ablations: {sorted(rows['ablation'].unique())}")
    print()
    print(summary[["dataset", "ablation", "n", "test_mean_%", "test_std_%", "split_policy"]].to_string(index=False))

    if len(bad) > 0:
        print("\n[WARN] Some dataset × ablation groups do not have expected n:")
        print(bad[["dataset", "ablation", "n"]].to_string(index=False))
    else:
        print(f"\nAll dataset × ablation groups have n={expected}.")

    if "full" in set(rows["ablation"].astype(str)):
        raise RuntimeError("Found ablation='full', but full should be excluded.")

    print("\nWrote:")
    print(out / "E05_ablation_rows.csv")
    print(out / "E05_ablation_summary.csv")
    if not traces.empty:
        print(out / "E05_ablation_candidate_trace.csv")
    if not splits.empty:
        print(out / "E05_split_indices.csv")


if __name__ == "__main__":
    main()
