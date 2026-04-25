# -*- coding: utf-8 -*-
"""
Merge per-repeat strongest-baseline result CSVs and produce paper-style summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_strong_baseline_cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    files = sorted((out / "single_runs").glob("*__best.csv"))
    if not files:
        raise SystemExit(f"No single run files found under {out / 'single_runs'}")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "method", "repeat"], keep="last")
    df = df.sort_values(["dataset", "repeat", "method"])
    merged_path = out / "strongest_baseline_repeats_merged.csv"
    df.to_csv(merged_path, index=False)

    grouped = df.groupby(["dataset", "method"], as_index=False).agg(
        n=("test_acc", "count"),
        train_mean=("train_acc", "mean"),
        train_std=("train_acc", "std"),
        val_mean=("val_acc", "mean"),
        val_std=("val_acc", "std"),
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", "std"),
        time_mean=("time_sec", "mean"),
        params_mean=("params", "mean"),
        train_size=("train_size", "first"),
        val_size=("val_size", "first"),
        test_size=("test_size", "first"),
        num_nodes=("num_nodes", "first"),
        num_features=("num_features", "first"),
        num_classes=("num_classes", "first"),
        split_policy=("split_policy", "first"),
    )

    for col in ["train_mean", "train_std", "val_mean", "val_std", "test_mean", "test_std"]:
        grouped[col + "_%"] = grouped[col] * 100.0

    summary_path = out / "strongest_baseline_summary.csv"
    grouped.to_csv(summary_path, index=False)

    print(f"Wrote: {merged_path}")
    print(f"Wrote: {summary_path}")
    print(grouped[["dataset", "method", "n", "test_mean_%", "test_std_%", "train_size", "val_size", "test_size", "split_policy"]].to_string(index=False))


if __name__ == "__main__":
    main()
