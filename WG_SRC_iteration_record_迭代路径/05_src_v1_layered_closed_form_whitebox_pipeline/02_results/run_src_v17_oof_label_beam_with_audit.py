#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for src_v17.
Place this file under your project scripts/ directory.
Place algo_src_v17_oof_label_beam.py under src_v17/ or pass --src-v17-file.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_project_root(start: Path) -> Path:
    start = Path(start).resolve()
    for p in [start] + list(start.parents):
        if (p / "scripts").exists() and ((p / "src_v17").exists() or (p / "src_v16").exists()):
            return p
    raise FileNotFoundError("Cannot locate project root containing scripts and src_v17/src_v16")


def discover_drive_root(start: Path) -> Path:
    start = Path(start).resolve()
    for p in [start] + list(start.parents):
        if (p / "planetoid" / "data").exists():
            return p
    raise FileNotFoundError("Cannot locate drive root containing planetoid/data")


def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_paths(dataset: str, src_v17_file: str | None):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / "planetoid" / "data" / dataset
    out_dir = project_root / "scripts" / f"results_src_v17_oof_label_beam_audit_{dataset}"

    if src_v17_file:
        algo_path = Path(src_v17_file)
    else:
        algo_path = find_first_existing([
            project_root / "src_v17" / "algo_src_v17_oof_label_beam.py",
            project_root / "src_v17" / "algo1_src_v17_oof_label_beam.py",
            project_root / "src_v17" / "src_v17_oof_label_beam.py",
        ])
    if algo_path is None or not algo_path.exists():
        raise FileNotFoundError("Cannot find src_v17 algorithm file. Pass --src-v17-file explicitly.")
    return project_root, drive_root, data_base, out_dir, algo_path


def main():
    parser = argparse.ArgumentParser(description="Run src_v17 OOF-safe label-informed beam search with audit.")
    parser.add_argument("--dataset", default="chameleon")
    parser.add_argument("--data-base", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--src-v17-file", default=None)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Use a smaller grid for quick smoke test.")
    args = parser.parse_args()

    project_root, drive_root, default_data_base, default_out_dir, algo_path = resolve_paths(args.dataset, args.src_v17_file)
    data_base = Path(args.data_base) if args.data_base else default_data_base
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    print(f"Project root: {project_root}")
    print(f"Drive root:   {drive_root}")
    print(f"Src_v17 file: {algo_path}")

    mod = import_module_from_path(algo_path, "src_v17_algo")
    mod.run_experiment(
        dataset=args.dataset,
        data_base=data_base,
        out_dir=out_dir,
        num_splits=int(args.num_splits),
        beam_size=int(args.beam_size),
        fast=bool(args.fast),
        audit=True,
    )


if __name__ == "__main__":
    main()


