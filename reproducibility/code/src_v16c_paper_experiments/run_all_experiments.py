#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience launcher for paper experiments.

Default mode runs a safe subset. Use --full to run all heavier grids.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def run(cmd):
    print('\n$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=None)
    p.add_argument('--datasets', nargs='+', default=['chameleon','squirrel','cornell','texas','wisconsin','amazon-photo','amazon-computers'])
    p.add_argument('--full', action='store_true')
    p.add_argument('--skip-e11', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable
    ds = args.datasets
    data_arg = [] if args.data_root is None else ['--data-root', args.data_root]
    grid = 'default' if args.full else 'fast'
    split_n = '10' if args.full else '2'
    repeat_n = '5' if args.full else '2'
    e09_splits = '10' if args.full else '2'

    run([py, str(root/'scripts'/'run_E01_baseline_table_aggregate.py')])
    run([py, str(root/'scripts'/'run_E03_random_split_stability.py'), *data_arg, '--datasets', *ds, '--repeats', repeat_n, '--grid', grid])
    run([py, str(root/'scripts'/'run_E05_ablation.py'), *data_arg, '--datasets', *ds, '--splits', split_n, '--grid', grid])
    run([py, str(root/'scripts'/'run_E07_param_scan.py'), *data_arg, '--datasets', *ds[:1], '--splits', split_n])
    run([py, str(root/'scripts'/'run_E08_interpretability.py'), *data_arg, '--datasets', *ds, '--splits', split_n, '--grid', grid])
    run([py, str(root/'scripts'/'run_E09_efficiency.py'), *data_arg, '--datasets', *ds, '--splits', e09_splits, '--grid', grid])
    if not args.skip_e11:
        run([py, str(root/'scripts'/'run_E11_sample_efficiency.py'), *data_arg, '--datasets', *ds, '--repeats', repeat_n, '--grid', grid])

if __name__ == '__main__':
    main()
