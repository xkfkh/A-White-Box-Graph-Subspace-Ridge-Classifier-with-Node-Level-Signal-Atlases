#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'src_v10').exists() and (p / 'scripts').exists():
            return p
    raise FileNotFoundError(
        'Cannot locate project root containing src_v10 and scripts.'
    )


def discover_workspace_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError(
        'Cannot locate workspace root containing planetoid/data.\n'
        'According to your structure tree, it should look like:\n'
        '.../MSR实验复现与创新/planetoid/data'
    )


def main():
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    workspace_root = discover_workspace_root(this_file.parent)

    algo_path = project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10.py'
    data_base = workspace_root / 'planetoid' / 'data' / 'chameleon'
    out_dir = project_root / 'scripts' / 'results_src_v10_safe_adaptive_branch_chameleon'

    if not algo_path.exists():
        raise FileNotFoundError(
            f'Cannot find algorithm file: {algo_path}\n'
            'Please put algo1_multihop_pca_safe_adaptive_branch_src_v10.py into src_v10 first.'
        )

    if not data_base.exists():
        raise FileNotFoundError(
            f'Cannot find dataset directory: {data_base}\n'
            'Expected chameleon under planetoid/data.'
        )

    cmd = [
        sys.executable,
        str(algo_path),
        '--dataset', 'chameleon',
        '--data-base', str(data_base),
        '--out-dir', str(out_dir),
    ]

    print('Project root :', project_root)
    print('Workspace root:', workspace_root)
    print('Data base    :', data_base)
    print('Output dir   :', out_dir)
    print('Running command:')
    print(' '.join(f'"{x}"' if ' ' in x else x for x in cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()


