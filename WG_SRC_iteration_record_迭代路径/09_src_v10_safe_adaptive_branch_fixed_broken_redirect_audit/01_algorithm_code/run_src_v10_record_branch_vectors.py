#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run src_v10 safe adaptive-branch PCA and record all selected branch vectors.

What this script records
------------------------
For the validation-selected best dimension of each split, this script saves:

1. Split summary:
   - best_dim / val_acc / test_acc / branch counts

2. Branch-level vector objects for every selected branch:
   - mu (subspace mean)
   - basis matrix B
   - local center
   - radius
   - member indices / original ids
   - branch type root / extra
   - confuser_class / seed_size / median_gain (if extra)

3. Geometry tables:
   - root-vs-root pairwise geometry between class root subspaces
   - all-branch pairwise geometry (branch centers + subspace overlaps)

Outputs
-------
Default output directory:
    <project_root>/scripts/results_src_v10_record_branch_vectors_<dataset>

Files:
    split_summary_src_v10_record_vectors.csv
    branch_summary_src_v10_vectors.csv
    pairwise_root_geometry_src_v10.csv
    pairwise_branch_geometry_src_v10.csv
    run_summary_src_v10_record_vectors.csv
    src_v10_branch_vectors.npz
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import List

import numpy as np


# ============================================================
# IO helpers
# ============================================================

def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            pass
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            full = {k: row.get(k, '') for k in keys}
            w.writerow(full)


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'scripts').exists() and (p / 'src_v10').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts and src_v10')


def discover_drive_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / 'planetoid' / 'data').exists():
            return p
    raise FileNotFoundError('Cannot locate drive root containing planetoid/data')


def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_default_paths(dataset: str, src_v10_file: str | None):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v10_record_branch_vectors_{dataset}'

    if src_v10_file is not None:
        src_v10_path = Path(src_v10_file)
    else:
        src_v10_path = find_first_existing([
            project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10.py',
            project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch_src_v10(1).py',
            project_root / 'src_v10' / 'algo1_multihop_pca_safe_adaptive_branch.py',
        ])
    if src_v10_path is None or not src_v10_path.exists():
        raise FileNotFoundError('Cannot find src_v10 algorithm file in src_v10. Please pass --src-v10-file explicitly.')

    return project_root, drive_root, data_base, out_dir, src_v10_path


# ============================================================
# Geometry helpers
# ============================================================

def pairwise_stats(mu_i, B_i, mu_j, B_j):
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    B_i = np.asarray(B_i, dtype=np.float64)
    B_j = np.asarray(B_j, dtype=np.float64)

    center_l2 = float(np.linalg.norm(mu_i - mu_j))
    center_sq = float(np.sum((mu_i - mu_j) ** 2))

    if B_i.size == 0 or B_j.size == 0:
        overlap_fro = 0.0
        mean_cos2 = 0.0
        singular_vals = np.empty((0,), dtype=np.float64)
        min_angle_deg = np.nan
        max_angle_deg = np.nan
    else:
        cross = B_i.T @ B_j
        singular_vals = np.linalg.svd(cross, compute_uv=False)
        overlap_fro = float(np.linalg.norm(cross, ord='fro'))
        mean_cos2 = float(np.mean(singular_vals ** 2)) if singular_vals.size > 0 else 0.0
        clipped = np.clip(singular_vals, -1.0, 1.0)
        angles = np.degrees(np.arccos(clipped))
        min_angle_deg = float(np.min(angles)) if angles.size > 0 else np.nan
        max_angle_deg = float(np.max(angles)) if angles.size > 0 else np.nan

    return {
        'center_l2_distance': center_l2,
        'center_sq_distance': center_sq,
        'subspace_overlap_fro': overlap_fro,
        'mean_cos2_principal': mean_cos2,
        'min_principal_angle_deg': min_angle_deg,
        'max_principal_angle_deg': max_angle_deg,
        'principal_cosines_json': json.dumps([float(x) for x in singular_vals.tolist()], ensure_ascii=False),
    }


# ============================================================
# Main runner
# ============================================================

def run_record(
    dataset='chameleon',
    data_base=None,
    out_dir=None,
    src_v10_file=None,
    dim_candidates=None,
    num_splits=10,
):
    if dim_candidates is None:
        dim_candidates = [16, 24, 32, 48, 64]

    project_root, drive_root, default_data_base, default_out_dir, src_v10_path = resolve_default_paths(dataset, src_v10_file)
    data_base = Path(data_base) if data_base is not None else default_data_base
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = import_module_from_path(src_v10_path, 'src_v10_algo_record_vectors')

    raw_root = mod.find_raw_root(data_base)
    X_raw, y, A_sym = mod.load_chameleon_raw(raw_root)

    # Same feature construction as src_v10
    P = mod.row_normalize(A_sym)
    PX = np.asarray(P @ X_raw)
    P2X = np.asarray(P @ PX)
    F = np.hstack([
        mod.row_l2_normalize(X_raw),
        mod.row_l2_normalize(PX),
        mod.row_l2_normalize(P2X),
        mod.row_l2_normalize(X_raw - PX),
        mod.row_l2_normalize(PX - P2X),
    ])

    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Src_v10 file: {src_v10_path}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

    branch_kwargs = dict(
        internal_n_folds=4,
        internal_seed=0,
        high_res_quantile=0.85,
        low_margin_quantile=0.15,
        min_seed_size=8,
        min_branch_size=12,
        local_quantile=0.35,
        max_group_frac=0.45,
        max_extra_branches=2,
        min_gain_ratio=0.10,
        repel_strength=0.03,
        extra_bias_scale=0.10,
        gate_strength=2.5,
    )

    split_rows = []
    branch_rows = []
    root_pair_rows = []
    all_branch_pair_rows = []
    npz_payload = {}

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod.load_split(raw_root, split)
        classes = np.unique(y[train_idx])

        best_val = -1.0
        best_test = -1.0
        best_dim = -1
        best_branch_models = None
        best_pred = None
        best_meta = None

        for dim in dim_candidates:
            branch_models, meta = mod.fit_conservative_adaptive_branches(
                F, y, train_idx, classes, max_dim=dim, **branch_kwargs
            )
            scores, _ = mod.class_scores_from_branch_models(F, branch_models, classes)
            pred = classes[np.argmin(scores, axis=1)]
            val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
            test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_dim = int(dim)
                best_branch_models = branch_models
                best_pred = pred
                best_meta = meta

        total_branches = 0
        extra_branches = 0
        for c in classes:
            bc = len(best_branch_models[int(c)])
            eb = sum(1 for b in best_branch_models[int(c)] if b.get('kind') == 'extra')
            total_branches += bc
            extra_branches += eb

        split_rows.append({
            'split': int(split),
            'best_dim': int(best_dim),
            'val_acc': float(best_val),
            'test_acc': float(best_test),
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
        })

        print(
            f'split={split:2d}  best_dim={best_dim:3d}  '
            f'val={best_val:.4f}  test={best_test:.4f}  '
            f'branches={total_branches} (extra={extra_branches})'
        )

        # Save each branch
        branch_list_for_pairs = []
        for c in classes:
            class_branches = best_branch_models[int(c)]
            for b_idx, b in enumerate(class_branches):
                kind = str(b.get('kind', ''))
                confuser = b.get('confuser_class', None)
                seed_size = b.get('seed_size', None)
                median_gain = b.get('median_gain', None)
                mu = np.asarray(b.get('mu', np.empty((F.shape[1],), dtype=np.float64)), dtype=np.float64)
                B = np.asarray(b.get('basis', np.empty((F.shape[1], 0), dtype=np.float64)), dtype=np.float64)
                center = np.asarray(b.get('center', np.mean(F[np.asarray(b.get('member_idx', []), dtype=np.int64)], axis=0) if len(np.asarray(b.get('member_idx', []), dtype=np.int64)) > 0 else np.zeros(F.shape[1])), dtype=np.float64)
                member_idx = np.asarray(b.get('member_idx', []), dtype=np.int64)
                radius = float(b.get('radius', np.nan))
                class_scale = float(b.get('class_scale', np.nan))
                extra_bias_scale = float(b.get('extra_bias_scale', np.nan))
                gate_strength = float(b.get('gate_strength', np.nan))

                key_prefix = f'split{split}_class{int(c)}_branch{int(b_idx)}'
                npz_payload[f'{key_prefix}_mu'] = mu
                npz_payload[f'{key_prefix}_basis'] = B
                npz_payload[f'{key_prefix}_center'] = center
                npz_payload[f'{key_prefix}_member_idx'] = member_idx
                npz_payload[f'{key_prefix}_class_label'] = np.asarray([int(c)], dtype=np.int64)
                npz_payload[f'{key_prefix}_branch_index'] = np.asarray([int(b_idx)], dtype=np.int64)
                npz_payload[f'{key_prefix}_kind'] = np.asarray([0 if kind == 'root' else 1], dtype=np.int64)
                npz_payload[f'{key_prefix}_radius'] = np.asarray([radius], dtype=np.float64)
                npz_payload[f'{key_prefix}_class_scale'] = np.asarray([class_scale], dtype=np.float64)
                npz_payload[f'{key_prefix}_extra_bias_scale'] = np.asarray([extra_bias_scale], dtype=np.float64)
                npz_payload[f'{key_prefix}_gate_strength'] = np.asarray([gate_strength], dtype=np.float64)

                if confuser is not None:
                    npz_payload[f'{key_prefix}_confuser_class'] = np.asarray([int(confuser)], dtype=np.int64)
                if seed_size is not None:
                    npz_payload[f'{key_prefix}_seed_size'] = np.asarray([int(seed_size)], dtype=np.int64)
                if median_gain is not None:
                    npz_payload[f'{key_prefix}_median_gain'] = np.asarray([float(median_gain)], dtype=np.float64)

                branch_rows.append({
                    'split': int(split),
                    'best_dim': int(best_dim),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': kind,
                    'confuser_class': '' if confuser is None else int(confuser),
                    'basis_dim': int(B.shape[1]),
                    'feature_dim': int(B.shape[0]),
                    'member_size': int(member_idx.size),
                    'radius': radius,
                    'class_scale': class_scale,
                    'extra_bias_scale': extra_bias_scale,
                    'gate_strength': gate_strength,
                    'seed_size': '' if seed_size is None else int(seed_size),
                    'median_gain': '' if median_gain is None else float(median_gain),
                    'mu_l2_norm': float(np.linalg.norm(mu)),
                    'center_l2_norm': float(np.linalg.norm(center)),
                    'member_idx_json': json.dumps([int(x) for x in member_idx.tolist()], ensure_ascii=False),
                })

                branch_list_for_pairs.append({
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'kind': kind,
                    'mu': mu,
                    'basis': B,
                })

        # Root-vs-root class geometry
        root_branches = [b for b in branch_list_for_pairs if b['kind'] == 'root']
        for i in range(len(root_branches)):
            bi = root_branches[i]
            for j in range(i + 1, len(root_branches)):
                bj = root_branches[j]
                stats = pairwise_stats(bi['mu'], bi['basis'], bj['mu'], bj['basis'])
                root_pair_rows.append({
                    'split': int(split),
                    'best_dim': int(best_dim),
                    'class_i': int(bi['class_label']),
                    'class_j': int(bj['class_label']),
                    'branch_i_index': int(bi['branch_index']),
                    'branch_j_index': int(bj['branch_index']),
                    **stats,
                })

        # All-branch geometry
        for i in range(len(branch_list_for_pairs)):
            bi = branch_list_for_pairs[i]
            for j in range(i + 1, len(branch_list_for_pairs)):
                bj = branch_list_for_pairs[j]
                stats = pairwise_stats(bi['mu'], bi['basis'], bj['mu'], bj['basis'])
                all_branch_pair_rows.append({
                    'split': int(split),
                    'best_dim': int(best_dim),
                    'class_i': int(bi['class_label']),
                    'branch_i_index': int(bi['branch_index']),
                    'branch_i_kind': bi['kind'],
                    'class_j': int(bj['class_label']),
                    'branch_j_index': int(bj['branch_index']),
                    'branch_j_kind': bj['kind'],
                    **stats,
                })

    vals = np.asarray([r['val_acc'] for r in split_rows], dtype=np.float64)
    tests = np.asarray([r['test_acc'] for r in split_rows], dtype=np.float64)

    val_mean = float(np.mean(vals))
    val_std = float(np.std(vals))
    test_mean = float(np.mean(tests))
    test_std = float(np.std(tests))

    print()
    print(f'val_mean  = {val_mean:.4f} +- {val_std:.4f}')
    print(f'test_mean = {test_mean:.4f} +- {test_std:.4f}')

    write_csv(out_dir / 'split_summary_src_v10_record_vectors.csv', split_rows)
    write_csv(out_dir / 'branch_summary_src_v10_vectors.csv', branch_rows)
    write_csv(out_dir / 'pairwise_root_geometry_src_v10.csv', root_pair_rows)
    write_csv(out_dir / 'pairwise_branch_geometry_src_v10.csv', all_branch_pair_rows)
    write_csv(out_dir / 'run_summary_src_v10_record_vectors.csv', [{
        'dataset': dataset,
        'data_base': str(data_base),
        'raw_root': str(raw_root),
        'src_v10_file': str(src_v10_path),
        'out_dir': str(out_dir),
        'dim_candidates': json.dumps([int(x) for x in dim_candidates], ensure_ascii=False),
        'num_splits': int(num_splits),
        'val_mean': val_mean,
        'val_std': val_std,
        'test_mean': test_mean,
        'test_std': test_std,
    }])

    npz_payload['feature_dim'] = np.asarray([F.shape[1]], dtype=np.int64)
    npz_payload['num_splits'] = np.asarray([int(num_splits)], dtype=np.int64)
    npz_payload['dim_candidates'] = np.asarray(dim_candidates, dtype=np.int64)
    np.savez_compressed(out_dir / 'src_v10_branch_vectors.npz', **npz_payload)

    return {
        'val_mean': val_mean,
        'val_std': val_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'out_dir': str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description='Run src_v10 and record selected branch vectors.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None,
                        help='Path like D:/planetoid/data/chameleon . Default: auto-discover.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory. Default: project/scripts/results_src_v10_record_branch_vectors_<dataset>')
    parser.add_argument('--src-v10-file', type=str, default=None,
                        help='Path to src_v10 algorithm file. Default: auto-discover under project/src_v10')
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    run_record(
        dataset=args.dataset,
        data_base=args.data_base,
        out_dir=args.out_dir,
        src_v10_file=args.src_v10_file,
        dim_candidates=[int(x) for x in args.dims],
        num_splits=int(args.num_splits),
    )


if __name__ == '__main__':
    main()


