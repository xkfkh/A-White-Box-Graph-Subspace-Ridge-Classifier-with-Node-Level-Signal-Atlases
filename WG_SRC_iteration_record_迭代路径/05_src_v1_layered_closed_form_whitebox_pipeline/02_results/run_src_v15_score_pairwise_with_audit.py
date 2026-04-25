#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run src_v15 score + pairwise calibrated model with audit records.

Outputs:
  - split summary
  - test node predictions
  - misclassified test nodes
  - branch summaries and member rows
  - dim search trace inherited from src_v14
  - class bias calibration trace
  - pairwise specialist trace
  - extra branch effect summary under final v15 prediction
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            pass
        return
    keys, seen = [], set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in keys})


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
        if (p / 'scripts').exists() and (p / 'src_v15').exists() and (p / 'src_v14').exists():
            return p
    raise FileNotFoundError('Cannot locate project root containing scripts, src_v15, src_v14')


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


def resolve_default_paths(dataset: str, src_v15_file: str | None, src_v14_file: str | None):
    this_file = Path(__file__).resolve()
    project_root = discover_project_root(this_file.parent)
    drive_root = discover_drive_root(this_file.parent)
    data_base = drive_root / 'planetoid' / 'data' / dataset
    out_dir = project_root / 'scripts' / f'results_src_v15_score_pairwise_audit_{dataset}'

    if src_v15_file:
        src15_path = Path(src_v15_file)
    else:
        src15_path = find_first_existing([
            project_root / 'src_v15' / 'algo1_multihop_pca_score_pairwise_calibrated_src_v15.py',
            project_root / 'src_v15' / 'algo1_multihop_pca_score_pairwise_calibrated.py',
        ])
    if src15_path is None or not src15_path.exists():
        raise FileNotFoundError('Cannot find src_v15 algorithm file. Please pass --src-v15-file.')

    if src_v14_file:
        src14_path = Path(src_v14_file)
    else:
        src14_path = find_first_existing([
            project_root / 'src_v14' / 'algo1_multihop_pca_perclass_dim_floor_coverage_branch_src_v14.py',
            project_root / 'src_v14' / 'algo1_multihop_pca_perclass_dim_floor_coverage_branch.py',
        ])
    if src14_path is None or not src14_path.exists():
        raise FileNotFoundError('Cannot find src_v14 algorithm file. Please pass --src-v14-file.')

    return project_root, drive_root, data_base, out_dir, src15_path, src14_path


def pairwise_branch_geometry(branch_a, branch_b):
    mu_a = np.asarray(branch_a['mu'], dtype=np.float64)
    mu_b = np.asarray(branch_b['mu'], dtype=np.float64)
    B_a = np.asarray(branch_a['basis'], dtype=np.float64)
    B_b = np.asarray(branch_b['basis'], dtype=np.float64)

    center_l2 = float(np.linalg.norm(mu_a - mu_b))
    center_sq = float(np.sum((mu_a - mu_b) ** 2))
    if B_a.size == 0 or B_b.size == 0:
        return center_l2, center_sq, 0.0, 0.0, np.nan
    cross = B_a.T @ B_b
    s = np.linalg.svd(cross, compute_uv=False)
    overlap = float(np.linalg.norm(cross, ord='fro'))
    mean_cos2 = float(np.mean(s * s)) if s.size > 0 else 0.0
    min_angle = float(np.degrees(np.arccos(np.clip(np.max(s), -1.0, 1.0)))) if s.size > 0 else np.nan
    return center_l2, center_sq, overlap, mean_cos2, min_angle


def scores_with_removed_branch(v15, mod14, F, branch_models, classes, remove_class, remove_branch_idx):
    reduced = {}
    for c in classes:
        c = int(c)
        reduced[c] = []
        for i, b in enumerate(branch_models[c]):
            if c == int(remove_class) and i == int(remove_branch_idx):
                continue
            reduced[c].append(b)
    return v15.class_stat_matrices(mod14, F, reduced, classes)


def apply_final_from_stats(v15, y, stats, classes, meta, nodes=None):
    class_bias = np.asarray([float(meta['class_bias'][int(c)]) for c in classes], dtype=np.float64)
    scores_cal = stats['score'] + class_bias[None, :]
    pred_bias = v15.pred_from_scores(scores_cal, classes)
    if nodes is None:
        nodes = np.arange(scores_cal.shape[0], dtype=np.int64)
    pred_final = pred_bias.copy()
    if meta.get('pairwise_specialists'):
        pred_final = v15.apply_pairwise_specialists(
            y,
            np.asarray(nodes, dtype=np.int64),
            classes,
            stats,
            scores_cal,
            pred_final,
            meta['pairwise_specialists'],
            meta['train_exposure_1hop'],
            meta['train_exposure_2hop'],
        )
    return scores_cal, pred_bias, pred_final


def run_audit(v15, mod14, dataset, data_base, out_dir, dim_candidates, num_splits, src_v14_file):
    raw_root = mod14.find_raw_root(data_base)
    X_raw, y, A_sym, original_node_ids = mod14.load_chameleon_raw(raw_root)
    F = v15.build_multihop_features(mod14, X_raw, A_sym)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Data base:    {data_base}')
    print(f'Raw root:     {raw_root}')
    print(f'Output dir:   {out_dir}')
    print(f'Feature dim:  {F.shape[1]}')

    split_rows = []
    node_rows = []
    mis_rows = []
    branch_rows = []
    member_rows = []
    dim_trace_rows = []
    class_bias_trace_rows = []
    pairwise_trace_rows = []
    oof_rows = []
    extra_effect_rows = []
    extra_rewrite_rows = []
    pairwise_geometry_rows = []

    for split in range(int(num_splits)):
        train_idx, val_idx, test_idx = mod14.load_split(raw_root, split)
        classes = np.unique(y[train_idx])
        class_to_pos = {int(c): pos for pos, c in enumerate(classes)}

        branch_models, meta = v15.fit_src_v15_model(
            mod14, F, y, train_idx, val_idx, classes,
            dim_candidates=dim_candidates,
            A_sym=A_sym,
        )
        stats = meta['stats']
        scores_cal = meta['scores_calibrated']
        base_pred = meta['base_pred']
        bias_pred = meta['pred_after_bias']
        pred = meta['pred_final']

        val_acc = float(np.mean(pred[val_idx] == y[val_idx]))
        test_acc = float(np.mean(pred[test_idx] == y[test_idx]))
        base_val_acc = float(np.mean(base_pred[val_idx] == y[val_idx]))
        base_test_acc = float(np.mean(base_pred[test_idx] == y[test_idx]))
        bias_val_acc = float(np.mean(bias_pred[val_idx] == y[val_idx]))
        bias_test_acc = float(np.mean(bias_pred[test_idx] == y[test_idx]))

        total_branches = sum(len(branch_models[int(c)]) for c in classes)
        extra_branches = sum(1 for c in classes for b in branch_models[int(c)] if b.get('kind') == 'extra')

        base_correct = base_pred[test_idx] == y[test_idx]
        final_correct = pred[test_idx] == y[test_idx]
        bias_correct = bias_pred[test_idx] == y[test_idx]
        split_row = {
            'split': int(split),
            'class_dims_json': json.dumps({str(k): int(v) for k, v in meta['class_dims'].items()}, ensure_ascii=False),
            'class_bias_json': json.dumps({str(k): float(v) for k, v in meta['class_bias'].items()}, ensure_ascii=False),
            'accepted_pairwise_count': int(len(meta['pairwise_specialists'])),
            'accepted_pairwise_pairs_json': json.dumps([f"{int(s['class_a'])}-{int(s['class_b'])}" for s in meta['pairwise_specialists']], ensure_ascii=False),
            'base_val_acc': base_val_acc,
            'base_test_acc': base_test_acc,
            'bias_val_acc': bias_val_acc,
            'bias_test_acc': bias_test_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'fixed_by_bias_vs_base': int(np.sum((~base_correct) & bias_correct)),
            'broken_by_bias_vs_base': int(np.sum(base_correct & (~bias_correct))),
            'fixed_final_vs_base': int(np.sum((~base_correct) & final_correct)),
            'broken_final_vs_base': int(np.sum(base_correct & (~final_correct))),
            'total_branches': int(total_branches),
            'extra_branches': int(extra_branches),
        }
        for c in classes:
            c = int(c)
            split_row[f'class_{c}_dim'] = int(meta['class_dims'][c])
            split_row[f'class_{c}_bias'] = float(meta['class_bias'][c])
            split_row[f'class_{c}_branch_count'] = int(len(branch_models[c]))
            split_row[f'class_{c}_extra_branch_count'] = int(sum(1 for b in branch_models[c] if b.get('kind') == 'extra'))
        split_rows.append(split_row)

        print(
            f"split={split:2d} val={val_acc:.4f} test={test_acc:.4f} "
            f"base_test={base_test_acc:.4f} bias_test={bias_test_acc:.4f} "
            f"pairs={len(meta['pairwise_specialists'])} branches={total_branches} (extra={extra_branches})"
        )

        # traces
        for r in meta.get('dim_trace', []):
            rr = dict(r); rr['split'] = int(split); dim_trace_rows.append(rr)
        for r in meta.get('class_bias_trace', []):
            rr = dict(r); rr['split'] = int(split); class_bias_trace_rows.append(rr)
        for r in meta.get('pairwise_trace', []):
            rr = dict(r); rr['split'] = int(split); pairwise_trace_rows.append(rr)

        # OOF confusion and branch debug
        for true_c, mp in meta.get('confusion_counts', {}).items():
            for pred_c, cnt in mp.items():
                oof_rows.append({
                    'split': int(split),
                    'kind': 'oof_root_confusion',
                    'true_class': int(true_c),
                    'other_class': int(pred_c),
                    'count': int(cnt),
                })
        for c, debug_list in meta.get('branch_debug', {}).items():
            for idx, d in enumerate(debug_list):
                row = {'split': int(split), 'kind': 'branch_debug', 'class_label': int(c), 'debug_index': int(idx)}
                row.update(d)
                oof_rows.append(row)

        # branch summary and members
        all_branch_refs = []
        for c in classes:
            c = int(c)
            for b_idx, b in enumerate(branch_models[c]):
                member_idx = np.asarray(b.get('member_idx', []), dtype=np.int64)
                confuser = b.get('confuser_class', '')
                branch_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'class_dim': int(meta['class_dims'][c]),
                    'branch_index': int(b_idx),
                    'kind': b.get('kind', ''),
                    'confuser_class': confuser,
                    'basis_dim': int(np.asarray(b['basis']).shape[1]),
                    'feature_dim': int(np.asarray(b['basis']).shape[0]),
                    'member_size': int(member_idx.size),
                    'seed_size': b.get('seed_size', ''),
                    'median_gain': b.get('median_gain', ''),
                    'max_dim_used': b.get('max_dim_used', ''),
                    'radius': float(b.get('radius', np.nan)),
                    'class_scale': float(b.get('class_scale', np.nan)),
                    'extra_bias_scale': float(b.get('extra_bias_scale', np.nan)),
                    'gate_strength': float(b.get('gate_strength', np.nan)),
                    'member_idx_json': json.dumps([int(x) for x in member_idx.tolist()], ensure_ascii=False),
                    'member_original_id_json': json.dumps([int(original_node_ids[x]) for x in member_idx.tolist()], ensure_ascii=False),
                })
                for node in member_idx:
                    member_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'class_dim': int(meta['class_dims'][c]),
                        'branch_index': int(b_idx),
                        'kind': b.get('kind', ''),
                        'confuser_class': confuser,
                        'node_idx_sorted': int(node),
                        'original_node_id': int(original_node_ids[node]),
                    })
                all_branch_refs.append((c, b_idx, b))

        for i in range(len(all_branch_refs)):
            ci, bi, bri = all_branch_refs[i]
            for j in range(i + 1, len(all_branch_refs)):
                cj, bj, brj = all_branch_refs[j]
                center_l2, center_sq, overlap, mean_cos2, min_angle = pairwise_branch_geometry(bri, brj)
                pairwise_geometry_rows.append({
                    'split': int(split),
                    'class_i': int(ci),
                    'branch_i': int(bi),
                    'kind_i': bri.get('kind', ''),
                    'class_j': int(cj),
                    'branch_j': int(bj),
                    'kind_j': brj.get('kind', ''),
                    'center_l2_distance': center_l2,
                    'center_sq_distance': center_sq,
                    'subspace_overlap_fro': overlap,
                    'mean_cos2_principal': mean_cos2,
                    'min_principal_angle_deg': min_angle,
                })

        # Node rows
        top1, top2, top1_pos, top2_pos, gap = v15.top2_info(scores_cal, classes)

        for node in test_idx:
            true_label = int(y[node])
            true_pos = class_to_pos[true_label]
            pred_label = int(pred[node])
            pred_pos = class_to_pos[pred_label]
            base_label = int(base_pred[node])
            bias_label = int(bias_pred[node])

            true_branch_i = int(stats['best_branch_idx'][node, true_pos])
            pred_branch_i = int(stats['best_branch_idx'][node, pred_pos])
            true_branch = branch_models[true_label][true_branch_i]
            pred_branch = branch_models[pred_label][pred_branch_i]
            center_l2, center_sq, overlap, mean_cos2, min_angle = pairwise_branch_geometry(true_branch, pred_branch)

            st_score = float(scores_cal[node, true_pos])
            sp_score = float(scores_cal[node, pred_pos])
            node_row = {
                'split': int(split),
                'node_idx_sorted': int(node),
                'original_node_id': int(original_node_ids[node]),
                'true_label': true_label,
                'base_pred_label': base_label,
                'bias_pred_label': bias_label,
                'adaptive_pred_label': pred_label,
                'is_correct_base': int(base_label == true_label),
                'is_correct_bias': int(bias_label == true_label),
                'is_correct_full': int(pred_label == true_label),
                'top1_after_bias': int(top1[node]),
                'top2_after_bias': int(top2[node]),
                'top2_gap_after_bias': float(gap[node]),
                'accepted_pairwise_count': int(len(meta['pairwise_specialists'])),
                'class_dims_json': json.dumps({str(k): int(v) for k, v in meta['class_dims'].items()}, ensure_ascii=False),
                'class_bias_json': json.dumps({str(k): float(v) for k, v in meta['class_bias'].items()}, ensure_ascii=False),
                'true_class_dim': int(meta['class_dims'][true_label]),
                'pred_class_dim': int(meta['class_dims'][pred_label]),
                'pred_branch_kind': pred_branch.get('kind', ''),
                'pred_branch_index': int(pred_branch_i),
                'pred_branch_confuser_class': pred_branch.get('confuser_class', ''),
                'true_branch_kind': true_branch.get('kind', ''),
                'true_branch_index': int(true_branch_i),
                'true_branch_confuser_class': true_branch.get('confuser_class', ''),
                'adaptive_true_score': st_score,
                'adaptive_pred_score': sp_score,
                'adaptive_true_total_sq': float(stats['total'][node, true_pos]),
                'adaptive_pred_total_sq': float(stats['total'][node, pred_pos]),
                'adaptive_true_evidence_sq': float(stats['evidence'][node, true_pos]),
                'adaptive_pred_evidence_sq': float(stats['evidence'][node, pred_pos]),
                'adaptive_true_residual': float(stats['residual'][node, true_pos]),
                'adaptive_pred_residual': float(stats['residual'][node, pred_pos]),
                'true_explained_ratio': float(stats['explained_ratio'][node, true_pos]),
                'pred_explained_ratio': float(stats['explained_ratio'][node, pred_pos]),
                'delta_total_pred_minus_true': float(stats['total'][node, pred_pos] - stats['total'][node, true_pos]),
                'delta_evidence_pred_minus_true': float(stats['evidence'][node, pred_pos] - stats['evidence'][node, true_pos]),
                'delta_residual_pred_minus_true': float(stats['residual'][node, pred_pos] - stats['residual'][node, true_pos]),
                'delta_explained_ratio_pred_minus_true': float(stats['explained_ratio'][node, pred_pos] - stats['explained_ratio'][node, true_pos]),
                'pred_bias_penalty': float(stats['bias_penalty'][node, pred_pos]),
                'pred_gate_penalty': float(stats['gate_penalty'][node, pred_pos]),
                'pred_normalized_center_dist': float(stats['normalized_center_dist'][node, pred_pos]),
                'true_pred_branch_center_l2_distance': center_l2,
                'true_pred_branch_center_sq_distance': center_sq,
                'true_pred_branch_subspace_overlap_fro': overlap,
                'true_pred_branch_mean_cos2_principal': mean_cos2,
                'true_pred_branch_min_principal_angle_deg': min_angle,
                'train_exposure_true_1hop': float(meta['train_exposure_1hop'][node, true_pos]),
                'train_exposure_pred_1hop': float(meta['train_exposure_1hop'][node, pred_pos]),
                'train_exposure_true_2hop': float(meta['train_exposure_2hop'][node, true_pos]),
                'train_exposure_pred_2hop': float(meta['train_exposure_2hop'][node, pred_pos]),
            }
            node_rows.append(node_row)
            if pred_label != true_label:
                mis_rows.append(node_row)

        # Extra branch effect under final calibration.
        full_pred_test = pred[test_idx]
        full_correct_test = full_pred_test == y[test_idx]
        full_pred_val = pred[val_idx]
        full_correct_val = full_pred_val == y[val_idx]

        for c in classes:
            c = int(c)
            for b_idx, b in enumerate(branch_models[c]):
                if b.get('kind') != 'extra':
                    continue
                stats_rm = scores_with_removed_branch(v15, mod14, F, branch_models, classes, c, b_idx)
                scores_rm, _, pred_rm = apply_final_from_stats(v15, y, stats_rm, classes, meta)

                rm_test = pred_rm[test_idx]
                rm_correct_test = rm_test == y[test_idx]
                rm_val = pred_rm[val_idx]
                rm_correct_val = rm_val == y[val_idx]

                test_fixed = int(np.sum(full_correct_test & (~rm_correct_test)))
                test_broken = int(np.sum((~full_correct_test) & rm_correct_test))
                test_redirect = int(np.sum((~full_correct_test) & (~rm_correct_test) & (full_pred_test != rm_test)))

                val_fixed = int(np.sum(full_correct_val & (~rm_correct_val)))
                val_broken = int(np.sum((~full_correct_val) & rm_correct_val))
                val_redirect = int(np.sum((~full_correct_val) & (~rm_correct_val) & (full_pred_val != rm_val)))

                extra_effect_rows.append({
                    'split': int(split),
                    'class_label': int(c),
                    'branch_index': int(b_idx),
                    'confuser_class': b.get('confuser_class', ''),
                    'class_dim': int(meta['class_dims'][c]),
                    'basis_dim': int(np.asarray(b['basis']).shape[1]),
                    'member_size': int(len(np.asarray(b.get('member_idx', [])))),
                    'seed_size': b.get('seed_size', ''),
                    'median_gain': b.get('median_gain', ''),
                    'val_fixed_by_this_branch': val_fixed,
                    'val_broken_by_this_branch': val_broken,
                    'val_wrong_redirect_by_this_branch': val_redirect,
                    'test_fixed_by_this_branch': test_fixed,
                    'test_broken_by_this_branch': test_broken,
                    'test_wrong_redirect_by_this_branch': test_redirect,
                    'net_fix_test': int(test_fixed - test_broken),
                    'net_fix_val': int(val_fixed - val_broken),
                })

                changed = test_idx[full_pred_test != rm_test]
                for node in changed:
                    extra_rewrite_rows.append({
                        'split': int(split),
                        'class_label': int(c),
                        'branch_index': int(b_idx),
                        'confuser_class': b.get('confuser_class', ''),
                        'node_idx_sorted': int(node),
                        'original_node_id': int(original_node_ids[node]),
                        'true_label': int(y[node]),
                        'full_pred_label': int(pred[node]),
                        'removed_branch_pred_label': int(pred_rm[node]),
                        'full_correct': int(pred[node] == y[node]),
                        'removed_correct': int(pred_rm[node] == y[node]),
                        'effect_type': (
                            'fixed_by_branch' if (pred[node] == y[node] and pred_rm[node] != y[node])
                            else 'broken_by_branch' if (pred[node] != y[node] and pred_rm[node] == y[node])
                            else 'wrong_redirect_by_branch' if (pred[node] != y[node] and pred_rm[node] != y[node])
                            else 'changed_between_correct_classes'
                        ),
                    })

    vals = np.asarray([r['val_acc'] for r in split_rows], dtype=np.float64)
    tests = np.asarray([r['test_acc'] for r in split_rows], dtype=np.float64)
    summary = {
        'dataset': dataset,
        'data_base': str(data_base),
        'raw_root': str(raw_root),
        'src_v14_file': str(src_v14_file),
        'out_dir': str(out_dir),
        'dim_candidates': json.dumps([int(x) for x in dim_candidates], ensure_ascii=False),
        'num_splits': int(num_splits),
        'val_mean': float(np.mean(vals)),
        'val_std': float(np.std(vals)),
        'test_mean': float(np.mean(tests)),
        'test_std': float(np.std(tests)),
    }

    write_csv(out_dir / 'split_summary_src_v15_score_pairwise_audit.csv', split_rows)
    write_csv(out_dir / 'test_node_predictions_all_src_v15.csv', node_rows)
    write_csv(out_dir / 'misclassified_test_nodes_src_v15.csv', mis_rows)
    write_csv(out_dir / 'branch_subspace_summary_src_v15.csv', branch_rows)
    write_csv(out_dir / 'branch_member_rows_src_v15.csv', member_rows)
    write_csv(out_dir / 'dim_search_trace_src_v15.csv', dim_trace_rows)
    write_csv(out_dir / 'class_bias_trace_src_v15.csv', class_bias_trace_rows)
    write_csv(out_dir / 'pairwise_specialist_trace_src_v15.csv', pairwise_trace_rows)
    write_csv(out_dir / 'oof_confusion_and_branch_debug_src_v15.csv', oof_rows)
    write_csv(out_dir / 'extra_branch_effect_summary_src_v15.csv', extra_effect_rows)
    write_csv(out_dir / 'extra_branch_rewrite_nodes_src_v15.csv', extra_rewrite_rows)
    write_csv(out_dir / 'pairwise_branch_geometry_src_v15.csv', pairwise_geometry_rows)
    (out_dir / 'run_summary_src_v15_score_pairwise_audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print()
    print(f"val_mean  = {summary['val_mean']:.4f} +- {summary['val_std']:.4f}")
    print(f"test_mean = {summary['test_mean']:.4f} +- {summary['test_std']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run src_v15 score/pairwise calibration audit.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--data-base', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--src-v15-file', type=str, default=None)
    parser.add_argument('--src-v14-file', type=str, default=None)
    parser.add_argument('--num-splits', type=int, default=10)
    parser.add_argument('--dims', type=int, nargs='*', default=[16, 24, 32, 48, 64])
    args = parser.parse_args()

    project_root, drive_root, default_data_base, default_out_dir, src15_path, src14_path = resolve_default_paths(
        args.dataset, args.src_v15_file, args.src_v14_file
    )
    data_base = Path(args.data_base) if args.data_base else default_data_base
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    v15 = import_module_from_path(src15_path, 'src_v15_algo')
    mod14 = import_module_from_path(src14_path, 'src_v14_base_for_v15_audit')

    print(f'Project root: {project_root}')
    print(f'Drive root:   {drive_root}')
    print(f'Src_v15 file: {src15_path}')
    print(f'Src_v14 file: {src14_path}')

    run_audit(
        v15,
        mod14,
        dataset=args.dataset,
        data_base=data_base,
        out_dir=out_dir,
        dim_candidates=[int(x) for x in args.dims],
        num_splits=int(args.num_splits),
        src_v14_file=src14_path,
    )


if __name__ == '__main__':
    main()


