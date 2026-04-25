#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit mispredicted nodes for the Chameleon-specific v8 algorithm.

What this script does
---------------------
1. Dynamically imports the user's v8 algorithm .py file.
2. Reads an existing all_grid_results.csv produced by that algorithm.
3. Re-runs selected configs (all or top-k per repeat).
4. Saves, for each config, every mispredicted node with:
   - node id, true/pred label
   - base/final predictions
   - component scores (subspace / ridge / geom / final)
   - graph features (degree, neighbor label ratios, entropy)
   - raw feature sparsity statistics / nonzero feature indices
5. Saves aggregate summaries:
   - per-node summary
   - per-parameter effects
   - per-node per-parameter effects

This is designed for the uploaded Chameleon-only algorithm file.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(',') if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_module_from_py(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def autodetect_algo_py(src_v8: Path) -> Path:
    cands = sorted(src_v8.glob('*.py'))
    cands = [p for p in cands if p.name != '__init__.py']
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise FileNotFoundError(f"No .py algorithm file found under {src_v8}")
    raise RuntimeError(
        f"Multiple .py files found under {src_v8}: {[p.name for p in cands]}. "
        f"Please pass --algo_py explicitly."
    )




def resolve_chameleon_raw_root(data_root: Path) -> Path:
    """Resolve the raw root for Chameleon only, avoiding accidental matches to other datasets."""
    data_root = Path(data_root).resolve()

    candidates = [
        data_root / 'chameleon' / 'raw',
        data_root / 'WikipediaNetwork' / 'chameleon' / 'raw',
        data_root / 'Chameleon' / 'raw',
        data_root / 'chameleon',
    ]
    for p in candidates:
        if (p / 'out1_node_feature_label.txt').exists() and (p / 'out1_graph_edges.txt').exists():
            return p

    # Fallback: only accept paths that contain "chameleon" in their parts.
    hits = []
    for hit in data_root.rglob('out1_node_feature_label.txt'):
        parts = {x.lower() for x in hit.parts}
        if 'chameleon' in parts and (hit.parent / 'out1_graph_edges.txt').exists():
            hits.append(hit.parent)

    hits = sorted(set(hits))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f'Multiple Chameleon raw roots found: {hits}. Please pass a cleaner data_root.')
    raise FileNotFoundError(
        f'Cannot locate Chameleon raw root under {data_root}. Expected something like data/chameleon/raw/'
    )

def read_rows_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def maybe_num(x: str) -> Any:
    if x is None:
        return x
    s = str(x).strip()
    if s == '':
        return s
    for fn in (int, float):
        try:
            return fn(s)
        except Exception:
            pass
    if s in ('True', 'False'):
        return s == 'True'
    return s


def select_rows(rows: List[Dict[str, str]], repeats: Iterable[int], select_mode: str, top_k: int) -> List[Dict[str, str]]:
    repeats = set(int(r) for r in repeats)
    rows = [r for r in rows if int(r['repeat']) in repeats]
    if select_mode == 'all':
        return rows
    if select_mode == 'topk_val':
        out = []
        by_rep = defaultdict(list)
        for r in rows:
            by_rep[int(r['repeat'])].append(r)
        for rep, group in by_rep.items():
            group = sorted(group, key=lambda r: (float(r['val']), float(r['test'])), reverse=True)
            out.extend(group[:top_k])
        return out
    raise ValueError(f'Unknown select_mode={select_mode}')


def build_neighbor_stats(A_sym, y: np.ndarray, node_id: int, classes: np.ndarray) -> Dict[str, Any]:
    start, end = A_sym.indptr[node_id], A_sym.indptr[node_id + 1]
    nbrs = A_sym.indices[start:end]
    deg = int(len(nbrs))
    if deg == 0:
        return {
            'degree_sym': 0,
            'same_label_neighbor_ratio': 0.0,
            'neighbor_label_entropy': 0.0,
            'neighbor_label_hist': '{}',
        }
    nbr_labels = y[nbrs]
    same = float(np.mean(nbr_labels == y[node_id]))
    cnt = Counter(int(v) for v in nbr_labels)
    probs = np.array(list(cnt.values()), dtype=np.float64) / max(sum(cnt.values()), 1)
    ent = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    hist = {str(int(c)): int(cnt.get(int(c), 0)) for c in classes}
    return {
        'degree_sym': deg,
        'same_label_neighbor_ratio': same,
        'neighbor_label_entropy': ent,
        'neighbor_label_hist': json.dumps(hist, ensure_ascii=False),
    }


def nonzero_feature_info(x: np.ndarray, max_keep: int = 128) -> Dict[str, Any]:
    nz = np.flatnonzero(np.abs(x) > 1e-12)
    return {
        'raw_feat_nnz': int(nz.size),
        'raw_feat_nz_idx_head': json.dumps(nz[:max_keep].tolist(), ensure_ascii=False),
    }


def run_one_verbose(mod, X_proc, X_raw_proc, y, A_sym, A_sym_weighted, A_relations,
                    train_idx, val_idx, test_idx, classes, cfg):
    # 1) gated subspace base
    S_sub, sub_info = mod.fit_subspace_base(
        X_proc, y, A_sym_weighted, train_idx, val_idx, classes,
        lambda_smooth=cfg['lambda_smooth'],
        dim=cfg['dim'],
        tau_gate=cfg['tau_gate'],
        residual_norm=cfg['residual_norm'],
    )
    S_sub_n, _ = mod.normalize_score_by_class(S_sub, y, train_idx, classes, mode=cfg['score_norm_mode'])

    # 2) RNR + ridge
    rnr_feat = mod.build_rnr_features(S_sub, A_sym, residual_temperature=cfg['residual_temperature'], variant=cfg['rnr_variant'])
    ridge_input = mod.compose_ridge_input(cfg['feature_mode'], X_raw_proc, rnr_feat)
    S_ridge, ridge_prob = mod.ridge_score_dual(
        ridge_input, y, train_idx, classes,
        alpha=cfg['ridge_alpha'], temperature=cfg['ridge_temperature']
    )
    S_ridge_n, _ = mod.normalize_score_by_class(S_ridge, y, train_idx, classes, mode=cfg['score_norm_mode'])

    # 3) local geometry
    geom_subspaces = sub_info['raw_sub'] if cfg['geometry_use'] == 'raw' else sub_info['smooth_sub']
    S_geom, _ = mod.local_geometry_scores(
        X_proc, A_sym, geom_subspaces, train_idx, y, classes,
        local_rank=cfg['local_rank'], max_neighbors=cfg['max_neighbors'],
        norm_mode=cfg['score_norm_mode']
    )

    # 4) base score
    S_base = (
        float(cfg['alpha_sub']) * S_sub_n +
        float(cfg['beta_ridge']) * S_ridge_n +
        float(cfg['gamma_geom']) * S_geom
    )
    base_pred = classes[np.argmin(S_base, axis=1)]

    # 5) class-level FCCP / final probability
    z, Gs = mod.infer_class_fccp(
        S_base, y, train_idx, A_relations, classes,
        lambda_compat=cfg['lambda_compat'], iterations=cfg['iterations'],
        temperature=cfg['fccp_temperature'], kappa=cfg['kappa'],
        reliability_mode=cfg['reliability_mode'], clamp_train=cfg['clamp_train']
    )
    pred = classes[np.argmax(z, axis=1)]
    score_final = -np.log(np.maximum(z, 1e-12))

    return {
        'S_sub_n': S_sub_n,
        'S_ridge_n': S_ridge_n,
        'S_geom': S_geom,
        'S_base': S_base,
        'score_final': score_final,
        'z': z,
        'pred': pred,
        'base_pred': base_pred,
        'gate': np.asarray(sub_info['gate'], dtype=np.float64),
        'ridge_prob': ridge_prob,
        'metrics': {
            'base_train': mod.accuracy(base_pred, y, train_idx),
            'base_val': mod.accuracy(base_pred, y, val_idx),
            'base_test': mod.accuracy(base_pred, y, test_idx),
            'train': mod.accuracy(pred, y, train_idx),
            'val': mod.accuracy(pred, y, val_idx),
            'test': mod.accuracy(pred, y, test_idx),
        },
    }


def cfg_from_result_row(r: Dict[str, str]) -> Dict[str, Any]:
    keys = [
        'dim', 'lambda_smooth', 'ridge_alpha', 'ridge_temperature', 'residual_temperature',
        'alpha_sub', 'beta_ridge', 'gamma_geom', 'lambda_compat', 'iterations', 'local_rank',
        'max_neighbors', 'geometry_use', 'feature_mode', 'rnr_variant', 'tau_gate',
        'residual_norm', 'score_norm_mode', 'reliability_mode', 'kappa', 'fccp_temperature', 'clamp_train',
    ]
    cfg = {}
    for k in keys:
        if k not in r:
            continue
        cfg[k] = maybe_num(r[k])
    # Fix numeric types expected by functions
    for k in ['dim', 'iterations', 'local_rank', 'max_neighbors']:
        if k in cfg:
            cfg[k] = int(cfg[k])
    for k in ['lambda_smooth', 'ridge_alpha', 'ridge_temperature', 'residual_temperature', 'alpha_sub',
              'beta_ridge', 'gamma_geom', 'lambda_compat', 'tau_gate', 'kappa', 'fccp_temperature']:
        if k in cfg:
            cfg[k] = float(cfg[k])
    if 'clamp_train' in cfg:
        cfg['clamp_train'] = bool(cfg['clamp_train'])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_v8', type=str, required=True, help='Folder containing the v8 Chameleon algorithm .py file')
    ap.add_argument('--algo_py', type=str, default=None, help='Explicit path to the algorithm .py file')
    ap.add_argument('--results_csv', type=str, required=True, help='Existing all_grid_results.csv from the v8 run')
    ap.add_argument('--data_root', type=str, required=True, help='Root folder containing Chameleon raw/split files')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--repeats', type=str, default='0', help='Comma-separated repeats to audit, e.g. 0 or 0,1,2')
    ap.add_argument('--select_mode', type=str, default='all', choices=['all', 'topk_val'])
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--max_feature_idx_keep', type=int, default=128)
    args = ap.parse_args()

    src_v8 = Path(args.src_v8).resolve()
    algo_py = Path(args.algo_py).resolve() if args.algo_py else autodetect_algo_py(src_v8)
    results_csv = Path(args.results_csv).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    print('src_v8 =', src_v8)
    print('algo_py =', algo_py)
    print('results_csv =', results_csv)
    print('data_root =', data_root)
    print('raw_root =', resolve_chameleon_raw_root(data_root))
    print('out_dir =', out_dir)

    mod = load_module_from_py(algo_py)
    raw_root = resolve_chameleon_raw_root(data_root)
    X_raw, y, A_out, A_in, A_sym, A_sym_weighted = mod.load_chameleon_raw(raw_root)

    rows = read_rows_csv(results_csv)
    selected = select_rows(rows, parse_int_list(args.repeats), args.select_mode, args.top_k)
    print(f'selected configs = {len(selected)}')

    error_rows = []
    grid_summary_rows = []

    for rr in selected:
        repeat = int(rr['repeat'])
        grid_id = int(rr['grid_id'])
        cfg = cfg_from_result_row(rr)
        relations = rr.get('relations', 'sym')
        preprocess_mode = rr.get('preprocess_mode', 'raw')

        train_idx, val_idx, test_idx = mod.load_split(raw_root, repeat)
        classes = np.unique(y[train_idx])
        A_relations = mod.build_relations(A_out, A_in, A_sym, relations)
        X_proc = mod.preprocess_features(X_raw, train_idx, mode=preprocess_mode)
        X_raw_proc = np.asarray(X_raw, dtype=np.float64).copy()

        out = run_one_verbose(
            mod, X_proc, X_raw_proc, y, A_sym, A_sym_weighted, A_relations,
            train_idx, val_idx, test_idx, classes, cfg
        )
        pred = out['pred']
        base_pred = out['base_pred']
        wrong_mask = pred != y
        wrong_idx = np.where(wrong_mask)[0]

        grid_summary_rows.append({
            'repeat': repeat,
            'grid_id': grid_id,
            'n_wrong_all': int(wrong_idx.size),
            'n_wrong_train': int(np.sum(wrong_mask[train_idx])),
            'n_wrong_val': int(np.sum(wrong_mask[val_idx])),
            'n_wrong_test': int(np.sum(wrong_mask[test_idx])),
            **{k: rr[k] for k in rr.keys() if k not in {'base_train', 'base_val', 'base_test', 'train', 'val', 'test'}},
            **out['metrics'],
        })

        for i in wrong_idx.tolist():
            split = 'train' if i in set(train_idx.tolist()) else 'val' if i in set(val_idx.tolist()) else 'test' if i in set(test_idx.tolist()) else 'other'
            ninfo = build_neighbor_stats(A_sym, y, i, classes)
            finfo = nonzero_feature_info(X_raw[i], max_keep=args.max_feature_idx_keep)
            true_pos = int(np.where(classes == y[i])[0][0])
            pred_pos = int(np.where(classes == pred[i])[0][0])
            base_pos = int(np.where(classes == base_pred[i])[0][0])
            error_rows.append({
                'repeat': repeat,
                'grid_id': grid_id,
                'split': split,
                'node_id': int(i),
                'true_label': int(y[i]),
                'pred_label': int(pred[i]),
                'base_pred_label': int(base_pred[i]),
                'true_score_final': float(out['score_final'][i, true_pos]),
                'pred_score_final': float(out['score_final'][i, pred_pos]),
                'true_score_base': float(out['S_base'][i, true_pos]),
                'pred_score_base': float(out['S_base'][i, pred_pos]),
                'true_score_sub': float(out['S_sub_n'][i, true_pos]),
                'pred_score_sub': float(out['S_sub_n'][i, pred_pos]),
                'true_score_ridge': float(out['S_ridge_n'][i, true_pos]),
                'pred_score_ridge': float(out['S_ridge_n'][i, pred_pos]),
                'true_score_geom': float(out['S_geom'][i, true_pos]),
                'pred_score_geom': float(out['S_geom'][i, pred_pos]),
                'pred_prob_final': float(out['z'][i, pred_pos]),
                'true_prob_final': float(out['z'][i, true_pos]),
                'gate_mean': float(np.mean(out['gate'])),
                'gate_true_class': float(out['gate'][true_pos]),
                'gate_pred_class': float(out['gate'][pred_pos]),
                **ninfo,
                **finfo,
                # keep config columns for later filtering
                'preprocess_mode': rr.get('preprocess_mode', ''),
                'feature_mode': rr.get('feature_mode', ''),
                'rnr_variant': rr.get('rnr_variant', ''),
                'relations': rr.get('relations', ''),
                'dim': rr.get('dim', ''),
                'lambda_smooth': rr.get('lambda_smooth', ''),
                'ridge_alpha': rr.get('ridge_alpha', ''),
                'ridge_temperature': rr.get('ridge_temperature', ''),
                'residual_temperature': rr.get('residual_temperature', ''),
                'alpha_sub': rr.get('alpha_sub', ''),
                'beta_ridge': rr.get('beta_ridge', ''),
                'gamma_geom': rr.get('gamma_geom', ''),
                'lambda_compat': rr.get('lambda_compat', ''),
                'iterations': rr.get('iterations', ''),
                'local_rank': rr.get('local_rank', ''),
            })

        print(f"[audit] repeat={repeat} grid={grid_id} wrong={len(wrong_idx)} val={out['metrics']['val']:.4f} test={out['metrics']['test']:.4f}")

    write_rows_csv(out_dir / 'error_nodes_by_grid.csv', error_rows)
    write_rows_csv(out_dir / 'grid_summary_with_error_counts.csv', grid_summary_rows)

    # Aggregate per-node summary
    node_group = defaultdict(list)
    for r in error_rows:
        node_group[int(r['node_id'])].append(r)
    node_rows = []
    total_configs = len(grid_summary_rows)
    for nid, group in node_group.items():
        true_label = group[0]['true_label']
        splits = Counter(g['split'] for g in group)
        pred_counts = Counter(g['pred_label'] for g in group)
        node_rows.append({
            'node_id': nid,
            'true_label': true_label,
            'error_count': len(group),
            'total_configs': total_configs,
            'error_rate': len(group) / max(total_configs, 1),
            'most_common_wrong_pred': pred_counts.most_common(1)[0][0],
            'split_mode': splits.most_common(1)[0][0],
            'degree_sym': group[0]['degree_sym'],
            'same_label_neighbor_ratio': group[0]['same_label_neighbor_ratio'],
            'neighbor_label_entropy': group[0]['neighbor_label_entropy'],
            'raw_feat_nnz': group[0]['raw_feat_nnz'],
            'raw_feat_nz_idx_head': group[0]['raw_feat_nz_idx_head'],
            'grid_ids': json.dumps(sorted({int(g['grid_id']) for g in group}), ensure_ascii=False),
        })
    node_rows = sorted(node_rows, key=lambda r: (-float(r['error_rate']), -int(r['error_count']), int(r['node_id'])))
    write_rows_csv(out_dir / 'error_node_summary.csv', node_rows)

    # Aggregate per parameter value effects
    param_names = ['lambda_smooth', 'ridge_alpha', 'alpha_sub', 'beta_ridge', 'gamma_geom', 'lambda_compat', 'dim', 'local_rank', 'preprocess_mode', 'feature_mode', 'rnr_variant']
    param_effect_rows = []
    for p in param_names:
        buckets = defaultdict(list)
        for r in grid_summary_rows:
            buckets[str(r[p])].append(r)
        for val, group in buckets.items():
            param_effect_rows.append({
                'parameter': p,
                'value': val,
                'n_configs': len(group),
                'mean_val': float(np.mean([float(g['val']) for g in group])),
                'mean_test': float(np.mean([float(g['test']) for g in group])),
                'mean_n_wrong_test': float(np.mean([float(g['n_wrong_test']) for g in group])),
                'mean_n_wrong_all': float(np.mean([float(g['n_wrong_all']) for g in group])),
            })
    write_rows_csv(out_dir / 'param_effects_summary.csv', param_effect_rows)

    # Aggregate per-node per-parameter error rates
    node_param_rows = []
    # configs_by_paramvalue counts
    configs_by_paramvalue = defaultdict(int)
    for p in param_names:
        for r in grid_summary_rows:
            configs_by_paramvalue[(p, str(r[p]))] += 1
    errors_by_node_param = defaultdict(int)
    template_by_node = {}
    for r in error_rows:
        nid = int(r['node_id'])
        template_by_node[nid] = r
        for p in param_names:
            errors_by_node_param[(nid, p, str(r[p]))] += 1
    for (nid, p, val), cnt in errors_by_node_param.items():
        tpl = template_by_node[nid]
        total = configs_by_paramvalue[(p, val)]
        node_param_rows.append({
            'node_id': nid,
            'true_label': tpl['true_label'],
            'parameter': p,
            'value': val,
            'error_count': cnt,
            'total_configs_with_value': total,
            'error_rate': cnt / max(total, 1),
            'degree_sym': tpl['degree_sym'],
            'same_label_neighbor_ratio': tpl['same_label_neighbor_ratio'],
            'neighbor_label_entropy': tpl['neighbor_label_entropy'],
            'raw_feat_nnz': tpl['raw_feat_nnz'],
        })
    write_rows_csv(out_dir / 'node_param_effects.csv', node_param_rows)

    with open(out_dir / 'audit_meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'src_v8': str(src_v8),
            'algo_py': str(algo_py),
            'results_csv': str(results_csv),
            'data_root': str(data_root),
            'selected_configs': len(selected),
            'repeats': parse_int_list(args.repeats),
        }, f, ensure_ascii=False, indent=2)

    print('Done.')
    print('Saved:')
    print(' -', out_dir / 'error_nodes_by_grid.csv')
    print(' -', out_dir / 'error_node_summary.csv')
    print(' -', out_dir / 'param_effects_summary.csv')
    print(' -', out_dir / 'node_param_effects.csv')
    print(' -', out_dir / 'grid_summary_with_error_counts.csv')


if __name__ == '__main__':
    main()


