import os
import sys
import time
import csv
import copy
from pathlib import Path

import numpy as np
from scipy import sparse

# 当前目录：scripts\v6_layer3_fast
THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR.parent

# 让脚本能找到上一层 scripts 里的 fair_utils.py
sys.path.insert(0, str(SCRIPTS_DIR))

from fair_utils import load_extended_dataset, accuracy_from_pred

SRC_V6 = Path(r"D:\桌面\MSR实验复现与创新\experiments_g1\claude_whitebox_g1_v2\src_v6")
DATA_ROOT = Path(r"D:\桌面\MSR实验复现与创新\planetoid\data")
OUT_DIR = SCRIPTS_DIR / "results_v6_layer3_direct_fast"

sys.path.insert(0, str(SRC_V6))

try:
    from ha_sdc import HASDC
except ModuleNotFoundError:
    from model import HASDC


DATASETS = [
    "chameleon",
    "wisconsin",
]

REPEATS = [0, 1, 2]

# 固定 base 参数
FIXED_BASE = {
    "chameleon": {
        "lambda_smooth": 0.1,
        "d_s": 8,
        "d_r": 8,
        "tau_gate": 5.0,
    },
    "wisconsin": {
        "lambda_smooth": 0.0,
        "d_s": 8,
        "d_r": 8,
        "tau_gate": 5.0,
    },
}

# 只调 Layer3
ALPHA_MODES = [
    "zero",
    "learned",
    "force_pos",
    "force_neg",
]

GAMMAS = [
    0.0,
    100.0,
    1000.0,
    10000.0,
]


def edge_index_to_scipy_adj(edge_index, num_nodes):
    edge_index_np = edge_index.cpu().numpy()
    row = edge_index_np[0]
    col = edge_index_np[1]
    data = np.ones(row.shape[0], dtype=np.float64)
    A = sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def set_alpha_mode(model, mode):
    state = model.state_

    if state is None:
        raise RuntimeError("model.state_ is None. Did you call fit()?")

    old_alpha = state.alpha.copy()
    overlap = state.overlap.copy()

    if mode == "learned":
        new_alpha = old_alpha

    elif mode == "zero":
        new_alpha = np.zeros_like(old_alpha)

    elif mode == "force_pos":
        new_alpha = overlap.copy()
        np.fill_diagonal(new_alpha, 0.0)

    elif mode == "force_neg":
        new_alpha = -overlap.copy()
        np.fill_diagonal(new_alpha, 0.0)

    else:
        raise ValueError(f"Unknown alpha mode: {mode}")

    state.alpha = new_alpha
    model.state_ = state

    alpha_nonzero = int(np.sum(np.abs(new_alpha) > 1e-12))
    alpha_abs_sum = float(np.sum(np.abs(new_alpha)))
    overlap_abs_sum = float(np.sum(np.abs(overlap)))
    overlap_max = float(np.max(np.abs(overlap))) if overlap.size > 0 else 0.0

    return {
        "alpha_nonzero": alpha_nonzero,
        "alpha_abs_sum": alpha_abs_sum,
        "overlap_abs_sum": overlap_abs_sum,
        "overlap_max": overlap_max,
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("[saved]", path)


def summarize(rows):
    groups = {}

    for row in rows:
        key = (
            row["dataset"],
            row["alpha_mode"],
            row["gamma"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, alpha_mode, gamma), rs in groups.items():
        vals = np.array([float(r["val_acc"]) for r in rs], dtype=float)
        tests = np.array([float(r["test_acc"]) for r in rs], dtype=float)

        summary.append({
            "dataset": dataset,
            "alpha_mode": alpha_mode,
            "gamma": gamma,
            "n": len(rs),
            "val_mean": float(vals.mean()),
            "val_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "test_mean": float(tests.mean()),
            "test_std": float(tests.std(ddof=1)) if len(tests) > 1 else 0.0,
        })

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for ds in DATASETS:
        base_cfg = FIXED_BASE[ds]

        for repeat in REPEATS:
            print("=" * 100)
            print(f"[load] dataset={ds} repeat={repeat}")

            graph = load_extended_dataset(
                ds,
                data_root=DATA_ROOT,
                repeat=repeat,
                train_per_class=20,
                val_per_class=30,
                feature_norm="row_l1",
                make_undirected=True,
            )

            A = edge_index_to_scipy_adj(graph.edge_index, graph.num_nodes)
            X = graph.x.cpu().numpy().astype(np.float64)
            y = graph.y.cpu().numpy()
            train_idx = graph.train_idx.cpu().numpy()
            val_idx = graph.val_idx.cpu().numpy()
            test_idx = graph.test_idx.cpu().numpy()

            print(
                f"nodes={graph.num_nodes}, features={graph.num_features}, classes={graph.num_classes}, "
                f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

            # 只 fit 一次，gamma 先设 0；后面直接改 model.gamma 和 alpha
            print(
                f"[fit base] lambda={base_cfg['lambda_smooth']} "
                f"d={base_cfg['d_s']}:{base_cfg['d_r']} tau={base_cfg['tau_gate']}"
            )

            t0 = time.time()
            model = HASDC(
                lambda_smooth=base_cfg["lambda_smooth"],
                d_s=base_cfg["d_s"],
                d_r=base_cfg["d_r"],
                tau_gate=base_cfg["tau_gate"],
                gamma=0.0,
                laplacian="normalized",
                add_self_loops=False,
                alpha_min_overlap=1e-8,
                alpha_min_acc_delta=0.0,
                residual_floor=0.0,
            )

            model.fit(A, X, y, train_idx, val_idx)
            fit_time = time.time() - t0

            original_state = copy.deepcopy(model.state_)

            for alpha_mode in ALPHA_MODES:
                for gamma in GAMMAS:
                    model.state_ = copy.deepcopy(original_state)
                    model.gamma = float(gamma)

                    alpha_info = set_alpha_mode(model, alpha_mode)

                    pred = model.predict(A, X)

                    val_acc = accuracy_from_pred(pred, graph.y, graph.val_idx)
                    test_acc = accuracy_from_pred(pred, graph.y, graph.test_idx)

                    row = {
                        "dataset": ds,
                        "repeat": repeat,
                        "split_id": graph.split_id,
                        "seed": graph.seed,
                        "alpha_mode": alpha_mode,
                        "gamma": gamma,
                        "lambda_smooth": base_cfg["lambda_smooth"],
                        "d_s": base_cfg["d_s"],
                        "d_r": base_cfg["d_r"],
                        "tau_gate": base_cfg["tau_gate"],
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "fit_time": round(fit_time, 3),
                        **alpha_info,
                    }

                    all_rows.append(row)

                    print(
                        f"[v6-L3-direct] ds={ds} rep={repeat} mode={alpha_mode} gamma={gamma} "
                        f"val={val_acc:.4f} test={test_acc:.4f} "
                        f"alpha_nz={alpha_info['alpha_nonzero']} "
                        f"alpha_sum={alpha_info['alpha_abs_sum']:.4e} "
                        f"overlap_max={alpha_info['overlap_max']:.4e}"
                    )

                    write_rows(OUT_DIR / "v6_layer3_direct_fast_all_partial.csv", all_rows)

    write_rows(OUT_DIR / "v6_layer3_direct_fast_all.csv", all_rows)

    summary = summarize(all_rows)
    write_rows(OUT_DIR / "v6_layer3_direct_fast_summary.csv", summary)

    print("=" * 100)
    print("完成。结果在：", OUT_DIR)


if __name__ == "__main__":
    main()