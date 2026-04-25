from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import sparse

THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fair_utils import (  # noqa: E402
    accuracy_from_pred,
    load_extended_dataset,
    normalize_dataset_name,
    parse_csv_list,
    parse_float_grid,
    parse_int_list,
    set_seed,
    summarize_csv,
    write_csv_rows,
)


def import_hasdc(src_v6: str | Path):
    src = Path(src_v6).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"找不到 src_v6 路径：{src}")

    # package import: parent/src_v6/__init__.py
    if (src / "__init__.py").exists():
        parent = src.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        try:
            import importlib
            mod = importlib.import_module(src.name)
            return mod.HASDC
        except Exception:
            pass

    # direct __init__.py import with package search location
    if (src / "__init__.py").exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ha_sdc_runtime_fixed_l3",
            str(src / "__init__.py"),
            submodule_search_locations=[str(src)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {src / '__init__.py'}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ha_sdc_runtime_fixed_l3"] = mod
        spec.loader.exec_module(mod)
        return mod.HASDC

    model_py = src / "model.py"
    if model_py.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("ha_sdc_model_runtime_fixed_l3", str(model_py))
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {model_py}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ha_sdc_model_runtime_fixed_l3"] = mod
        spec.loader.exec_module(mod)
        return mod.HASDC

    raise FileNotFoundError(f"src_v6 中没有找到 __init__.py 或 model.py：{src}")


def edge_index_to_scipy(edge_index: torch.Tensor, num_nodes: int) -> sparse.csr_matrix:
    ei = edge_index.detach().cpu().numpy()
    row = ei[0].astype(np.int64)
    col = ei[1].astype(np.int64)
    data = np.ones(row.shape[0], dtype=np.float64)
    A = sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A


def get_fixed_base(ds: str) -> Tuple[float, int, int, float]:
    """
    固定非 Layer3 参数。
    这些数据集之前 smooth+Layer2 表现不好；异配图上优先降低/关闭平滑。
    返回: lambda_smooth, d_s, d_r, tau_gate
    """
    ds = normalize_dataset_name(ds)
    table: Dict[str, Tuple[float, int, int, float]] = {
        # Actor 上平滑/子空间整体弱，先用 raw-dominant 基线。
        "actor": (0.0, 8, 8, 5.0),
        # 你当前日志显示 Chameleon: lam=0.1,d=8 val=0.5034，比 lam=0 或 1 更好。
        "chameleon": (0.1, 8, 8, 5.0),
        # Squirrel/网页类小图倾向异配，先关闭平滑看 Layer3 是否能从 raw crosslink 获益。
        "squirrel": (0.0, 8, 8, 5.0),
        "cornell": (0.0, 8, 8, 5.0),
        "texas": (0.0, 8, 8, 5.0),
        "wisconsin": (0.0, 8, 8, 5.0),
    }
    return table.get(ds, (0.0, 8, 8, 5.0))


def set_alpha_mode(model, mode: str) -> dict:
    st = model.get_state()
    overlap = np.asarray(st.overlap, dtype=float)
    learned = np.asarray(st.alpha, dtype=float).copy()
    offdiag = np.ones_like(overlap, dtype=bool)
    np.fill_diagonal(offdiag, False)

    if mode == "zero":
        alpha = np.zeros_like(overlap)
    elif mode == "learned":
        alpha = learned.copy()
    elif mode == "force_pos":
        alpha = overlap.copy()
        alpha[~offdiag] = 0.0
    elif mode == "force_neg":
        alpha = -overlap.copy()
        alpha[~offdiag] = 0.0
    else:
        raise ValueError(f"unknown alpha_mode={mode}")

    st.alpha = alpha
    return {
        "alpha_mode": mode,
        "alpha_nonzero": int(np.sum(np.abs(alpha) > 1e-12)),
        "learned_alpha_nonzero": int(np.sum(np.abs(learned) > 1e-12)),
        "alpha_abs_sum": float(np.sum(np.abs(alpha))),
        "overlap_abs_sum": float(np.sum(np.abs(overlap))),
        "overlap_max": float(np.max(overlap)) if overlap.size else 0.0,
    }


def eval_model(model, A, X, graph) -> dict:
    pred_np = model.predict(A, X)
    pred = torch.as_tensor(pred_np, dtype=graph.y.dtype)
    return {
        "train_acc": accuracy_from_pred(pred, graph.y.cpu(), graph.train_idx.cpu()),
        "val_acc": accuracy_from_pred(pred, graph.y.cpu(), graph.val_idx.cpu()),
        "test_acc": accuracy_from_pred(pred, graph.y.cpu(), graph.test_idx.cpu()),
    }


def diagnostic_stats(model, A, X) -> dict:
    try:
        st = model.get_state()
        X_np = np.asarray(X, dtype=float)
        A_sp = model._as_square_sparse_adjacency(A, n=X_np.shape[0])
        Z_np = model._smooth_features(A_sp, X_np)
        residual_s, _ = model._residuals_and_coords(Z_np, st.smooth_subspaces, st.classes)
        residual_r, coords_r = model._residuals_and_coords(X_np, st.raw_subspaces, st.classes)
        base = model._base_residual(residual_s, residual_r, st.classes, st.gate)
        cross = model._crosslink_modulation(coords_r, st.smooth_subspaces, st.raw_subspaces, st.classes, st.alpha)
        final = base - model.gamma * cross
        gates = np.asarray(list(st.gate.values()), dtype=float)
        return {
            "gate_mean": float(np.mean(gates)) if gates.size else None,
            "gate_min": float(np.min(gates)) if gates.size else None,
            "gate_max": float(np.max(gates)) if gates.size else None,
            "base_abs_mean": float(np.mean(np.abs(base))),
            "cross_abs_mean": float(np.mean(np.abs(cross))),
            "gamma_cross_abs_mean": float(model.gamma * np.mean(np.abs(cross))),
            "final_abs_mean": float(np.mean(np.abs(final))),
        }
    except Exception as e:
        return {"diagnostic_error": repr(e)}


def select_best_by_val(rows: List[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise ValueError("没有成功结果可以选。")
    return dict(max(ok, key=lambda r: (float(r.get("val_acc", -1)), -float(r.get("time_sec", 1e18)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_v6", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results_v6_layer3_fixed_base_fast")
    ap.add_argument("--datasets", type=str, default="actor,chameleon,squirrel,cornell,texas,wisconsin")
    ap.add_argument("--repeats", type=str, default="0,1,2,3,4")
    ap.add_argument("--explicit_split_id", type=int, default=None)
    ap.add_argument("--feature_norm", type=str, default="standard", choices=["none", "row_l1", "standard"])
    ap.add_argument("--train_per_class", type=int, default=20)
    ap.add_argument("--val_per_class", type=int, default=30)
    ap.add_argument("--alpha_modes", type=str, default="zero,learned,force_pos,force_neg")
    # 这里是唯一主要 Layer3 扫描项。cross 很小，所以加入 10/100/1000 观察是否能真正改变决策。
    ap.add_argument("--gamma_grid", type=str, default="0,1,10,100,1000")
    ap.add_argument("--alpha_min_overlap", type=float, default=0.0)
    ap.add_argument("--alpha_min_acc_delta", type=float, default=0.0)
    ap.add_argument("--laplacian", type=str, default="normalized", choices=["normalized", "combinatorial"])
    ap.add_argument("--add_self_loops", action="store_true")
    ap.add_argument("--residual_floor", type=float, default=0.0)
    args = ap.parse_args()

    HASDC = import_hasdc(args.src_v6)
    datasets = [normalize_dataset_name(x) for x in parse_csv_list(args.datasets, [])]
    repeats = parse_int_list(args.repeats, list(range(5)))
    alpha_modes = parse_csv_list(args.alpha_modes, ["zero", "learned", "force_pos", "force_neg"])
    gamma_grid = parse_float_grid(args.gamma_grid, [0.0, 1.0, 10.0, 100.0])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows_all: List[dict] = []
    rows_best: List[dict] = []

    print("[src_v6]", Path(args.src_v6).resolve())
    print("[data]", Path(args.data_root).resolve())
    print("[out]", out.resolve())
    print("[datasets]", datasets)
    print("[repeats]", repeats)
    print("[alpha_modes]", alpha_modes)
    print("[gamma_grid]", gamma_grid)

    for ds in datasets:
        lam, d_s, d_r, tau = get_fixed_base(ds)
        print(f"\n[fixed-base] ds={ds} lambda={lam} d={d_s}:{d_r} tau={tau}")
        for repeat in repeats:
            set_seed(repeat)
            print("\n" + "=" * 100)
            print(f"[load] ds={ds} repeat={repeat}")
            try:
                graph = load_extended_dataset(
                    ds,
                    data_root=args.data_root,
                    repeat=repeat,
                    explicit_split_id=args.explicit_split_id,
                    train_per_class=args.train_per_class,
                    val_per_class=args.val_per_class,
                    feature_norm=args.feature_norm,
                    make_undirected=True,
                )
            except Exception as e:
                row = {"dataset": ds, "repeat": repeat, "error": f"load_failed: {e!r}"}
                rows_all.append(row)
                write_csv_rows(out / "v6_layer3_fixed_base_all_partial.csv", rows_all)
                print("LOAD ERROR:", repr(e))
                continue

            X = graph.x.detach().cpu().numpy().astype(np.float64, copy=False)
            y = graph.y.detach().cpu().numpy()
            A = edge_index_to_scipy(graph.edge_index, graph.num_nodes)
            train_idx = graph.train_idx.detach().cpu().numpy().astype(np.int64)
            val_idx = graph.val_idx.detach().cpu().numpy().astype(np.int64)

            # 只 fit 一次：固定非 Layer3 参数。随后只切换 gamma 和 alpha。
            fit_cfg = {
                "lambda_smooth": lam,
                "d_s": d_s,
                "d_r": d_r,
                "tau_gate": tau,
                "gamma": 1.0,
                "laplacian": args.laplacian,
                "add_self_loops": bool(args.add_self_loops),
                "alpha_min_overlap": args.alpha_min_overlap,
                "alpha_min_acc_delta": args.alpha_min_acc_delta,
                "residual_floor": args.residual_floor,
            }
            t_fit = time.time()
            try:
                model = HASDC(**fit_cfg)
                model.fit(A, X, y, train_idx, val_idx)
                fit_time = round(time.time() - t_fit, 3)
            except Exception as e:
                row = {"dataset": graph.name, "repeat": repeat, "error": f"fit_failed: {e!r}", **fit_cfg}
                rows_all.append(row)
                write_csv_rows(out / "v6_layer3_fixed_base_all_partial.csv", rows_all)
                print("FIT ERROR:", repr(e))
                continue

            grid_rows: List[dict] = []
            grid_id = 0
            for mode in alpha_modes:
                for gamma in gamma_grid:
                    t0 = time.time()
                    row = {
                        "dataset": graph.name,
                        "repeat": repeat,
                        "split_id": graph.split_id,
                        "seed": graph.seed,
                        "split_policy": getattr(graph, "split_policy", ""),
                        "method": "v6_HASDC_fixed_base_layer3_fast",
                        "grid_id": grid_id,
                        "num_nodes": graph.num_nodes,
                        "num_features": graph.num_features,
                        "num_classes": graph.num_classes,
                        "train_size": int(graph.train_idx.numel()),
                        "val_size": int(graph.val_idx.numel()),
                        "test_size": int(graph.test_idx.numel()),
                        "lambda_smooth": lam,
                        "d_s": d_s,
                        "d_r": d_r,
                        "tau_gate": tau,
                        "gamma": gamma,
                        "fit_time_sec": fit_time,
                        "selection_rule": "fixed_base_then_layer3_grid_select_by_val_only_no_test",
                    }
                    try:
                        model.gamma = float(gamma)
                        row.update(set_alpha_mode(model, mode))
                        row.update(diagnostic_stats(model, A, X))
                        row.update(eval_model(model, A, X, graph))
                        row["time_sec"] = round(time.time() - t0 + fit_time, 3)
                        print(
                            f"[v6-fixed-L3] ds={ds} rep={repeat} grid={grid_id} mode={mode} gamma={gamma} "
                            f"val={row['val_acc']:.4f} test={row['test_acc']:.4f} "
                            f"alpha_nz={row.get('alpha_nonzero')} gx={row.get('gamma_cross_abs_mean')}"
                        )
                    except Exception as e:
                        row["error"] = repr(e)
                        print("ERROR:", repr(e))

                    rows_all.append(row)
                    grid_rows.append(row)
                    write_csv_rows(out / "v6_layer3_fixed_base_all_partial.csv", rows_all)
                    grid_id += 1

            try:
                best = select_best_by_val(grid_rows)
                rows_best.append(best)
                print(
                    f"[BEST] ds={ds} repeat={repeat} val={best['val_acc']:.4f} test={best['test_acc']:.4f} "
                    f"mode={best.get('alpha_mode')} gamma={best.get('gamma')}"
                )
            except Exception as e:
                rows_best.append({"dataset": graph.name, "repeat": repeat, "error": f"selection_failed: {e!r}"})
            write_csv_rows(out / "v6_layer3_fixed_base_best_partial.csv", rows_best)

    all_path = out / "v6_layer3_fixed_base_all.csv"
    best_path = out / "v6_layer3_fixed_base_best_by_val.csv"
    summary_path = out / "v6_layer3_fixed_base_summary.csv"
    write_csv_rows(all_path, rows_all)
    write_csv_rows(best_path, rows_best)
    summarize_csv(best_path, summary_path, group_cols=["dataset", "method"])
    print("\nDone.")
    print("all:", all_path.resolve())
    print("best:", best_path.resolve())
    print("summary:", summary_path.resolve())


if __name__ == "__main__":
    main()
