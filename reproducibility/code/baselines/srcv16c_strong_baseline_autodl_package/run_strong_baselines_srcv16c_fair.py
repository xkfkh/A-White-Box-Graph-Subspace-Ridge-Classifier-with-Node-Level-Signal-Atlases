# -*- coding: utf-8 -*-
"""
run_strong_baselines_srcv16c_fair.py

Purpose
-------
Run only the strongest baseline method for each SRC-v16c final-table dataset,
using exactly the same split protocol recorded in SRC_v16c_final_results_no_squirrel.xlsx:

- amazon-computers: GraphSAGE, class-balanced random 20 train + 30 val per class,
  split seed = 20260419 + repeat.
- amazon-photo: GraphSAGE, class-balanced random 20 train + 30 val per class,
  split seed = 20260419 + repeat.
- chameleon: LINKX, fixed Geom-GCN split id = repeat, for repeats 0..9.
- cornell: GraphSAGE, random stratified split matching PyG/WebKB fixed split-0
  class counts, split seed = 20260419 + repeat.
- texas: GraphSAGE, same as cornell.
- wisconsin: GraphSAGE, same as cornell.

No squirrel is run.

Anti-leakage rules
------------------
1. The test mask is never used for training, early stopping, or hyperparameter selection.
2. Epoch/model-state selection uses validation accuracy only.
3. If a small hyperparameter grid is used, best row is selected by validation accuracy only.
4. All split sizes, seeds, and policies are written to CSV for audit.

Run examples
------------
python run_strong_baselines_srcv16c_fair.py \
  --data_root /root/autodl-tmp/planetoid/data \
  --dataset amazon-computers --repeat 0 --out_dir results_strong_baseline_cpu

python run_strong_baselines_srcv16c_fair.py \
  --data_root /root/autodl-tmp/planetoid/data \
  --dataset all --repeats 0,1,2,3,4,5,6,7,8,9 --out_dir results_strong_baseline_cpu
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DATASET_TO_METHOD: Dict[str, str] = {
    "amazon-computers": "graphsage",
    "amazon-photo": "graphsage",
    "chameleon": "linkx",
    "cornell": "graphsage",
    "texas": "graphsage",
    "wisconsin": "graphsage",
}

FINAL_DATASETS: Tuple[str, ...] = tuple(DATASET_TO_METHOD.keys())


@dataclass
class GraphPack:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    split_policy: str
    split_id: Optional[int]
    split_seed: int
    train_seed: int

    @property
    def num_nodes(self) -> int:
        return int(self.y.numel())

    @property
    def num_features(self) -> int:
        return int(self.x.size(-1))

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item() + 1)


class GraphSAGENet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.dropout = float(dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def normalize_dataset_name(name: str) -> str:
    s = name.strip().lower().replace("_", "-")
    aliases = {
        "computers": "amazon-computers",
        "amazoncomputers": "amazon-computers",
        "amazon-computer": "amazon-computers",
        "photo": "amazon-photo",
        "amazonphoto": "amazon-photo",
        "chameleon-filtered": "chameleon",
    }
    return aliases.get(s, s)


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_grid(text: str, default: Sequence[float]) -> List[float]:
    vals = parse_csv_list(text)
    return [float(x) for x in vals] if vals else list(default)


def parse_int_grid(text: str, default: Sequence[int]) -> List[int]:
    vals = parse_csv_list(text)
    return [int(x) for x in vals] if vals else list(default)


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_1d_mask(mask: torch.Tensor, split_id: int) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.dim() == 1:
        return mask
    if mask.dim() == 2:
        if split_id < 0 or split_id >= mask.size(1):
            raise ValueError(f"split_id={split_id} out of range for mask shape={tuple(mask.shape)}")
        return mask[:, split_id]
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


def mask_to_idx(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).view(-1).long()


def split_seed_for_repeat(repeat: int) -> int:
    return 20260419 + int(repeat)


def row_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    denom = x.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return x / denom


def standard_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-12)
    return (x - mean) / std


def apply_feature_norm(x: torch.Tensor, feature_norm: str) -> torch.Tensor:
    feature_norm = feature_norm.lower()
    if feature_norm == "none":
        return x.float()
    if feature_norm == "row_l1":
        return row_l1_normalize(x)
    if feature_norm == "standard":
        return standard_normalize(x)
    raise ValueError(f"Unknown feature_norm={feature_norm}")


def class_balanced_random_split(
    y: torch.Tensor,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_cpu = y.detach().cpu().long()
    num_nodes = int(y_cpu.numel())
    train_parts: List[torch.Tensor] = []
    val_parts: List[torch.Tensor] = []
    test_mask = torch.ones(num_nodes, dtype=torch.bool)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    for c in range(int(y_cpu.max().item()) + 1):
        cls_idx = (y_cpu == c).nonzero(as_tuple=False).view(-1)
        need = int(train_per_class) + int(val_per_class)
        if cls_idx.numel() < need:
            raise ValueError(
                f"Class {c} has only {cls_idx.numel()} nodes, but needs {need} "
                f"for train_per_class={train_per_class}, val_per_class={val_per_class}."
            )
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=gen)]
        tr = perm[:train_per_class]
        va = perm[train_per_class:train_per_class + val_per_class]
        train_parts.append(tr)
        val_parts.append(va)
        test_mask[tr] = False
        test_mask[va] = False

    train_idx = torch.cat(train_parts).long()
    val_idx = torch.cat(val_parts).long()
    test_idx = mask_to_idx(test_mask)
    return train_idx, val_idx, test_idx


def random_stratified_matching_split0_counts(
    y: torch.Tensor,
    fixed_train_mask0: torch.Tensor,
    fixed_val_mask0: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[int]]]:
    y_cpu = y.detach().cpu().long()
    fixed_train_mask0 = fixed_train_mask0.detach().cpu().bool()
    fixed_val_mask0 = fixed_val_mask0.detach().cpu().bool()
    num_nodes = int(y_cpu.numel())
    num_classes = int(y_cpu.max().item()) + 1
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    train_parts: List[torch.Tensor] = []
    val_parts: List[torch.Tensor] = []
    test_mask = torch.ones(num_nodes, dtype=torch.bool)
    train_counts: List[int] = []
    val_counts: List[int] = []

    for c in range(num_classes):
        cls_idx = (y_cpu == c).nonzero(as_tuple=False).view(-1)
        n_train_c = int(((y_cpu == c) & fixed_train_mask0).sum().item())
        n_val_c = int(((y_cpu == c) & fixed_val_mask0).sum().item())
        need = n_train_c + n_val_c
        if cls_idx.numel() < need:
            raise ValueError(f"Class {c} has {cls_idx.numel()} nodes but needs {need}.")
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=gen)]
        tr = perm[:n_train_c]
        va = perm[n_train_c:n_train_c + n_val_c]
        train_parts.append(tr)
        val_parts.append(va)
        train_counts.append(n_train_c)
        val_counts.append(n_val_c)
        test_mask[tr] = False
        test_mask[va] = False

    meta = {"train_counts_per_class": train_counts, "val_counts_per_class": val_counts}
    return torch.cat(train_parts).long(), torch.cat(val_parts).long(), mask_to_idx(test_mask), meta


def check_no_overlap(train_idx: torch.Tensor, val_idx: torch.Tensor, test_idx: torch.Tensor, num_nodes: int) -> None:
    sets = [set(train_idx.cpu().tolist()), set(val_idx.cpu().tolist()), set(test_idx.cpu().tolist())]
    names = ["train", "val", "test"]
    for i in range(3):
        for j in range(i + 1, 3):
            inter = sets[i].intersection(sets[j])
            if inter:
                raise AssertionError(f"Split leakage: {names[i]} and {names[j]} overlap, e.g. {list(inter)[:5]}")
    total = len(sets[0] | sets[1] | sets[2])
    if total != int(num_nodes):
        raise AssertionError(f"Split does not cover all nodes: covered={total}, num_nodes={num_nodes}")


def load_dataset_for_split(
    dataset_name: str,
    data_root: str,
    repeat: int,
    feature_norm: str,
    train_per_class: int,
    val_per_class: int,
    make_undirected: bool,
) -> GraphPack:
    from torch_geometric.utils import to_undirected
    from torch_geometric.datasets import Amazon, WebKB, WikipediaNetwork

    name = normalize_dataset_name(dataset_name)
    repeat = int(repeat)
    root = str(data_root)

    if name == "amazon-computers":
        ds = Amazon(root=root, name="Computers")
        data = ds[0]
        split_seed = split_seed_for_repeat(repeat)
        train_idx, val_idx, test_idx = class_balanced_random_split(
            data.y, train_per_class=train_per_class, val_per_class=val_per_class, seed=split_seed
        )
        split_policy = f"class_balanced_random_{train_per_class}_train_{val_per_class}_val_per_class"
        split_id = 0
    elif name == "amazon-photo":
        ds = Amazon(root=root, name="Photo")
        data = ds[0]
        split_seed = split_seed_for_repeat(repeat)
        train_idx, val_idx, test_idx = class_balanced_random_split(
            data.y, train_per_class=train_per_class, val_per_class=val_per_class, seed=split_seed
        )
        split_policy = f"class_balanced_random_{train_per_class}_train_{val_per_class}_val_per_class"
        split_id = 0
    elif name == "chameleon":
        ds = WikipediaNetwork(root=root, name="chameleon", geom_gcn_preprocess=True)
        data = ds[0]
        split_id = repeat
        train_mask = ensure_1d_mask(data.train_mask, split_id)
        val_mask = ensure_1d_mask(data.val_mask, split_id)
        test_mask = ensure_1d_mask(data.test_mask, split_id)
        train_idx, val_idx, test_idx = mask_to_idx(train_mask), mask_to_idx(val_mask), mask_to_idx(test_mask)
        split_seed = repeat
        split_policy = f"fixed_geom_gcn_split_id_{split_id}"
    elif name in {"cornell", "texas", "wisconsin"}:
        py_name = {"cornell": "Cornell", "texas": "Texas", "wisconsin": "Wisconsin"}[name]
        ds = WebKB(root=root, name=py_name)
        data = ds[0]
        fixed_train0 = ensure_1d_mask(data.train_mask, 0)
        fixed_val0 = ensure_1d_mask(data.val_mask, 0)
        split_seed = split_seed_for_repeat(repeat)
        train_idx, val_idx, test_idx, _ = random_stratified_matching_split0_counts(
            data.y, fixed_train0, fixed_val0, seed=split_seed
        )
        split_policy = "random_stratified_matching_fixed_split0_class_counts"
        split_id = 0
    else:
        raise ValueError(f"Dataset {dataset_name!r} is not in final SRC-v16c table. Allowed: {FINAL_DATASETS}")

    x = apply_feature_norm(data.x, feature_norm)
    edge_index = data.edge_index.long()
    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=int(data.num_nodes))

    check_no_overlap(train_idx, val_idx, test_idx, int(data.num_nodes))
    train_seed = int(split_seed) + 12345

    return GraphPack(
        name=name,
        x=x,
        y=data.y.long(),
        edge_index=edge_index,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_policy=split_policy,
        split_id=split_id,
        split_seed=int(split_seed),
        train_seed=int(train_seed),
    )


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return float("nan")
    pred = logits[idx].argmax(dim=-1)
    return float((pred == y[idx]).float().mean().item())


def build_model(method: str, graph: GraphPack, hidden_channels: int, dropout: float) -> nn.Module:
    method = method.lower()
    if method == "graphsage":
        return GraphSAGENet(graph.num_features, hidden_channels, graph.num_classes, dropout)
    if method == "linkx":
        from torch_geometric.nn.models import LINKX
        return LINKX(
            num_nodes=graph.num_nodes,
            in_channels=graph.num_features,
            hidden_channels=hidden_channels,
            out_channels=graph.num_classes,
            num_layers=2,
            num_edge_layers=1,
            num_node_layers=1,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported strongest method: {method}")


def train_one_config(
    graph: GraphPack,
    method: str,
    device: torch.device,
    epochs: int,
    patience: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
    hidden_channels: int,
    dropout: float,
) -> Dict[str, object]:
    set_global_seed(graph.train_seed)
    x = graph.x.to(device)
    y = graph.y.to(device)
    edge_index = graph.edge_index.to(device)
    train_idx = graph.train_idx.to(device)
    val_idx = graph.val_idx.to(device)
    test_idx = graph.test_idx.to(device)

    model = build_model(method, graph, int(hidden_channels), float(dropout)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        if epoch % int(eval_every) == 0 or epoch == int(epochs):
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                val_acc = accuracy_from_logits(logits, y, val_idx)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += int(eval_every)
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        train_acc = accuracy_from_logits(logits, y, train_idx)
        val_acc = accuracy_from_logits(logits, y, val_idx)
        test_acc = accuracy_from_logits(logits, y, test_idx)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "best_epoch": int(best_epoch),
        "time_sec": round(time.time() - t0, 3),
        "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "hidden_channels": int(hidden_channels),
        "dropout": float(dropout),
        "selection_rule": "early_stop_by_val_acc_only_no_test",
    }


def select_best_by_val_only(rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        raise RuntimeError("All configs failed; no best row can be selected.")

    def key(r: Dict[str, object]) -> Tuple[float, int, float, float, float]:
        val = float(r.get("val_acc", -1.0))
        # Tie-breaks still avoid test_acc.
        return (
            val,
            -int(r.get("hidden_channels", 10**9)),
            float(r.get("weight_decay", 0.0)),
            -float(r.get("dropout", 0.0)),
            -float(r.get("lr", 10**9)),
        )

    out = dict(max(ok, key=key))
    out["selection_rule"] = "grid_selected_by_val_acc_only_no_test"
    return out


def write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_single_csv(path: Path, row: Dict[str, object]) -> None:
    write_rows_csv(path, [row])


def base_row(graph: GraphPack, method: str, repeat: int) -> Dict[str, object]:
    return {
        "dataset": graph.name,
        "method": method,
        "repeat": int(repeat),
        "split_id": graph.split_id,
        "split_seed": graph.split_seed,
        "train_seed": graph.train_seed,
        "split_policy": graph.split_policy,
        "num_nodes": graph.num_nodes,
        "num_features": graph.num_features,
        "num_classes": graph.num_classes,
        "train_size": int(graph.train_idx.numel()),
        "val_size": int(graph.val_idx.numel()),
        "test_size": int(graph.test_idx.numel()),
        "feature_norm": "",
        "make_undirected": "",
    }


def run_dataset_repeat(args: argparse.Namespace, dataset_name: str, repeat: int) -> None:
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "squirrel":
        print("[skip] squirrel is excluded by request.")
        return
    if dataset_name not in DATASET_TO_METHOD:
        raise ValueError(f"{dataset_name} not allowed. Use one of: {FINAL_DATASETS}")

    method = DATASET_TO_METHOD[dataset_name]
    print(f"\n[load] dataset={dataset_name} method={method} repeat={repeat}")
    graph = load_dataset_for_split(
        dataset_name=dataset_name,
        data_root=args.data_root,
        repeat=repeat,
        feature_norm=args.feature_norm,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        make_undirected=not args.directed,
    )
    print(
        f"[split] nodes={graph.num_nodes} features={graph.num_features} classes={graph.num_classes} "
        f"train={graph.train_idx.numel()} val={graph.val_idx.numel()} test={graph.test_idx.numel()} "
        f"policy={graph.split_policy} split_id={graph.split_id} split_seed={graph.split_seed}"
    )

    device = get_device(args.device)
    lr_grid = parse_float_grid(args.lr_grid, [args.lr])
    wd_grid = parse_float_grid(args.weight_decay_grid, [args.weight_decay])
    hidden_grid = parse_int_grid(args.hidden_grid, [args.hidden_channels])
    dropout_grid = parse_float_grid(args.dropout_grid, [args.dropout])

    out_dir = Path(args.out_dir)
    all_rows: List[Dict[str, object]] = []
    for lr in lr_grid:
        for wd in wd_grid:
            for hidden in hidden_grid:
                for dropout in dropout_grid:
                    row = base_row(graph, method, repeat)
                    row["feature_norm"] = args.feature_norm
                    row["make_undirected"] = not args.directed
                    row.update({
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "hidden_channels": int(hidden),
                        "dropout": float(dropout),
                    })
                    print(f"[train] {dataset_name} r={repeat} {method} lr={lr} wd={wd} hidden={hidden} dropout={dropout}")
                    try:
                        stats = train_one_config(
                            graph=graph,
                            method=method,
                            device=device,
                            epochs=args.epochs,
                            patience=args.patience,
                            eval_every=args.eval_every,
                            lr=lr,
                            weight_decay=wd,
                            hidden_channels=hidden,
                            dropout=dropout,
                        )
                        row.update(stats)
                        print(f"[done] val={row['val_acc']:.6f} test={row['test_acc']:.6f} epoch={row['best_epoch']} time={row['time_sec']}s")
                    except Exception as e:
                        row["error"] = repr(e)
                        print(f"[error] {repr(e)}")
                    all_rows.append(row)

    all_path = out_dir / "all_grid_runs" / f"{dataset_name}__repeat{repeat}__all_grid.csv"
    write_rows_csv(all_path, all_rows)
    best = select_best_by_val_only(all_rows)
    best["selection_rule"] = "strongest_method_for_dataset_then_grid_selected_by_val_only_no_test"
    single_path = out_dir / "single_runs" / f"{dataset_name}__repeat{repeat}__best.csv"
    write_single_csv(single_path, best)
    print(f"[all] wrote {all_path}")
    print(f"[best] wrote {single_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Linux path to the uploaded data root on AutoDL, e.g. /root/autodl-tmp/planetoid/data")
    ap.add_argument("--dataset", type=str, default="all", help="one dataset or all; squirrel is never run")
    ap.add_argument("--repeats", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_dir", type=str, default="results_strong_baseline_cpu")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--hidden_channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr_grid", type=str, default="", help="optional small val-only grid, e.g. 0.01,0.005")
    ap.add_argument("--weight_decay_grid", type=str, default="", help="optional small val-only grid, e.g. 0,0.0005")
    ap.add_argument("--hidden_grid", type=str, default="", help="optional small val-only grid, e.g. 64,128")
    ap.add_argument("--dropout_grid", type=str, default="", help="optional small val-only grid, e.g. 0.3,0.5")
    ap.add_argument("--feature_norm", type=str, default="row_l1", choices=["none", "row_l1", "standard"])
    ap.add_argument("--train_per_class", type=int, default=20)
    ap.add_argument("--val_per_class", type=int, default=30)
    ap.add_argument("--directed", action="store_true", help="do not make graph undirected; default matches prior fair script: undirected")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(max(1, int(args.threads)))

    datasets: Iterable[str]
    if normalize_dataset_name(args.dataset) == "all":
        datasets = FINAL_DATASETS
    else:
        datasets = [normalize_dataset_name(args.dataset)]
    repeats = parse_int_list(args.repeats)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("[plan] strongest methods:")
    for d in datasets:
        if d == "squirrel":
            continue
        print(f"  {d}: {DATASET_TO_METHOD[d]}")

    for d in datasets:
        for r in repeats:
            run_dataset_repeat(args, d, int(r))


if __name__ == "__main__":
    main()
