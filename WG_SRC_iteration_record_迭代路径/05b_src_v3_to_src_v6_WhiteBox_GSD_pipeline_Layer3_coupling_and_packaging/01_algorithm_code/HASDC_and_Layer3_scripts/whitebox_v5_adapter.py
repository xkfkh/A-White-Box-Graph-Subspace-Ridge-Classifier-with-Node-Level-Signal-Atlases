"""
Shared adapter for running the experiment scripts against the user's current src_v5.

Design rules for reproducible experiments:
- Prefer explicit WHITEBOX_SRC_V5 / WHITEBOX_DATA_DIR paths.
- Never silently modify algorithm source files at runtime.
- Purge previously imported shadow modules so experiments really use current src_v5.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _candidate_roots(start: Path):
    seen = set()
    for p in [start, *start.parents]:
        if p not in seen:
            seen.add(p)
            yield p
        if p.parent not in seen:
            seen.add(p.parent)
            yield p.parent


def find_src_v5(start_file: Optional[str] = None) -> Path:
    env_src = os.environ.get("WHITEBOX_SRC_V5") or os.environ.get("WHITEBOX_SRC_DIR")
    if env_src:
        p = Path(env_src).expanduser().resolve()
        if p.is_dir() and (p / "data_loader.py").exists():
            return p
        raise FileNotFoundError(f"WHITEBOX_SRC_V5 must point to src_v5 containing data_loader.py, got: {p}")

    start = Path(start_file or __file__).resolve().parent
    names = ("src_v5", "src", "srccv5", "srcv5", "src-v5")
    for root in _candidate_roots(start):
        for name in names:
            cand = (root / name).resolve()
            if cand.is_dir() and (cand / "data_loader.py").exists():
                return cand

    searched = " -> ".join(str(p) for p in list(_candidate_roots(start))[:8])
    raise FileNotFoundError(
        "Cannot locate src_v5/src. Put it in the project tree or set WHITEBOX_SRC_V5. "
        "Searched near: " + searched
    )


def _check_no_known_syntax_issue(src_dir: Path) -> None:
    """Fail loudly instead of auto-patching source code."""
    layer3 = src_dir / "layer3_discriminative.py"
    if not layer3.exists():
        return
    text = layer3.read_text(encoding="utf-8")
    bad = "\n        elif aggregation_mode == 'linear':\n        elif aggregation_mode == 'linear':"
    if bad in text:
        raise SyntaxError(
            "layer3_discriminative.py contains a duplicated `elif aggregation_mode == 'linear':`. "
            "Please fix the source file manually; scripts will not silently edit algorithm code."
        )


def _purge_shadowed_modules(src_dir: Path) -> None:
    module_names = [
        "data_loader",
        "layer1_tikhonov",
        "layer2_subspace",
        "layer3_discriminative",
        "framework",
        "framework.configs",
        "framework.pipeline",
    ]
    src_dir_str = str(src_dir.resolve())
    for name in module_names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        if mod_file and src_dir_str not in str(Path(mod_file).resolve()):
            sys.modules.pop(name, None)


def _require_planetoid_data(data_dir: Path) -> None:
    required = []
    for ds in ("cora", "citeseer", "pubmed"):
        for suffix in ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"):
            required.append(data_dir / f"ind.{ds}.{suffix}")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        preview = "\n  ".join(missing[:12])
        raise FileNotFoundError(
            f"WHITEBOX_DATA_DIR is incomplete: {data_dir}\nMissing examples:\n  {preview}"
        )


def setup_src_v5(start_file: Optional[str] = None) -> str:
    src_dir = find_src_v5(start_file)
    _check_no_known_syntax_issue(src_dir)
    _purge_shadowed_modules(src_dir)

    src_str = str(src_dir)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != src_dir]
    sys.path.insert(0, src_str)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    data_env = os.environ.get("WHITEBOX_DATA_DIR")
    if data_env:
        data_dir = Path(data_env).expanduser().resolve()
    else:
        data_dir = (src_dir / "data").resolve()
        os.environ["WHITEBOX_DATA_DIR"] = str(data_dir)

    _require_planetoid_data(data_dir)
    print(f"[adapter] src_v5 = {src_dir}")
    print(f"[adapter] data   = {data_dir}")
    return src_str


def l3_kwargs_from_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config or {}
    return {
        "eta_react": config.get("eta_react", config.get("eta_pos", 0.10)),
        "max_activation_scale": config.get("max_activation_scale", None),
        "activation_mode": config.get("activation_mode", "mahalanobis"),
        "activation_rho": config.get("activation_rho", 1e-6),
        "aggregation_mode": config.get("aggregation_mode", "suppression"),
    }


def compute_layer3_v5(Z, sub, num_classes: int, config: Optional[Dict[str, Any]] = None):
    from layer3_discriminative import compute_discriminative_rperp
    return compute_discriminative_rperp(Z, sub, num_classes, **l3_kwargs_from_config(config))


def base_weight_vectors_for_plot(disc_weights: Dict[str, Any]) -> Dict[int, Any]:
    if "base_weights" in disc_weights:
        return disc_weights["base_weights"]
    if "base_matrices" not in disc_weights:
        raise KeyError("disc_weights has neither 'base_weights' nor 'base_matrices'.")
    return {c: W.diag().detach().cpu() for c, W in disc_weights["base_matrices"].items()}


def optional_src_v1_path(src_v5_path: str) -> Optional[str]:
    root = Path(src_v5_path).resolve().parent
    cand = root / "src_v1"
    return str(cand) if cand.is_dir() else None
