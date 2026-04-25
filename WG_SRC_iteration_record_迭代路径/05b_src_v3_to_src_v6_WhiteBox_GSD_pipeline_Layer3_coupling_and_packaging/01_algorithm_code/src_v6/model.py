from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


ArrayLike = Any
Label = Hashable


@dataclass
class Subspace:
    """One class-specific linear subspace.

    Attributes
    ----------
    mu:
        Class center, shape ``(D,)``.
    basis:
        Orthonormal basis matrix, shape ``(D, d_eff)``. Columns are basis vectors.
    n_samples:
        Number of training samples used to build this subspace.
    """

    mu: np.ndarray
    basis: np.ndarray
    n_samples: int


@dataclass
class HASDCModelState:
    """Learned white-box quantities of HA-SDC."""

    classes: np.ndarray
    smooth_subspaces: Dict[Label, Subspace]
    raw_subspaces: Dict[Label, Subspace]
    gate: Dict[Label, float]
    smooth_accuracy_by_class: Dict[Label, float]
    raw_accuracy_by_class: Dict[Label, float]
    alpha: np.ndarray
    overlap: np.ndarray


class HASDC:
    """Homophily-Heterophily Adaptive Signed Dual-Subspace Crosslinking.

    Parameters
    ----------
    lambda_smooth:
        Tikhonov graph smoothing strength in ``Z = (I + lambda_smooth * L)^(-1) X``.
    d_s:
        Maximum dimension of every class smooth subspace.
    d_r:
        Maximum dimension of every class raw subspace.
    tau_gate:
        Sharpness of the sigmoid gate ``g_c = sigmoid(tau_gate * (Acc_s - Acc_r))``.
    gamma:
        Strength of the signed crosslink modulation.
    laplacian:
        ``"normalized"`` for symmetric normalized Laplacian or ``"combinatorial"`` for ``D-A``.
    add_self_loops:
        Whether to add self-loops to the adjacency matrix before building the Laplacian.
    alpha_min_overlap:
        Pairwise subspace overlap below this threshold is treated as no crosslink.
    alpha_min_acc_delta:
        Minimal validation class-accuracy difference needed to assign a non-zero sign.
    residual_floor:
        Small numerical floor applied to residuals.
    """

    def __init__(
        self,
        lambda_smooth: float = 1.0,
        d_s: int = 8,
        d_r: int = 8,
        tau_gate: float = 5.0,
        gamma: float = 0.1,
        laplacian: str = "normalized",
        add_self_loops: bool = False,
        alpha_min_overlap: float = 1e-8,
        alpha_min_acc_delta: float = 0.0,
        residual_floor: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative.")
        if d_s < 0 or d_r < 0:
            raise ValueError("d_s and d_r must be non-negative.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if laplacian not in {"normalized", "combinatorial"}:
            raise ValueError("laplacian must be 'normalized' or 'combinatorial'.")

        self.lambda_smooth = float(lambda_smooth)
        self.d_s = int(d_s)
        self.d_r = int(d_r)
        self.tau_gate = float(tau_gate)
        self.gamma = float(gamma)
        self.laplacian = laplacian
        self.add_self_loops = bool(add_self_loops)
        self.alpha_min_overlap = float(alpha_min_overlap)
        self.alpha_min_acc_delta = float(alpha_min_acc_delta)
        self.residual_floor = float(residual_floor)
        self.eps = float(eps)

        self.state_: Optional[HASDCModelState] = None
        self.Z_fit_: Optional[np.ndarray] = None
        self.X_fit_: Optional[np.ndarray] = None
        self.train_idx_: Optional[np.ndarray] = None
        self.val_idx_: Optional[np.ndarray] = None
        self.y_train_encoded_: Optional[np.ndarray] = None
        self._label_to_pos: Optional[Dict[Label, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_from_labels(
        self,
        A: ArrayLike,
        X: ArrayLike,
        train_labels: Any,
        val_labels: Any,
    ) -> "HASDC":
        """Fit from separated training and validation labels.

        ``train_labels`` and ``val_labels`` may be one of the following forms:

        1. ``{node_index: label, ...}``
        2. ``(indices, labels)``
        3. ``[(node_index, label), ...]``

        This method builds the full ``y`` array internally and then calls ``fit``.
        """
        X_np = self._as_2d_float_array(X, name="X")
        n = X_np.shape[0]
        train_idx, train_y = self._parse_split_labels(train_labels, n, "train_labels")
        val_idx, val_y = self._parse_split_labels(val_labels, n, "val_labels")

        y = np.empty(n, dtype=object)
        y[:] = None
        y[train_idx] = train_y
        y[val_idx] = val_y
        return self.fit(A, X_np, y, train_idx, val_idx)

    def fit(
        self,
        A: ArrayLike,
        X: ArrayLike,
        y: ArrayLike,
        train_idx: Iterable[int],
        val_idx: Iterable[int],
    ) -> "HASDC":
        """Fit HA-SDC.

        ``y`` should have length ``n``. Labels are required for ``train_idx`` and ``val_idx``.
        Unlabeled nodes outside these two sets may contain any placeholder.
        """
        X_np = self._as_2d_float_array(X, name="X")
        n, _ = X_np.shape
        A_sp = self._as_square_sparse_adjacency(A, n=n)
        y_np = np.asarray(y)
        if y_np.shape[0] != n:
            raise ValueError("y must have length n, the number of graph nodes.")

        train_idx_np = np.asarray(list(train_idx), dtype=int)
        val_idx_np = np.asarray(list(val_idx), dtype=int)
        self._validate_indices(train_idx_np, n, "train_idx")
        self._validate_indices(val_idx_np, n, "val_idx")
        if train_idx_np.size == 0:
            raise ValueError("train_idx must contain at least one node.")
        if val_idx_np.size == 0:
            raise ValueError("val_idx must contain at least one node.")

        classes = np.unique(y_np[train_idx_np])
        if classes.size < 2:
            raise ValueError("At least two classes are required in the training set.")
        label_to_pos = {label: pos for pos, label in enumerate(classes)}
        self._label_to_pos = label_to_pos

        Z_np = self._smooth_features(A_sp, X_np)
        smooth_subspaces = self._fit_subspaces(Z_np, y_np, train_idx_np, classes, self.d_s)
        raw_subspaces = self._fit_subspaces(X_np, y_np, train_idx_np, classes, self.d_r)

        residual_s, _ = self._residuals_and_coords(Z_np, smooth_subspaces, classes)
        residual_r, coords_r = self._residuals_and_coords(X_np, raw_subspaces, classes)

        pred_s = classes[np.argmin(residual_s, axis=1)]
        pred_r = classes[np.argmin(residual_r, axis=1)]
        acc_s, acc_r = self._classwise_channel_accuracy(pred_s, pred_r, y_np, val_idx_np, classes)
        gate = self._compute_gate(acc_s, acc_r, classes)

        base = self._base_residual(residual_s, residual_r, classes, gate)
        overlap = self._compute_overlaps(smooth_subspaces, raw_subspaces, classes)
        alpha = self._estimate_signed_alpha(
            base_residual=base,
            coords_raw=coords_r,
            smooth_subspaces=smooth_subspaces,
            raw_subspaces=raw_subspaces,
            y=y_np,
            val_idx=val_idx_np,
            classes=classes,
            overlap=overlap,
        )

        self.state_ = HASDCModelState(
            classes=classes,
            smooth_subspaces=smooth_subspaces,
            raw_subspaces=raw_subspaces,
            gate=gate,
            smooth_accuracy_by_class=acc_s,
            raw_accuracy_by_class=acc_r,
            alpha=alpha,
            overlap=overlap,
        )
        self.Z_fit_ = Z_np
        self.X_fit_ = X_np
        self.train_idx_ = train_idx_np
        self.val_idx_ = val_idx_np
        self.y_train_encoded_ = y_np
        return self

    def decision_function(self, A: ArrayLike, X: ArrayLike) -> np.ndarray:
        """Return final HA-SDC residual matrix, shape ``(n, C)``.

        Smaller residual means stronger class evidence.
        """
        self._require_fitted()
        X_np = self._as_2d_float_array(X, name="X")
        A_sp = self._as_square_sparse_adjacency(A, n=X_np.shape[0])
        Z_np = self._smooth_features(A_sp, X_np)
        return self._final_residual_from_features(Z_np, X_np)

    def predict(self, A: ArrayLike, X: ArrayLike) -> np.ndarray:
        """Predict class labels for all nodes."""
        self._require_fitted()
        residual = self.decision_function(A, X)
        classes = self.state_.classes  # type: ignore[union-attr]
        return classes[np.argmin(residual, axis=1)]

    def fit_predict(
        self,
        A: ArrayLike,
        X: ArrayLike,
        y: ArrayLike,
        train_idx: Iterable[int],
        val_idx: Iterable[int],
    ) -> np.ndarray:
        """Fit and predict all nodes."""
        self.fit(A, X, y, train_idx, val_idx)
        return self.predict(A, X)

    def explain_node(self, A: ArrayLike, X: ArrayLike, node_id: int) -> Dict[Label, Dict[str, float]]:
        """Return a white-box decomposition for one node and every class.

        The returned dictionary contains smooth residual, raw residual, gate, base residual,
        crosslink modulation, and final residual.
        """
        self._require_fitted()
        X_np = self._as_2d_float_array(X, name="X")
        if not 0 <= int(node_id) < X_np.shape[0]:
            raise IndexError("node_id is out of range.")
        A_sp = self._as_square_sparse_adjacency(A, n=X_np.shape[0])
        Z_np = self._smooth_features(A_sp, X_np)

        state = self.state_
        assert state is not None
        residual_s, _ = self._residuals_and_coords(Z_np, state.smooth_subspaces, state.classes)
        residual_r, coords_r = self._residuals_and_coords(X_np, state.raw_subspaces, state.classes)
        base = self._base_residual(residual_s, residual_r, state.classes, state.gate)
        cross = self._crosslink_modulation(coords_r, state.smooth_subspaces, state.raw_subspaces, state.classes, state.alpha)
        final = base - self.gamma * cross
        if self.residual_floor > 0:
            final = np.maximum(final, self.residual_floor)

        k = int(node_id)
        result: Dict[Label, Dict[str, float]] = {}
        for c_pos, c in enumerate(state.classes):
            result[c] = {
                "smooth_residual": float(residual_s[k, c_pos]),
                "raw_residual": float(residual_r[k, c_pos]),
                "gate": float(state.gate[c]),
                "base_residual": float(base[k, c_pos]),
                "crosslink_modulation": float(cross[k, c_pos]),
                "final_residual": float(final[k, c_pos]),
            }
        return result

    def get_state(self) -> HASDCModelState:
        """Return learned white-box state."""
        self._require_fitted()
        assert self.state_ is not None
        return self.state_

    # ------------------------------------------------------------------
    # Core implementation
    # ------------------------------------------------------------------
    def _smooth_features(self, A: sparse.spmatrix, X: np.ndarray) -> np.ndarray:
        if self.lambda_smooth == 0:
            return X.copy()
        L = self._laplacian(A)
        n = X.shape[0]
        system = sparse.eye(n, format="csr") + self.lambda_smooth * L
        Z = splinalg.spsolve(system.tocsc(), X)
        if Z.ndim == 1:
            Z = Z[:, None]
        return np.asarray(Z, dtype=float)

    def _laplacian(self, A: sparse.spmatrix) -> sparse.csr_matrix:
        A = A.tocsr().astype(float)
        if self.add_self_loops:
            A = A + sparse.eye(A.shape[0], format="csr")
        # Symmetrize for an undirected graph interpretation. If the user passes
        # a directed adjacency, this classifier uses its undirected support.
        A = 0.5 * (A + A.T)
        degrees = np.asarray(A.sum(axis=1)).ravel()
        n = A.shape[0]
        if self.laplacian == "combinatorial":
            return (sparse.diags(degrees) - A).tocsr()

        inv_sqrt = np.zeros_like(degrees, dtype=float)
        mask = degrees > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
        D_inv_sqrt = sparse.diags(inv_sqrt)
        return (sparse.eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

    def _fit_subspaces(
        self,
        features: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        classes: np.ndarray,
        max_dim: int,
    ) -> Dict[Label, Subspace]:
        result: Dict[Label, Subspace] = {}
        D = features.shape[1]
        for c in classes:
            idx = train_idx[y[train_idx] == c]
            if idx.size == 0:
                raise ValueError(f"No training samples for class {c!r}.")
            Xc = features[idx]
            mu = Xc.mean(axis=0)
            centered = Xc - mu
            d_eff = min(max_dim, max(0, idx.size - 1), D)
            if d_eff == 0:
                basis = np.empty((D, 0), dtype=float)
            else:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                basis = vt[:d_eff].T.copy()
            result[c] = Subspace(mu=mu, basis=basis, n_samples=int(idx.size))
        return result

    def _residuals_and_coords(
        self,
        features: np.ndarray,
        subspaces: Dict[Label, Subspace],
        classes: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[Label, np.ndarray]]:
        n = features.shape[0]
        C = classes.size
        residuals = np.zeros((n, C), dtype=float)
        coords: Dict[Label, np.ndarray] = {}
        for pos, c in enumerate(classes):
            sub = subspaces[c]
            centered = features - sub.mu
            if sub.basis.shape[1] == 0:
                coord = np.empty((n, 0), dtype=float)
                evidence = np.zeros(n, dtype=float)
            else:
                coord = centered @ sub.basis
                evidence = np.sum(coord * coord, axis=1)
            total = np.sum(centered * centered, axis=1)
            residual = total - evidence
            residuals[:, pos] = np.maximum(residual, self.residual_floor)
            coords[c] = coord
        return residuals, coords

    def _classwise_channel_accuracy(
        self,
        pred_s: np.ndarray,
        pred_r: np.ndarray,
        y: np.ndarray,
        val_idx: np.ndarray,
        classes: np.ndarray,
    ) -> Tuple[Dict[Label, float], Dict[Label, float]]:
        acc_s: Dict[Label, float] = {}
        acc_r: Dict[Label, float] = {}
        global_s = float(np.mean(pred_s[val_idx] == y[val_idx]))
        global_r = float(np.mean(pred_r[val_idx] == y[val_idx]))
        for c in classes:
            mask = val_idx[y[val_idx] == c]
            if mask.size == 0:
                # No class-specific validation signal: use equal accuracies so gate is neutral.
                acc_s[c] = 0.5 * (global_s + global_r)
                acc_r[c] = 0.5 * (global_s + global_r)
            else:
                acc_s[c] = float(np.mean(pred_s[mask] == y[mask]))
                acc_r[c] = float(np.mean(pred_r[mask] == y[mask]))
        return acc_s, acc_r

    def _compute_gate(
        self,
        acc_s: Dict[Label, float],
        acc_r: Dict[Label, float],
        classes: np.ndarray,
    ) -> Dict[Label, float]:
        gate: Dict[Label, float] = {}
        for c in classes:
            h = acc_s[c] - acc_r[c]
            value = 1.0 / (1.0 + np.exp(-self.tau_gate * h))
            gate[c] = float(value)
        return gate

    def _base_residual(
        self,
        residual_s: np.ndarray,
        residual_r: np.ndarray,
        classes: np.ndarray,
        gate: Dict[Label, float],
    ) -> np.ndarray:
        g = np.asarray([gate[c] for c in classes], dtype=float)[None, :]
        return g * residual_s + (1.0 - g) * residual_r

    def _compute_overlaps(
        self,
        smooth_subspaces: Dict[Label, Subspace],
        raw_subspaces: Dict[Label, Subspace],
        classes: np.ndarray,
    ) -> np.ndarray:
        C = classes.size
        overlap = np.zeros((C, C), dtype=float)
        for i, c in enumerate(classes):
            Bs = smooth_subspaces[c].basis
            for j, cp in enumerate(classes):
                if i == j:
                    continue
                Br = raw_subspaces[cp].basis
                if Bs.shape[1] == 0 or Br.shape[1] == 0:
                    overlap[i, j] = 0.0
                    continue
                M = Bs.T @ Br
                denom = max(1, min(Bs.shape[1], Br.shape[1]))
                overlap[i, j] = float(np.sum(M * M) / denom)
        return overlap

    def _estimate_signed_alpha(
        self,
        base_residual: np.ndarray,
        coords_raw: Dict[Label, np.ndarray],
        smooth_subspaces: Dict[Label, Subspace],
        raw_subspaces: Dict[Label, Subspace],
        y: np.ndarray,
        val_idx: np.ndarray,
        classes: np.ndarray,
        overlap: np.ndarray,
    ) -> np.ndarray:
        C = classes.size
        alpha = np.zeros((C, C), dtype=float)
        base_val = base_residual[val_idx]

        for c_pos, c in enumerate(classes):
            true_c_mask = y[val_idx] == c
            if not np.any(true_c_mask):
                continue
            for cp_pos, cp in enumerate(classes):
                if c_pos == cp_pos:
                    continue
                strength = overlap[c_pos, cp_pos]
                if strength < self.alpha_min_overlap:
                    continue
                q = self._pair_cross_feature(coords_raw[cp][val_idx], smooth_subspaces[c], raw_subspaces[cp])
                if np.all(np.abs(q) <= self.eps):
                    continue

                pos_res = base_val.copy()
                neg_res = base_val.copy()
                pos_res[:, c_pos] -= self.gamma * strength * q
                neg_res[:, c_pos] += self.gamma * strength * q

                pos_pred = np.argmin(pos_res, axis=1)
                neg_pred = np.argmin(neg_res, axis=1)
                pos_acc = float(np.mean(pos_pred[true_c_mask] == c_pos))
                neg_acc = float(np.mean(neg_pred[true_c_mask] == c_pos))

                delta = pos_acc - neg_acc
                if abs(delta) <= self.alpha_min_acc_delta:
                    alpha[c_pos, cp_pos] = 0.0
                elif delta > 0:
                    alpha[c_pos, cp_pos] = strength
                else:
                    alpha[c_pos, cp_pos] = -strength
        return alpha

    def _crosslink_modulation(
        self,
        coords_raw: Dict[Label, np.ndarray],
        smooth_subspaces: Dict[Label, Subspace],
        raw_subspaces: Dict[Label, Subspace],
        classes: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        n = next(iter(coords_raw.values())).shape[0]
        C = classes.size
        modulation = np.zeros((n, C), dtype=float)
        for c_pos, c in enumerate(classes):
            for cp_pos, cp in enumerate(classes):
                if c_pos == cp_pos:
                    continue
                a = alpha[c_pos, cp_pos]
                if abs(a) <= self.eps:
                    continue
                q = self._pair_cross_feature(coords_raw[cp], smooth_subspaces[c], raw_subspaces[cp])
                modulation[:, c_pos] += a * q
        return modulation

    def _pair_cross_feature(self, raw_coord: np.ndarray, smooth_sub: Subspace, raw_sub: Subspace) -> np.ndarray:
        Bs = smooth_sub.basis
        Br = raw_sub.basis
        if Bs.shape[1] == 0 or Br.shape[1] == 0 or raw_coord.shape[1] == 0:
            return np.zeros(raw_coord.shape[0], dtype=float)
        M = Bs.T @ Br
        transferred = raw_coord @ M.T
        denom = max(1, Bs.shape[1])
        return np.sum(transferred * transferred, axis=1) / denom

    def _final_residual_from_features(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        state = self.state_
        assert state is not None
        residual_s, _ = self._residuals_and_coords(Z, state.smooth_subspaces, state.classes)
        residual_r, coords_r = self._residuals_and_coords(X, state.raw_subspaces, state.classes)
        base = self._base_residual(residual_s, residual_r, state.classes, state.gate)
        cross = self._crosslink_modulation(
            coords_r,
            state.smooth_subspaces,
            state.raw_subspaces,
            state.classes,
            state.alpha,
        )
        final = base - self.gamma * cross
        if self.residual_floor > 0:
            final = np.maximum(final, self.residual_floor)
        return final

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if self.state_ is None:
            raise RuntimeError("HASDC is not fitted. Call fit(...) first.")


    @staticmethod
    def _parse_split_labels(split_labels: Any, n: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(split_labels, dict):
            idx = np.asarray(list(split_labels.keys()), dtype=int)
            labels = np.asarray(list(split_labels.values()), dtype=object)
        elif isinstance(split_labels, tuple) and len(split_labels) == 2:
            idx = np.asarray(split_labels[0], dtype=int)
            labels = np.asarray(split_labels[1], dtype=object)
        else:
            arr = list(split_labels)
            if len(arr) == 0:
                idx = np.asarray([], dtype=int)
                labels = np.asarray([], dtype=object)
            else:
                idx = np.asarray([pair[0] for pair in arr], dtype=int)
                labels = np.asarray([pair[1] for pair in arr], dtype=object)
        if idx.ndim != 1 or labels.ndim != 1 or idx.size != labels.size:
            raise ValueError(f"{name} must contain matching one-dimensional indices and labels.")
        if idx.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if idx.min() < 0 or idx.max() >= n:
            raise IndexError(f"{name} contains node indices outside [0, n).")
        return idx, labels

    @staticmethod
    def _as_2d_float_array(X: ArrayLike, name: str) -> np.ndarray:
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim != 2:
            raise ValueError(f"{name} must be a 2D array of shape (n, D).")
        if not np.all(np.isfinite(X_np)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return X_np

    @staticmethod
    def _as_square_sparse_adjacency(A: ArrayLike, n: int) -> sparse.csr_matrix:
        if sparse.issparse(A):
            A_sp = A.tocsr().astype(float)
        else:
            A_sp = sparse.csr_matrix(np.asarray(A, dtype=float))
        if A_sp.shape != (n, n):
            raise ValueError(f"A must have shape ({n}, {n}).")
        if A_sp.nnz > 0 and not np.all(np.isfinite(A_sp.data)):
            raise ValueError("A contains NaN or infinite values.")
        return A_sp

    @staticmethod
    def _validate_indices(idx: np.ndarray, n: int, name: str) -> None:
        if idx.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional iterable of indices.")
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise IndexError(f"{name} contains indices outside [0, n).")
