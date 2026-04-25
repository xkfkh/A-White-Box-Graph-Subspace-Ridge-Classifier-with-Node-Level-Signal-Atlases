import numpy as np
from scipy import sparse

from ha_sdc import HASDC


def make_synthetic_graph(seed=0):
    rng = np.random.default_rng(seed)
    n_per_class = 40
    n = 2 * n_per_class
    D = 6

    y = np.array([0] * n_per_class + [1] * n_per_class)
    X = np.zeros((n, D), dtype=float)
    X[y == 0] = rng.normal(loc=[2, 0, 0, 0, 0, 0], scale=0.8, size=(n_per_class, D))
    X[y == 1] = rng.normal(loc=[0, 2, 0, 0, 0, 0], scale=0.8, size=(n_per_class, D))

    rows = []
    cols = []
    # Class 0: mostly homophilous edges.
    for i in range(n_per_class):
        candidates = np.where(y == 0)[0]
        neigh = rng.choice(candidates[candidates != i], size=5, replace=False)
        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())
    # Class 1: more heterophilous edges.
    for i in range(n_per_class, n):
        candidates = np.where(y == 0)[0]
        neigh = rng.choice(candidates, size=5, replace=False)
        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())

    data = np.ones(len(rows), dtype=float)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float).tocsr()

    train_idx = np.r_[np.arange(0, 10), np.arange(n_per_class, n_per_class + 10)]
    val_idx = np.r_[np.arange(10, 20), np.arange(n_per_class + 10, n_per_class + 20)]
    test_idx = np.setdiff1d(np.arange(n), np.r_[train_idx, val_idx])
    return A, X, y, train_idx, val_idx, test_idx


def main():
    A, X, y, train_idx, val_idx, test_idx = make_synthetic_graph()

    # Standardize features using train nodes only.
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0) + 1e-12
    X_std = (X - mean) / std

    model = HASDC(lambda_smooth=1.0, d_s=3, d_r=3, tau_gate=5.0, gamma=0.1)
    model.fit(A, X_std, y, train_idx, val_idx)
    pred = model.predict(A, X_std)

    print("Validation accuracy:", np.mean(pred[val_idx] == y[val_idx]))
    print("Test accuracy:", np.mean(pred[test_idx] == y[test_idx]))
    print("Gate:", model.get_state().gate)
    print("Alpha:\n", model.get_state().alpha)
    print("Explanation for node 0:")
    print(model.explain_node(A, X_std, node_id=0))


if __name__ == "__main__":
    main()
