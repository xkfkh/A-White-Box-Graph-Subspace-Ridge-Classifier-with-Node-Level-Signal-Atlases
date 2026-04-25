import itertools
import numpy as np

from ha_sdc import HASDC
from synthetic_demo import make_synthetic_graph


def main():
    A, X, y, train_idx, val_idx, test_idx = make_synthetic_graph(seed=1)
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0) + 1e-12
    X = (X - mean) / std

    best = None
    for lambda_smooth, d_s, d_r, gamma in itertools.product(
        [0.1, 1.0, 5.0],
        [2, 3, 5],
        [2, 3, 5],
        [0.01, 0.05, 0.1, 0.5],
    ):
        model = HASDC(
            lambda_smooth=lambda_smooth,
            d_s=d_s,
            d_r=d_r,
            gamma=gamma,
            tau_gate=5.0,
            alpha_min_acc_delta=0.0,
        )
        model.fit(A, X, y, train_idx, val_idx)
        pred = model.predict(A, X)
        val_acc = np.mean(pred[val_idx] == y[val_idx])
        candidate = (val_acc, lambda_smooth, d_s, d_r, gamma, model)
        if best is None or candidate[0] > best[0]:
            best = candidate

    val_acc, lambda_smooth, d_s, d_r, gamma, model = best
    pred = model.predict(A, X)
    print("Best validation accuracy:", val_acc)
    print("Best params:", dict(lambda_smooth=lambda_smooth, d_s=d_s, d_r=d_r, gamma=gamma))
    print("Test accuracy:", np.mean(pred[test_idx] == y[test_idx]))
    print("Gate:", model.get_state().gate)
    print("Alpha:\n", model.get_state().alpha)


if __name__ == "__main__":
    main()
