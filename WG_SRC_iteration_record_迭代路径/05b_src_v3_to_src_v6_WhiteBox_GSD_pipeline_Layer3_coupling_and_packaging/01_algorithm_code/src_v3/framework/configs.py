"""
Empirical parameter settings for each dataset
(v3: Layer 3 reactive version)

Notes:
- The parameters in this file are empirical settings adopted in the current implementation
  and are used to reproduce the experimental pipeline.
- They are not claimed to be strictly optimal hyperparameters, nor do they imply
  optimality on an independent test set.
- The Layer 1 / Layer 2 parameters are mainly chosen based on development-stage
  empirical observations and stable performance.
- The Layer 3 parameters use the default empirical settings of the v3 reactive version.

Protocol note:
- The white-box model construction uses training labels only.
- Train / validation / test metrics are computed during evaluation.
"""

DEFAULT_CONFIGS = {
    # ── Cora ──────────────────────────────────────────────────
    # Layer 1/2 parameters are empirical settings adopted during development;
    # Layer 3 uses the default empirical values of the v3 reactive version
    'cora': {
        'lam':        10.0,   # Layer 1: Tikhonov smoothing strength (empirical setting)
        'sub_dim':    12,     # Layer 2: PCA subspace dimension (empirical setting)
        'eta_pos':    0.10,   # Layer 3: positive reaction enhancement coefficient (default empirical value)
        'eta_neg':    0.30,   # Layer 3: negative reaction suppression coefficient (default empirical value)
        'weight_min': 0.01,   # Layer 3: lower bound of dynamic weights
        'weight_max': 10.0,   # Layer 3: upper bound of dynamic weights
    },

    # ── CiteSeer ──────────────────────────────────────────────
    'citeseer': {
        'lam':        7.0,    # Layer 1: Tikhonov smoothing strength (empirical setting)
        'sub_dim':    12,     # Layer 2: PCA subspace dimension (empirical setting)
        'eta_pos':    0.10,   # Layer 3: positive reaction enhancement coefficient (default empirical value)
        'eta_neg':    0.30,   # Layer 3: negative reaction suppression coefficient (default empirical value)
        'weight_min': 0.01,   # Layer 3: lower bound of dynamic weights
        'weight_max': 10.0,   # Layer 3: upper bound of dynamic weights
    },

    # ── PubMed ────────────────────────────────────────────────
    'pubmed': {
        'lam':        10.0,   # Layer 1: Tikhonov smoothing strength (empirical setting)
        'sub_dim':    12,     # Layer 2: PCA subspace dimension (empirical setting)
        'eta_pos':    0.10,   # Layer 3: positive reaction enhancement coefficient (default empirical value)
        'eta_neg':    0.30,   # Layer 3: negative reaction suppression coefficient (default empirical value)
        'weight_min': 0.01,   # Layer 3: lower bound of dynamic weights
        'weight_max': 10.0,   # Layer 3: upper bound of dynamic weights
    },
}


def get_config(dataset):
    """Return the empirical configuration adopted in the current implementation."""
    ds = dataset.lower()
    if ds not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown dataset: {ds}. Choose from: cora, citeseer, pubmed")
    return DEFAULT_CONFIGS[ds].copy()