"""
Empirical parameter settings for each dataset
(v4: normalized activation + competition-aware suppression Layer 3)
"""

DEFAULT_CONFIGS = {
    'cora': {
        'lam': 10.0,
        'sub_dim': 12,
        'eta_react': 0.10,
        'activation_mode': 'mahalanobis',
        'activation_rho': 1e-6,
        'aggregation_mode': 'suppression',
        'max_activation_scale': None,
    },
    'citeseer': {
        'lam': 7.0,
        'sub_dim': 12,
        'eta_react': 0.10,
        'activation_mode': 'mahalanobis',
        'activation_rho': 1e-6,
        'aggregation_mode': 'suppression',
        'max_activation_scale': None,
    },
    'pubmed': {
        'lam': 10.0,
        'sub_dim': 12,
        'eta_react': 0.10,
        'activation_mode': 'mahalanobis',
        'activation_rho': 1e-6,
        'aggregation_mode': 'suppression',
        'max_activation_scale': None,
    },
}


def get_config(dataset):
    ds = dataset.lower()
    if ds not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown dataset: {ds}. Choose from: cora, citeseer, pubmed")
    return DEFAULT_CONFIGS[ds].copy()
