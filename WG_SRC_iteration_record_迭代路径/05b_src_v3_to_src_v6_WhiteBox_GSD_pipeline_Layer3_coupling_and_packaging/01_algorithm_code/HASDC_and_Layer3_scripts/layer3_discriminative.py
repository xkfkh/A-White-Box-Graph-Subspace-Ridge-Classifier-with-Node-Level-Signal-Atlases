"""
Layer 3: White-Box Cross-Coupled Geometric Discriminant Layer
==============================================================

This version fixes two theoretical issues in the previous implementation:

1) Normalized activation (Issue 4)
----------------------------------
The old activation used a_c(z)=||u_c(z)||^2, whose scale depended on feature
magnitude and class-wise variance.  The default activation is now the
subspace Mahalanobis energy

    a_c(z) = (1/d_c) * u_c(z)^T (Sigma_c + rho I)^{-1} u_c(z),

where u_c(z)=B_c^T(z-mu_c).  This is dimension-normalized and uses the
covariance statistics estimated in Layer 2.

2) Permutation-invariant aggregation of class reactions
-------------------------------------------------------
The old layer used an ordered matrix product over other classes.  Matrix
products do not generally commute, so the result may depend on arbitrary class
index order.  This file keeps the old ordered_product mode for ablation, but
adds two order-invariant modes:

    aggregation_mode='suppression'   (recommended main version)
        W_c(z) = [ I + sum_{c'!=c} (1 + eta * a_{c'}(z)) G_{c<-c'} ]^{-1}

    aggregation_mode='linear'        (old ablation)
        W_c(z)=W0_c^{1/2} [ I + eta * sum_{c'!=c} a_{c'}(z) G_{c<-c'} ] W0_c^{1/2}

    aggregation_mode='log_euclidean' (old ablation)
        H_c(z)=sum_{c'!=c} 0.5 * log[ I + eta * a_{c'}(z) G_{c<-c'} ]
        W_c(z)=W0_c^{1/2} exp[H_c(z)] exp[H_c(z)] W0_c^{1/2}

The recommended suppression version has a cleaner interpretation:
if a competing class c' is strongly activated, the directions in class-c
subspace that overlap with c' are suppressed more strongly.  The score remains

    score_c(z)=||z-mu_c||^2 - u_c(z)^T W_c(z) u_c(z),
    predict(z)=argmin_c score_c(z).

Here W_c(z) acts as a confidence/usable-projection matrix rather than a reward
amplifier.
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def _sqrtm_psd(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    evals, evecs = torch.linalg.eigh(0.5 * (M + M.t()))
    evals = evals.clamp(min=eps)
    return (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.t()


def _inv_psd(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Stable inverse of a symmetric positive definite matrix."""
    evals, evecs = torch.linalg.eigh(0.5 * (M + M.t()))
    evals = evals.clamp(min=eps)
    return (evecs * (1.0 / evals).unsqueeze(0)) @ evecs.t()


def _expm_sym_batch(H_batch: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Batch matrix exponential for symmetric matrices.

    Args:
        H_batch: [n, d, d]
    Returns:
        exp(H_batch): [n, d, d]
    """
    H_batch = 0.5 * (H_batch + H_batch.transpose(1, 2))
    evals, evecs = torch.linalg.eigh(H_batch)
    evals = evals.clamp(min=-50.0, max=50.0)  # numerical guard
    exp_evals = torch.exp(evals).clamp(min=eps)
    return (evecs * exp_evals.unsqueeze(1)) @ evecs.transpose(1, 2)


def compute_geometric_overlap_matrices(B_dict, num_classes):
    """
    Compute G_{c<-c'} = B_c^T P_{c'} B_c = B_c^T B_{c'} B_{c'}^T B_c.

    Returns:
        G_dict: {(c, c2): Tensor[d_c, d_c]}
    """
    G_dict = {}
    for c in range(num_classes):
        if c not in B_dict:
            continue
        B_c = B_dict[c]
        for c2 in range(num_classes):
            if c2 == c or c2 not in B_dict:
                continue
            K = B_c.t() @ B_dict[c2]
            G = K @ K.t()
            G_dict[(c, c2)] = 0.5 * (G + G.t())
    return G_dict


def compute_base_reaction_matrices(B_dict, num_classes, eps: float = 1e-8):
    """
    Compute W0_c = (I + sum_{c'!=c} G_{c<-c'})^{-1} and eigendecompose each G.
    """
    G_dict = compute_geometric_overlap_matrices(B_dict, num_classes)
    W0_dict, W0_sqrt_dict, G_eig_dict = {}, {}, {}

    for c in range(num_classes):
        if c not in B_dict:
            continue
        d_c = B_dict[c].shape[1]
        I_c = torch.eye(d_c, dtype=B_dict[c].dtype, device=B_dict[c].device)
        S = torch.zeros_like(I_c)
        for c2 in range(num_classes):
            if c2 == c or c2 not in B_dict:
                continue
            G = G_dict[(c, c2)]
            S = S + G
            evals, evecs = torch.linalg.eigh(G)
            G_eig_dict[(c, c2)] = (evals.clamp(min=0.0), evecs)

        W0 = _inv_psd(I_c + S, eps=eps)
        W0_dict[c] = W0
        W0_sqrt_dict[c] = _sqrtm_psd(W0, eps=eps)

    return W0_dict, W0_sqrt_dict, G_dict, G_eig_dict


def _compute_projection_and_activation(
    Z,
    subspace_result,
    num_classes,
    activation_mode: str = 'mahalanobis',
    activation_rho: float = 1e-6,
    max_activation_scale=None,
):
    """Compute u_c(z) and normalized activation a_c(z) for every class."""
    B_dict = subspace_result['B_dict']
    mu_dict = subspace_result['mu_dict']
    Sigma_dict = subspace_result.get('Sigma_dict', {})

    proj_dict = {}
    activation_dict = {}

    for c in range(num_classes):
        if c not in B_dict:
            continue
        B_c = B_dict[c]
        u_c = (Z - mu_dict[c]) @ B_c
        proj_dict[c] = u_c
        d_c = max(u_c.shape[1], 1)

        if activation_mode in ('mahalanobis', 'normalized', 'sigma'):
            if c not in Sigma_dict:
                raise KeyError(f"Sigma_dict missing class {c}; required for normalized activation.")
            sigma_diag = torch.diag(Sigma_dict[c]).to(dtype=Z.dtype, device=Z.device)
            # Add rho instead of only clamp, matching (Sigma_c + rho I)^{-1}.
            denom = (sigma_diag + activation_rho).clamp(min=activation_rho)
            a_c = (u_c.pow(2) / denom.unsqueeze(0)).sum(dim=1) / float(d_c)
        elif activation_mode == 'raw':
            a_c = u_c.pow(2).sum(dim=1)
        elif activation_mode in ('raw_mean', 'mean'):
            a_c = u_c.pow(2).sum(dim=1) / float(d_c)
        else:
            raise ValueError(
                f"Unknown activation_mode={activation_mode!r}. "
                "Use 'mahalanobis', 'raw', or 'raw_mean'."
            )

        if max_activation_scale is not None:
            a_c = a_c.clamp(max=float(max_activation_scale))
        activation_dict[c] = a_c

    return proj_dict, activation_dict


def compute_discriminative_rperp(
    Z,
    subspace_result,
    num_classes,
    eta_react=0.10,
    eps=1e-8,
    max_activation_scale=None,
    activation_mode='mahalanobis',
    activation_rho=1e-6,
    aggregation_mode='suppression',
):
    """
    Compute white-box cross-coupled geometric discriminant scores.

    Parameters:
        aggregation_mode:
            'suppression'    : recommended competition-aware suppression.
            'linear'         : old order-invariant first-order aggregation.
            'log_euclidean'  : old order-invariant log-Euclidean aggregation.
            'ordered_product': old ordered product, kept for ablation only.
        activation_mode:
            'mahalanobis'    : default normalized activation using Sigma_c.
            'raw'            : old ||u_c||^2 activation, kept for ablation.
            'raw_mean'       : ||u_c||^2 / d_c.

    Returns:
        R_disc: [n, C] discriminant scores. Lower is better.
        disc_weights: diagnostic dictionary.
    """
    n = Z.shape[0]
    B_dict = subspace_result['B_dict']
    mu_dict = subspace_result['mu_dict']

    if aggregation_mode not in ('suppression', 'linear', 'log_euclidean', 'ordered_product'):
        raise ValueError(
            f"Unknown aggregation_mode={aggregation_mode!r}. "
            "Use 'suppression', 'linear', 'log_euclidean', or 'ordered_product'."
        )

    W0_dict, W0_sqrt_dict, G_dict, G_eig_dict = compute_base_reaction_matrices(
        B_dict, num_classes, eps=eps
    )

    R_disc = torch.full((n, num_classes), 1e10, dtype=Z.dtype, device=Z.device)
    proj_dict, activation_dict = _compute_projection_and_activation(
        Z,
        subspace_result,
        num_classes,
        activation_mode=activation_mode,
        activation_rho=activation_rho,
        max_activation_scale=max_activation_scale,
    )

    sorted_classes = [c for c in range(num_classes) if c in B_dict]
    dyn_info_dict = {}

    for c in sorted_classes:
        u_c = proj_dict[c]
        diff = Z - mu_dict[c]
        total_sq = diff.pow(2).sum(dim=1)
        d_c = u_c.shape[1]
        I_c = torch.eye(d_c, dtype=Z.dtype, device=Z.device)
        W0_sqrt = W0_sqrt_dict[c]
        W0s = W0_sqrt.unsqueeze(0)
        other_classes = [c2 for c2 in sorted_classes if c2 != c]

        if aggregation_mode == 'suppression':
            # Recommended version:
            # W_c(z) = [ I + sum_{c'!=c} (1 + eta * a_{c'}(z)) G_{c<-c'} ]^{-1}
            #
            # Interpretation:
            # If z strongly activates a competing class c', then directions of
            # class-c subspace overlapping with c' receive stronger suppression.
            M_batch = I_c.unsqueeze(0).expand(n, -1, -1).clone()
            for c2 in other_classes:
                G = G_dict[(c, c2)]
                a_c2 = activation_dict[c2]
                coeff = 1.0 + eta_react * a_c2
                M_batch = M_batch + coeff.view(n, 1, 1) * G.unsqueeze(0)
            W_dyn_batch = torch.linalg.inv(0.5 * (M_batch + M_batch.transpose(1, 2)))

        elif aggregation_mode == 'linear':
            # Old ablation:
            # W = W0^1/2 [I + eta * sum a_{c'} G_{c<-c'}] W0^1/2.
            # This is order-invariant, but its interpretation is less clean:
            # competing activation can increase the projection reward.
            M_batch = I_c.unsqueeze(0).expand(n, -1, -1).clone()
            for c2 in other_classes:
                G = G_dict[(c, c2)]
                a_c2 = activation_dict[c2]
                M_batch = M_batch + eta_react * a_c2.view(n, 1, 1) * G.unsqueeze(0)
            W_dyn_batch = W0s @ M_batch @ W0s

        elif aggregation_mode == 'log_euclidean':
            # Scheme B: H = sum 1/2 log(I + eta*a*G); W = W0^1/2 exp(H) exp(H) W0^1/2
            H_batch = torch.zeros((n, d_c, d_c), dtype=Z.dtype, device=Z.device)
            for c2 in other_classes:
                evals_g, evecs_g = G_eig_dict[(c, c2)]
                a_c2 = activation_dict[c2]
                log_scale = 0.5 * torch.log(
                    (1.0 + eta_react * a_c2.unsqueeze(1) * evals_g.unsqueeze(0)).clamp(min=eps)
                )
                logF_half_batch = (evecs_g.unsqueeze(0) * log_scale.unsqueeze(1)) @ evecs_g.t().unsqueeze(0)
                H_batch = H_batch + logF_half_batch
            E_batch = _expm_sym_batch(H_batch, eps=eps)
            WE = W0s @ E_batch
            W_dyn_batch = WE @ E_batch @ W0s

        else:
            # Old ablation: ordered product over class index. Not permutation-invariant.
            A_batch = I_c.unsqueeze(0).expand(n, -1, -1).clone()
            for c2 in other_classes:
                evals_g, evecs_g = G_eig_dict[(c, c2)]
                a_c2 = activation_dict[c2]
                scale = torch.sqrt(
                    (1.0 + eta_react * a_c2.unsqueeze(1) * evals_g.unsqueeze(0)).clamp(min=eps)
                )
                F_sqrt_batch = (evecs_g.unsqueeze(0) * scale.unsqueeze(1)) @ evecs_g.t().unsqueeze(0)
                A_batch = A_batch @ F_sqrt_batch
            WA = W0s @ A_batch
            W_dyn_batch = WA @ WA.transpose(1, 2)

        W_dyn_batch = 0.5 * (W_dyn_batch + W_dyn_batch.transpose(1, 2))
        u_col = u_c.unsqueeze(2)
        quad = (u_col.transpose(1, 2) @ W_dyn_batch @ u_col).squeeze(-1).squeeze(-1)
        R_disc[:, c] = total_sq - quad
        dyn_info_dict[c] = W_dyn_batch

    disc_weights = {
        'base_matrices': W0_dict,
        'base_sqrt_matrices': W0_sqrt_dict,
        'dynamic_matrices': dyn_info_dict,
        'overlap_matrices': G_dict,
        'overlap_eig': G_eig_dict,
        'activations': activation_dict,
        'eta_react': eta_react,
        'eps': eps,
        'max_activation_scale': max_activation_scale,
        'activation_mode': activation_mode,
        'activation_rho': activation_rho,
        'aggregation_mode': aggregation_mode,
    }
    return R_disc, disc_weights


def analyze_discriminative_weights(disc_weights, num_classes=None, idx=None):
    """Analyze spectral properties of base and dynamic matrices."""
    W0_dict = disc_weights['base_matrices']
    Wdyn_dict = disc_weights['dynamic_matrices']

    cls_list = sorted(W0_dict.keys()) if num_classes is None else list(range(num_classes))
    result = {}
    for c in cls_list:
        if c not in W0_dict or c not in Wdyn_dict:
            continue
        W0 = W0_dict[c]
        evals0 = torch.linalg.eigvalsh(W0)
        W_dyn = Wdyn_dict[c]
        if idx is not None:
            idx_t = idx if isinstance(idx, torch.Tensor) else torch.tensor(idx, dtype=torch.long)
            W_dyn = W_dyn[idx_t]
        dyn_eigs = torch.linalg.eigvalsh(W_dyn)
        result[c] = {
            'base_min_eig': evals0.min().item(),
            'base_max_eig': evals0.max().item(),
            'dyn_avg_min_eig': dyn_eigs.min(dim=1).values.mean().item(),
            'dyn_avg_max_eig': dyn_eigs.max(dim=1).values.mean().item(),
            'dyn_max_eig': dyn_eigs.max().item(),
            'dim': W0.shape[0],
        }
    return result
