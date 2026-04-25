"""
Layer 3: Geometric White-Box Cross-Coupled Reactive Discriminant Layer
=====================================================================

Core formula
------------
For each class c, define
    u_c(z) = B_c^T (z - mu_c)                         [d_c]

Classification score:
    R_c(z) = ||z - mu_c||^2 - u_c(z)^T W_c(z) u_c(z)

where W_c(z) is the dynamic cross-coupled reaction matrix within class-c subspace:
    G_{c<-c'} = B_c^T P_{c'} B_c
               = B_c^T B_{c'} B_{c'}^T B_c          [d_c, d_c]

    a_{c'}(z) = ||P_{c'} (z - mu_{c'})||^2
              = ||u_{c'}(z)||^2                      scalar

    F_{c<-c'}(z) = I + eta * a_{c'}(z) * G_{c<-c'}

    W0_c = (I + sum_{c'!=c} G_{c<-c'})^{-1}

    W_c(z) = W0_c^{1/2}
             (prod^-> F_{c<-c'}(z)^{1/2})
             (prod^<- F_{c<-c'}(z)^{1/2})
             W0_c^{1/2}

Geometric meaning:
- G_{c<-c'}  : projection operator P_{c'} restricted to subspace S_c;
               measures geometric intrusion of class c' into class c
- a_{c'}(z)  : activation strength of sample z in class c' subspace
- F_{c<-c'}  : cross-coupled reaction factor, accumulated by matrix multiplication
- W_c(z)     : dynamic reaction matrix within subspace c
- All quantities depend on projection operators / subspace overlaps only,
  avoiding coordinate-sign-dependent decompositions

Performance optimization:
- Each G_{c<-c'} is pre-decomposed as G = Q Lambda Q^T (once)
- F^{1/2} = Q diag(sqrt(1 + eta*a*lambda_i)) Q^T (per sample: only diagonal scaling)
- The per-sample loop is replaced by batched diagonal operations
- No per-sample eigendecomposition is needed

White-box verification:
- No learnable parameters, no MLP, no gradient-based training
- Every step is an explicit closed-form matrix operation
- eta_react is a fixed scalar hyperparameter
"""

import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def _sqrtm_psd(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    evals, evecs = torch.linalg.eigh(M)
    evals = evals.clamp(min=eps)
    sqrt_evals = torch.sqrt(evals)
    return (evecs * sqrt_evals.unsqueeze(0)) @ evecs.t()


def _inv_psd(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Stable inverse of symmetric PSD matrix via eigendecomposition."""
    evals, evecs = torch.linalg.eigh(M)
    evals = evals.clamp(min=eps)
    inv_evals = 1.0 / evals
    return (evecs * inv_evals.unsqueeze(0)) @ evecs.t()


def compute_geometric_overlap_matrices(B_dict, num_classes):
    """
    Compute G_{c<-c'} = B_c^T P_{c'} B_c = K K^T.

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
            K = B_c.t() @ B_dict[c2]       # [d_c, d_c2]
            G = K @ K.t()                  # [d_c, d_c]
            G = 0.5 * (G + G.t())          # numerical symmetrization
            G_dict[(c, c2)] = G
    return G_dict


def compute_base_reaction_matrices(B_dict, num_classes, eps: float = 1e-8):
    """
    Compute base matrix W0_c = (I + sum_{c'!=c} G_{c<-c'})^{-1}.

    Also pre-decompose each G for fast per-sample F^{1/2} computation.

    Returns:
        W0_dict:          {c: Tensor[d_c, d_c]}
        W0_sqrt_dict:     {c: Tensor[d_c, d_c]}
        G_dict:           {(c, c2): Tensor[d_c, d_c]}
        G_eig_dict:       {(c, c2): (evals [d_c], evecs [d_c, d_c])}
    """
    G_dict = compute_geometric_overlap_matrices(B_dict, num_classes)
    W0_dict, W0_sqrt_dict = {}, {}
    G_eig_dict = {}

    for c in range(num_classes):
        if c not in B_dict:
            continue
        d_c = B_dict[c].shape[1]
        I_c = torch.eye(d_c, dtype=B_dict[c].dtype, device=B_dict[c].device)
        S = torch.zeros_like(I_c)
        for c2 in range(num_classes):
            if c2 == c or c2 not in B_dict:
                continue
            S = S + G_dict[(c, c2)]
            # Pre-decompose G for fast F^{1/2}
            evals, evecs = torch.linalg.eigh(G_dict[(c, c2)])
            evals = evals.clamp(min=0.0)  # G is PSD
            G_eig_dict[(c, c2)] = (evals, evecs)
        M = I_c + S
        W0 = _inv_psd(M, eps=eps)
        W0_sqrt = _sqrtm_psd(W0, eps=eps)
        W0_dict[c] = W0
        W0_sqrt_dict[c] = W0_sqrt

    return W0_dict, W0_sqrt_dict, G_dict, G_eig_dict


def compute_discriminative_rperp(
    Z,
    subspace_result,
    num_classes,
    eta_react=0.10,
    eps=1e-8,
    max_activation_scale=None,
):
    """
    Compute geometric white-box cross-coupled reactive discriminant scores.

    Optimized version: G is pre-decomposed as Q Lambda Q^T so that
    F^{1/2} = Q diag(sqrt(1 + eta*a*lambda_i)) Q^T can be computed
    without per-sample eigendecomposition.

    Parameters:
        Z:               [n, D]  smoothed node features
        subspace_result: Layer 2 output
        num_classes:     int
        eta_react:       cross-coupled reaction strength (scalar hyperparameter)
        eps:             numerical stability
        max_activation_scale: optional upper bound on a_{c'}(z)

    Returns:
        R_disc:       [n, C]
        disc_weights: dict with base/dynamic matrices, activations, overlap matrices
    """
    n = Z.shape[0]
    B_dict = subspace_result['B_dict']
    mu_dict = subspace_result['mu_dict']

    W0_dict, W0_sqrt_dict, G_dict, G_eig_dict = \
        compute_base_reaction_matrices(B_dict, num_classes, eps=eps)

    R_disc = torch.full((n, num_classes), 1e10, dtype=Z.dtype, device=Z.device)
    proj_dict = {}
    activation_dict = {}

    # Compute subspace coordinates and activation strengths for all classes
    for c in range(num_classes):
        if c not in B_dict:
            continue
        u_c = (Z - mu_dict[c]) @ B_dict[c]          # [n, d_c]
        proj_dict[c] = u_c
        a_c = (u_c ** 2).sum(dim=1)                 # [n]
        if max_activation_scale is not None:
            a_c = a_c.clamp(max=max_activation_scale)
        activation_dict[c] = a_c

    sorted_classes = [c for c in range(num_classes) if c in B_dict]

    # Store W_c(z) diagonal eigenvalues for analysis (compact representation)
    dyn_info_dict = {}

    for c in sorted_classes:
        u_c = proj_dict[c]                          # [n, d_c]
        diff = Z - mu_dict[c]                       # [n, D]
        total_sq = (diff ** 2).sum(dim=1)           # [n]
        d_c = u_c.shape[1]
        W0_sqrt = W0_sqrt_dict[c]                   # [d_c, d_c]

        other_classes = [c2 for c2 in sorted_classes if c2 != c]

        # ---- Optimized batch computation ----
        # For each pair (c, c2), G = Q Lambda Q^T is pre-decomposed.
        # F^{1/2} = Q diag(sqrt(1 + eta*a*lambda_i)) Q^T
        #
        # The product A = prod_{c2} F_{c<-c2}^{1/2} is accumulated per sample.
        # Key insight: A is a d_c x d_c matrix, and d_c is small (e.g. 12).
        # We vectorize over samples (n) while keeping d_c x d_c matrix ops.

        # Initialize A_batch = [n, d_c, d_c], each as identity
        I_c = torch.eye(d_c, dtype=Z.dtype, device=Z.device)
        A_batch = I_c.unsqueeze(0).expand(n, -1, -1).clone()  # [n, d_c, d_c]

        for c2 in other_classes:
            evals_g, evecs_g = G_eig_dict[(c, c2)]  # evals [d_c], evecs [d_c, d_c]
            a_c2 = activation_dict[c2]               # [n]

            # sqrt(1 + eta * a * lambda_i) for each sample and each eigenvalue
            # shape: [n, d_c]
            scale = torch.sqrt(
                (1.0 + eta_react * a_c2.unsqueeze(1) * evals_g.unsqueeze(0)).clamp(min=eps)
            )

            # F^{1/2} = Q diag(scale) Q^T
            # For batch: F_sqrt_batch[i] = evecs_g @ diag(scale[i]) @ evecs_g^T
            # Efficient: (evecs_g * scale[i]) @ evecs_g^T
            # evecs_g: [d_c, d_c], scale: [n, d_c]
            # F_sqrt_batch = evecs_g[None,:,:] * scale[:,:,None]  =>  [n, d_c, d_c]
            #              @ evecs_g.t()[None,:,:]                =>  [n, d_c, d_c]
            F_sqrt_batch = (evecs_g.unsqueeze(0) * scale.unsqueeze(1)) @ evecs_g.t().unsqueeze(0)
            # [n, d_c, d_c]

            # Accumulate: A = A @ F_sqrt
            A_batch = A_batch @ F_sqrt_batch

        # W_dyn = W0_sqrt @ A @ A^T @ W0_sqrt
        # [n, d_c, d_c]
        W0s = W0_sqrt.unsqueeze(0)  # [1, d_c, d_c]
        WA = W0s @ A_batch          # [n, d_c, d_c]
        W_dyn_batch = WA @ WA.transpose(1, 2)  # [n, d_c, d_c], this is W0^½ A A^T W0^½

        # Symmetrize
        W_dyn_batch = 0.5 * (W_dyn_batch + W_dyn_batch.transpose(1, 2))

        # R_c(z) = ||z - mu_c||^2 - u_c^T W_c u_c
        # u_c: [n, d_c] -> [n, d_c, 1]
        u_col = u_c.unsqueeze(2)                    # [n, d_c, 1]
        quad = (u_col.transpose(1, 2) @ W_dyn_batch @ u_col).squeeze()  # [n]
        R_disc[:, c] = total_sq - quad

        dyn_info_dict[c] = W_dyn_batch  # store for analysis if needed

    disc_weights = {
        'base_matrices': W0_dict,
        'base_sqrt_matrices': W0_sqrt_dict,
        'dynamic_matrices': dyn_info_dict,   # [n, d_c, d_c] tensor per class
        'overlap_matrices': G_dict,
        'overlap_eig': G_eig_dict,
        'activations': activation_dict,
        'eta_react': eta_react,
        'eps': eps,
        'max_activation_scale': max_activation_scale,
    }

    return R_disc, disc_weights


def analyze_discriminative_weights(disc_weights, num_classes=None, idx=None):
    """
    Analyze spectral properties of base and dynamic matrices.
    """
    W0_dict = disc_weights['base_matrices']
    Wdyn_dict = disc_weights['dynamic_matrices']

    if num_classes is None:
        cls_list = sorted(W0_dict.keys())
    else:
        cls_list = list(range(num_classes))

    result = {}
    for c in cls_list:
        if c not in W0_dict:
            continue
        W0 = W0_dict[c]
        evals0 = torch.linalg.eigvalsh(W0)

        W_dyn = Wdyn_dict[c]  # [n, d_c, d_c]
        if idx is not None:
            idx_t = idx if isinstance(idx, torch.Tensor) else torch.tensor(idx, dtype=torch.long)
            W_dyn = W_dyn[idx_t]

        # Batch eigenvalues: [m, d_c]
        dyn_eigs = torch.linalg.eigvalsh(W_dyn)

        result[c] = {
            'base_min_eig': evals0.min().item(),
            'base_max_eig': evals0.max().item(),
            'dyn_avg_min_eig': dyn_eigs.min(dim=1).values.mean().item(),
            'dyn_avg_max_eig': dyn_eigs.max(dim=1).values.mean().item(),
            'dim': W0.shape[0],
        }
    return result
