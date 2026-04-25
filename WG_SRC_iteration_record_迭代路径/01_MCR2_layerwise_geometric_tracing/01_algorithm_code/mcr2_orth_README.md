# MCR2 Orthogonality Code

This folder contains the code used in the first-stage MCR2 representation-geometry experiment.

## Main files

- `mcr2_trace_fashionmnist.py`  
  Used for tracing layer-wise representation geometry, including coding-rate behavior, Delta-R, effective rank, and subspace coherence.

- `mcr2_orth.py`  
  Used for comparing CE, CE + MCR2, and CE + MCR2 + explicit orthogonality penalty.

## Purpose

The goal of this stage was to test whether MCR2-style objectives can improve representation geometry and whether an additional orthogonality penalty can make class subspaces more orthogonal.

## Main conclusion

MCR2 significantly increases Delta-R, but it does not automatically enforce inter-class subspace orthogonality. A direct orthogonality penalty can interfere with classification, motivating the later move toward explicit class-subspace modeling instead of forcing neural hidden layers to become ideal subspaces.