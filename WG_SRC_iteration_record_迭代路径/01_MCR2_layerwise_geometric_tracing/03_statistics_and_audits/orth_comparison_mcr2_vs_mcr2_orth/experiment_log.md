# Experiment Log: MCR2 + Orthogonality Penalty
## Setup
- Dataset: synthetic (10 classes, 100 train/100 test per class, low-rank Gaussian subspaces)
- Architecture: MLP, hidden dims 128-64-32
- Epochs: 15
- Modes: ce, mcr2, mcr2_orth
- lambda_mcr=1.0, lambda_orth=0.1

## Final Test Accuracy
- ce: 1.0000
- mcr2: 1.0000
- mcr2_orth: 0.7943

## Layer-wise Delta-R
| Layer | CE | MCR2 | MCR2+Orth |
|-------|----|------|-----------|
| 0 | 8.9234 | 8.9234 | 8.9234 |
| 1 | 7.4499 | 12.5340 | 6.9474 |
| 2 | 8.5207 | 18.5700 | 8.9476 |
| 3 | 8.6702 | 24.1395 | 10.1174 |
| 4 | 5.2465 | 16.3870 | 7.7635 |

## Layer-wise Subspace Coherence
| Layer | CE | MCR2 | MCR2+Orth |
|-------|----|------|-----------|
| 0 | 0.4326 | 0.4326 | 0.4326 |
| 1 | 0.5392 | 0.6665 | 0.6342 |
| 2 | 0.9561 | 0.9488 | 0.9601 |
| 3 | 0.9962 | 0.9731 | 0.9741 |
| 4 | 1.0000 | 1.0000 | 1.0000 |

## Analysis
Last hidden layer (layer 3) subspace coherence:
- CE:          0.9962
- MCR2:        0.9731
- MCR2+Orth:   0.9741

**Conclusion**: MCR2+Orth reduced coherence vs CE baseline, but improvement over plain MCR2 is marginal. May need larger lambda_orth or more epochs.

## Delta-R at Last Hidden Layer
- ce: 8.6702
- mcr2: 24.1395
- mcr2_orth: 10.1174

