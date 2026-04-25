# Orthogonality Comparison Results

This folder contains the cleaned results of the `mcr2_orth.py` experiment.

## Compared modes

1. CE only
2. CE + MCR2
3. CE + MCR2 + explicit orthogonality penalty

## Main files

- `experiment_summary.txt`  
  Clean summary of the experiment setting, results, and key findings.

- `experiment_log.md`  
  Full experiment log.

- `comparison_3way.png`  
  Visual comparison of CE, MCR2, and MCR2 + Orth.

- `ce/metrics.json`  
  Metrics for CE only.

- `mcr2/metrics.json`  
  Metrics for CE + MCR2.

- `mcr2_orth/metrics.json`  
  Metrics for CE + MCR2 + Orth.

## Main conclusion

MCR2 increases Delta-R strongly, but high Delta-R does not imply low subspace coherence. Adding explicit orthogonality penalty did not meaningfully reduce last-hidden-layer coherence and caused a large accuracy drop.

This result motivated the later transition from enforcing geometry inside neural representations to explicitly constructing class PCA subspaces in a white-box pipeline.