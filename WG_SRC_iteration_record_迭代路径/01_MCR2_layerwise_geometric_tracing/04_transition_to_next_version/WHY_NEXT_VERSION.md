# Why Move to the Next Stage?

The first-stage MCR2 experiments showed an important negative result.

MCR2 successfully increased Delta-R and changed the representation geometry, but it did not guarantee inter-class subspace orthogonality. The explicit orthogonality penalty also failed to provide a clean solution: it did not meaningfully reduce last-hidden-layer coherence and caused a large drop in test accuracy.

This means that directly forcing neural hidden layers to satisfy ideal subspace geometry is unstable.

Therefore, the next research step was to move from "training a neural network to form ideal subspaces" toward a more controllable white-box direction:

1. construct graph signals explicitly;
2. fit class subspaces explicitly;
3. compute residuals explicitly;
4. use closed-form classification.

This transition eventually led to the layered closed-form pipeline and then to WG-SRC.