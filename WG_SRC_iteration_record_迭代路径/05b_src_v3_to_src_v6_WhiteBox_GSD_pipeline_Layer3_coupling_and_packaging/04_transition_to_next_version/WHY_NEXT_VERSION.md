# Why Move from src_v3-src_v6 to src_v7?

`src_v3` to `src_v6` successfully turned the early layered white-box idea into a WhiteBox-GSD pipeline and reusable package.

However, these versions also exposed an important limitation.

The Layer3 discriminative / coupling route mainly reasoned about class subspaces after graph smoothing. This was useful, but it did not yet fully address heterophilic graphs where neighbors may belong to different classes.

The key unresolved question became:

**What graph signals should be constructed before fitting class subspaces?**

This led to `src_v7`, which focused on Chameleon and tested explicit multihop graph signals.

The central discovery in `src_v7` was that high-pass graph differences such as:

- `X - PX`
- `PX - P²X`

are important for heterophilic graphs.

Therefore, the project moved from Layer3 coupling inside the subspace pipeline to a more explicit graph-signal dictionary design. This transition eventually led to the final WG-SRC nine-block graph-signal dictionary.