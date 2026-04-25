# HASDC / Raw-Smooth Dual Channel / Signed Coupling Results

This folder records the intermediate experiments between `src_v5` and `src_v6`.

These experiments were designed to test whether the class-subspace interaction should be treated only as negative competition, or whether a more flexible coupling mechanism is needed.

## Background

The earlier discriminative Layer3 mainly used a competition-aware suppression idea:

`subspace overlap -> suppress`

This assumes that overlapping class-subspace directions are harmful.

However, this assumption can be too rigid. Some shared directions may carry useful semantic information, especially when raw features and graph-smoothed features behave differently.

Therefore, this stage explored a more flexible idea:

1. raw feature channel;
2. smoothed feature channel;
3. signed or adaptive coupling between class subspaces;
4. Layer3 correction beyond simple suppression.

## Included code

The key code is stored in:

`01_algorithm_code/HASDC_and_Layer3_scripts/`

Important files include:

- `layer3_hasdc.py`
- `layer3_discriminative.py`
- `run_whitebox_hasdc.py`
- `whitebox_src_v5_bridge.py`
- `whitebox_v5_adapter.py`
- `diagnose_and_new_layer3.py`
- `v6_layer3_fast/`

## Included result folders

The result and audit files are stored in:

`03_statistics_and_audits/HASDC_raw_smooth_signed_coupling_audits/`

Included result groups:

- `results_hasdc_freq_test`
- `results_hasdc_phase_a`
- `results_hasdc_phase_b`
- `results_hasdc_phase_b2`
- `results_hasdc_phase_c`
- `results_hasdc_test`
- `results_hasdc_test2`
- `results_hasdc_v3`
- `results_v6_layer3_direct_fast`
- `results_v6_layer3_fixed_base_fast`

## How to read the evidence

For each HASDC result folder, the most important files are usually:

- `hasdc_summary.csv`
- `hasdc_best_by_val.csv`
- `hasdc_all_grid.csv`

These files show whether a raw/smooth or coupling variant improved validation or test behavior.

For v6 Layer3 results, read:

- `v6_layer3_direct_fast_all_partial.csv`
- `v6_layer3_fixed_base_summary.csv`
- `v6_layer3_fixed_base_best_by_val.csv`

if available.

## Main interpretation

This stage should be described as an exploratory bridge, not as the final successful mechanism.

The evidence supports the following research conclusion:

1. simple subspace-overlap suppression is too rigid;
2. class-subspace interaction may need a more flexible coupling view;
3. raw and smoothed channels may contain complementary information;
4. however, the HASDC / Layer3 coupling route was still not clean enough as the final method;
5. this motivated the next transition: explicitly construct graph-signal blocks before subspace fitting.

That transition led to `src_v7`, where the project moved from implicit raw/smooth coupling to explicit graph-signal decomposition:

- raw signal `X`;
- low-pass signal `PX`, `P²X`;
- high-pass differences `X-PX`, `PX-P²X`.

This is the conceptual bridge from `src_v5/src_v6` to the final WG-SRC graph-signal dictionary.