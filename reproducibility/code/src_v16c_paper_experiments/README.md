# src_v16c paper experiment suite

This package contains reproducible experiment scripts for the paper-stage evaluation of `src_v16c`.

## Core protocol guarantees

1. **No test leakage**: test accuracy is never used for hyper-parameter selection.
2. **Train-only feature selection**: Fisher feature selection uses training nodes only.
3. **Train-only PCA/Ridge fitting**: PCA class subspaces and Ridge closed-form coefficients use training nodes only.
4. **Val-only selection**: all model choices are selected by validation accuracy / validation objective only.
5. **Mask preservation**: fixed Geom-GCN masks are used where available; random/few-shot masks are saved for external baseline reuse.

## Datasets

Default datasets:

```text
chameleon, squirrel, cornell, texas, wisconsin, amazon-photo, amazon-computers
```

Actor is excluded by default because it may be large. Add `--datasets actor` manually if needed.

The scripts auto-detect the root like:

```text
D:\桌面\MSR实验复现与创新\planetoid\data
```

or you can pass:

```bash
--data-root "D:\桌面\MSR实验复现与创新\planetoid\data"
```

## Main one-click run

Smoke test:

```bash
python run_all_experiments.py --skip-e11
```

Full run:

```bash
python run_all_experiments.py --full
```

Full run with explicit data root:

```bash
python run_all_experiments.py --full --data-root "D:\桌面\MSR实验复现与创新\planetoid\data"
```

## Individual scripts

### E01 baseline table aggregation

```bash
python scripts/run_E01_baseline_table_aggregate.py
```

### E03 random split stability

```bash
python scripts/run_E03_random_split_stability.py --repeats 5 --grid default
```

### E05 ablation

```bash
python scripts/run_E05_ablation.py --splits 10 --grid fast
```

Use `--grid default` for a heavier final paper run.

### E07 parameter scan

```bash
python scripts/run_E07_param_scan.py --datasets chameleon --splits 10
```

### E08 interpretability

```bash
python scripts/run_E08_interpretability.py --splits 10 --grid default
```

### E09 efficiency

```bash
python scripts/run_E09_efficiency.py --splits 3 --grid fast
```

### E11 sample efficiency

```bash
python scripts/run_E11_sample_efficiency.py --repeats 5 --grid fast
```

The E11 script also writes exact few-shot masks under:

```text
results/E11_sample_efficiency/fewshot_masks/
```

Use these masks for external PyG baselines to guarantee identical train/val/test splits.

## Outputs

Each experiment writes rows and summaries under `results/`.

Important files:

- `E03_src_v16c_summary.csv`
- `E03_combined_summary.csv`
- `E05_ablation_summary.csv`
- `E07_all_candidates.csv`
- `E08_feature_block_fisher_importance.csv`
- `E09_src_v16c_efficiency_summary.csv`
- `E11_learning_curve_summary.csv`

## Notes on baseline comparison

The package includes your uploaded baseline CSVs under `baselines/`. Scripts aggregate them when relevant, but `src_v16c` itself is always run independently and never tuned using baseline results.
