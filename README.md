# Operational Feature Fingerprints of Graph Datasets via a White-Box Signal-Subspace Probe

**A first-author-led white-box graph learning project that turns node prediction into dataset diagnosis.**

This repository contains the implementation, experiment package, atlas/fingerprint analysis, paper, and full research-iteration record for **WG-SRC**, a white-box signal–subspace probe developed for the paper:

**Operational Feature Fingerprints of Graph Datasets via a White-Box Signal-Subspace Probe**

The project was led by **Xiong Yuchen** as first author, with **Yeap Swee Keong** and **Ban Zhen Hong** as corresponding supervisors.

Rather than learning an opaque message-passing representation, WG-SRC constructs a named graph-signal dictionary, selects Fisher coordinates, fits class-wise PCA subspaces, solves closed-form multi-alpha Ridge classifiers, and fuses PCA/Ridge scores by validation.

The key point is dual use: the same fitted white-box scaffold both predicts node labels and produces a node-level mechanism atlas. By aggregating this atlas, WG-SRC yields operational feature fingerprints of graph datasets, describing raw-feature reliance, low-pass propagation reliance, high-pass sensitivity, class-subspace complexity, and ridge-boundary dependence.

> The main contribution is not only an accurate white-box node classifier, but a fitted predictive scaffold that converts graph datasets into measurable operational fingerprints.

---

## Research Log


This repository includes not only the final WG-SRC scaffold, but also the first-author-led research process that produced it.



The research trajectory records several major stages:

1. the project moved from MCR² / ReduNet representation geometry to white-box GNN unfolding attempts;
2. the work then shifted toward a cleaner closed-form graph signal–subspace classifier;
3. the Chameleon experiments revealed the importance of high-pass graph differences;
4. the model was further modularized, so that improving individual modules could improve the whole scaffold;
5. later versions analyzed model weaknesses not only through accuracy, but also through hard-node statistics, parameter effects, per-class PCA dimension traces, pairwise-specialist overfitting checks, and other statistical audits;
6. after the final model was optimized, statistical probes were embedded according to the model’s own white-box structure, producing dataset fingerprints and evaluation features;
7. these fingerprints were then used to further attempt to decompose and understand how components of previous black-box models behave on datasets with different fingerprint characteristics.


---

## Start Here

For a quick understanding of the research process:

- [Research Process Quick Overview English](./Research_Process_and_Iteration/研究历程与思路/01_Research_Quick_Overview_English_strict.md)
- [Research Process Quick Overview 中文](./Research_Process_and_Iteration/研究历程与思路/01_科研过程速读_中文.md)
- [Research Process Quick Overview 繁體中文](./Research_Process_and_Iteration/研究历程与思路/01_科研過程速讀_繁體中文(1).md)
- [Full Research Narrative English](./Research_Process_and_Iteration/研究历程与思路/02_Full_Research_Narrative_English_strict_long.md)
- [Complete Iteration Logic 中文](./Research_Process_and_Iteration/研究历程与思路/02_WG-SRC_src_v16c完整科研迭代逻辑.md)
- [Complete Iteration Logic 繁體中文](./Research_Process_and_Iteration/研究历程与思路/02_WG-SRC_src_v16c完整科研迭代邏輯(1).md)

For the final algorithm and experiments:

- `src_v16c_paper_experiments/`
- `paperexp/core.py`
- `run_all_experiments.py`

For the paper:

- [`A_White_Box_Graph_Subspace__Ridge_Classifier_with_Node_Level_Signal_Atlases.pdf`](./A_White_Box_Graph_Subspace__Ridge_Classifier_with_Node_Level_Signal_Atlases.pdf)

---

## Core Idea

WG-SRC uses:

1. an explicit multi-block graph-signal dictionary;
2. Fisher coordinate selection;
3. class-wise PCA subspace residuals;
4. closed-form multi-alpha Ridge classification;
5. validation-selected PCA/Ridge fusion;
6. node-level signal atlas and dataset-level operational fingerprint.

The final goal is not merely to classify nodes, but to expose **which mechanisms a dataset uses during classification**.
