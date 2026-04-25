# src_v3 to src_v6: WhiteBox-GSD Pipeline, Layer3 Coupling, and Packaging

This stage fills the gap between `src_v1` and `src_v7`.

The project did not jump directly from the initial layered closed-form pipeline to the final Chameleon high-pass breakthrough. Between them, `src_v3` to `src_v6` gradually turned the early white-box subspace idea into a more systematic WhiteBox-GSD pipeline.

## Why this stage matters

`src_v1` established the layered closed-form idea, but it was still an early prototype.

`src_v3` to `src_v6` did three important things:

1. turned the layered idea into a configurable pipeline;
2. tested geometric-reactive and discriminative Layer3 mechanisms;
3. packaged the method into a reusable model-style interface.

This stage is therefore important for showing the engineering and conceptual bridge between the early closed-form white-box pipeline and the later WG-SRC design.

---

## src_v3: Pipeline construction

`src_v3` organized the earlier layer-wise idea into a clearer pipeline.

Main components include:

- `data_loader.py`
- `layer1_tikhonov.py`
- `layer2_subspace.py`
- `layer3_discriminative.py`
- `framework/configs.py`
- `framework/pipeline.py`
- `run_all.py`

The main purpose was not yet to produce the final performance breakthrough, but to make the method modular, configurable, and easier to run.

Research meaning:

`src_v3` converted the early white-box subspace idea from separate scripts into a pipeline.

---

## src_v4: Cora geometric-reactive validation

`src_v4` tested the pipeline on Cora and produced a recorded result file:

- `cora_geometric_reactive_result.json`

It also included a run note:

- `README_run_note.txt`

The purpose of this version was to verify that the geometric-reactive / discriminative Layer3 mechanism could run on a real citation graph and produce recorded results.

Research meaning:

`src_v4` was the first more concrete Cora validation version of this WhiteBox-GSD line.

---

## src_v5: WhiteBox-GSD core and Layer3 coupling

`src_v5` is the most important version in this intermediate stage.

It contained both:

- `layer3_discriminative.py`
- `layer3_hasdc.py`

The earlier Layer3 idea treated subspace overlap mainly as competition:

`subspace overlap -> suppress`

This is a reasonable idea when overlapping directions represent ambiguity, but it can be harmful when shared directions carry useful semantic information.

The later HASDC-style direction started to treat subspace interaction more flexibly:

- some coupling directions may need suppression;
- some coupling directions may be useful and should be preserved or promoted;
- raw and smoothed channels may contain complementary information.

This is an important precursor to the later raw / low-pass / high-pass graph-signal decomposition.

Research meaning:

`src_v5` revealed that a simple "overlap means competition" assumption is too rigid. This motivated a more explicit graph-signal decomposition later.

---

## src_v6: Packaging and reusable interface

`src_v6` reorganized the method into a package-like structure:

- `model.py`
- `pyproject.toml`
- `requirements.txt`
- `__init__.py`
- `examples/grid_search.py`
- `examples/synthetic_demo.py`

This version was less about a new mathematical mechanism and more about usability, reproducibility, and a reusable model interface.

Research meaning:

`src_v6` shows the transition from exploratory scripts to a reusable method package.

---

## Main conclusion of this stage

This stage showed that the closed-form white-box subspace idea could be organized into a real pipeline and package.

However, the Layer3 suppression / coupling route also exposed a limitation:

class-subspace interaction is not always negative competition.

This pushed the next stage toward a more explicit representation of graph signals, especially the distinction between:

- raw feature signal;
- low-pass smoothed signal;
- high-pass difference signal.

That transition led to `src_v7`, where Chameleon experiments made high-pass graph differences central.
---

## HASDC / raw-smooth dual channel result evidence

The raw / smooth dual-channel and signed-coupling idea is supported by the HASDC and v6 Layer3 experiment records added in this stage.

Relevant folders:

- `02_results/HASDC_raw_smooth_signed_coupling_results`
- `03_statistics_and_audits/HASDC_raw_smooth_signed_coupling_audits`
- `01_algorithm_code/HASDC_and_Layer3_scripts`

These files document the intermediate attempts to move beyond simple Layer3 suppression.

The key research conclusion is not that HASDC became the final winning method. Instead, the result is more important as a transition:

- simple suppression of class-subspace overlap was too rigid;
- raw and smoothed features appeared to require separate treatment;
- class-subspace interaction needed a more explicit signal decomposition;
- this pushed the project toward the explicit high-pass / low-pass graph-signal dictionary in `src_v7`.

Therefore, this stage is the bridge between WhiteBox-GSD Layer3 coupling and the later WG-SRC graph-signal dictionary.