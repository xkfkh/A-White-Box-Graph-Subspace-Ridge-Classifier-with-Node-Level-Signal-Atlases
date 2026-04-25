# Final paper figures and tables source package

This package was extracted from the uploaded LaTeX archive:
`A_White_Box_Graph_Subspace__Ridge_Classifier_with_Node_Level_Signal_Atlases__11_ (3).zip`.

## Contents

- `figures/pdf/`: final external PDF figure files used by `main.tex`.
- `figures/tikz/`: TikZ source extracted from `main.tex` for Figure 9, which is embedded in the LaTeX source rather than stored as an external PDF.
- `tables/latex/`: final LaTeX table source files. Tables 3 and 6 were inline in `main.tex`; they were extracted into standalone `.tex` files.
- `tables/csv/`: machine-readable CSV copies of the final table values.
- `paper_latex/`: `main.tex` and `math_commands.tex` copied from the LaTeX package.
- `figure_source_map.csv`: mapping from paper figure number to source file.
- `table_source_map.csv`: mapping from paper table number to LaTeX and CSV source files.
- `MANIFEST.csv`: file size and SHA256 hash manifest.

## Notes

- Figure 9 is embedded as TikZ inside `main.tex`; its extracted source is `figures/tikz/figure9_atlas_guided_diagnosis_tikz.tex`.
- Tables 3 and 6 are embedded directly in `main.tex`; their extracted sources are:
  - `tables/latex/paired_effect_table.tex`
  - `tables/latex/atlas_diagnostic_guidance_table.tex`
- The CSV files are provided for repository users who want to audit the numerical table values without parsing LaTeX.
