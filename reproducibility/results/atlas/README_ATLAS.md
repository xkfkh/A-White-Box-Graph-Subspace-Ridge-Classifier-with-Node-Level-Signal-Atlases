# Atlas Reproduction Notes

Final atlas tables and figures use only the six final paper datasets:

- amazon-computers
- amazon-photo
- chameleon
- cornell
- texas
- wisconsin

Squirrel and actor are excluded from final paper atlas tables.

The family-size-adjusted signal mixture is computed by first averaging block evidence within each family:

- raw: P_X
- low-pass: P_PrX, P_Pr2X, P_Pr3X, P_PsX, P_Ps2X
- high-pass: P_XminusPrX, P_PrXminusPr2X, P_XminusPsX

Then the three family means are normalized across raw, low-pass, and high-pass.

The final generated tables are:

- final_tables/table5_signal_family_mixture.csv
- final_tables/figure8_highpass_error_shift.csv
