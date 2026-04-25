from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

ROOT = Path(r"D:\桌面\复现")

def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None

def acc_to_percent(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    if s.max() <= 1.5:
        return s * 100.0
    return s

# ------------------------------------------------------------
# Paired stability from authoritative workbook
# ------------------------------------------------------------
wb = ROOT / "results/main_table/paper_main_table_final_corrected_baseline.xlsx"
paired_dir = ROOT / "results/paired_stability"
paired_dir.mkdir(parents=True, exist_ok=True)

if wb.exists():
    src = pd.read_excel(wb, sheet_name="Raw_SRC_V16C_Repeats")
    base = pd.read_excel(wb, sheet_name="Raw_Strongest_Baseline_Repeats")
    src = norm_cols(src)
    base = norm_cols(base)

    dcol_s = find_col(src, ["dataset"])
    rcol_s = find_col(src, ["repeat", "rep"])
    acol_s = find_col(src, ["test_acc", "test"])
    dcol_b = find_col(base, ["dataset"])
    rcol_b = find_col(base, ["repeat", "rep"])
    acol_b = find_col(base, ["test_acc", "test"])

    src2 = src[[dcol_s, rcol_s, acol_s]].copy()
    base2 = base[[dcol_b, rcol_b, acol_b]].copy()
    src2.columns = ["dataset", "repeat", "src_test_acc"]
    base2.columns = ["dataset", "repeat", "baseline_test_acc"]

    src2["src_test_acc"] = acc_to_percent(src2["src_test_acc"])
    base2["baseline_test_acc"] = acc_to_percent(base2["baseline_test_acc"])

    m = src2.merge(base2, on=["dataset", "repeat"], how="inner")
    m["delta_pp"] = m["src_test_acc"] - m["baseline_test_acc"]
    m["src_win"] = m["delta_pp"] > 0
    m = m.sort_values(["dataset", "repeat"])
    m.to_csv(paired_dir / "paired_split_differences_final.csv", index=False, encoding="utf-8-sig")

    rows = []
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        have_scipy = False

    for ds, g in m.groupby("dataset"):
        vals = g["delta_pp"].to_numpy(dtype=float)
        n = len(vals)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        sem = sd / math.sqrt(n) if n > 1 else float("nan")
        if have_scipy and n > 1:
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        else:
            ci = (mean - 1.96 * sem, mean + 1.96 * sem)
        rows.append({
            "dataset": ds,
            "wins": f"{int((vals > 0).sum())}/{n}",
            "mean_delta_pp": mean,
            "median_delta_pp": float(np.median(vals)),
            "min_delta_pp": float(np.min(vals)),
            "max_delta_pp": float(np.max(vals)),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "paired_dz": mean / sd if sd and not math.isnan(sd) and sd != 0 else float("nan")
        })

    eff = pd.DataFrame(rows)

    # Dataset-level aggregate over six dataset means
    means = eff["mean_delta_pp"].to_numpy(dtype=float)
    n = len(means)
    mean = float(np.mean(means))
    sd = float(np.std(means, ddof=1))
    sem = sd / math.sqrt(n)
    if have_scipy and n > 1:
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        ttest_p = float(stats.ttest_1samp(means, 0.0).pvalue)
        wilcoxon_p = float(stats.wilcoxon(means, alternative="two-sided").pvalue)
        sign_p = float(stats.binomtest(int((means > 0).sum()), n, 0.5, alternative="two-sided").pvalue)
    else:
        ci = (mean - 1.96 * sem, mean + 1.96 * sem)
        ttest_p = float("nan")
        wilcoxon_p = float("nan")
        sign_p = float("nan")

    agg = pd.DataFrame([{
        "dataset": "Dataset-level aggregate",
        "wins": f"{int((means > 0).sum())}/{n}",
        "mean_delta_pp": mean,
        "median_delta_pp": float(np.median(means)),
        "min_delta_pp": float(np.min(means)),
        "max_delta_pp": float(np.max(means)),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "paired_dz": mean / sd if sd != 0 else float("nan"),
        "ttest_p": ttest_p,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p
    }])

    eff["ttest_p"] = ""
    eff["wilcoxon_p"] = ""
    eff["sign_test_p"] = ""
    pd.concat([eff, agg], ignore_index=True).to_csv(
        paired_dir / "paired_effect_summary_final.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ------------------------------------------------------------
# Atlas Table 5 and Figure 8 high-pass shift
# ------------------------------------------------------------
atlas_dir = ROOT / "results/atlas"
node_root = atlas_dir / "node_signal_barcode_slim"
final_tables = atlas_dir / "final_tables"
final_tables.mkdir(parents=True, exist_ok=True)

datasets = [
    ("amazon-computers", "Amazon-Computers"),
    ("amazon-photo", "Amazon-Photo"),
    ("chameleon", "Chameleon"),
    ("cornell", "Cornell"),
    ("texas", "Texas"),
    ("wisconsin", "Wisconsin"),
]

table5_rows = []
shift_rows = []

for ds_slug, ds_name in datasets:
    f = node_root / ds_slug / "split_0" / "node_graph_signal_barcode.csv"
    if not f.exists():
        continue

    df = pd.read_csv(f)

    cols = {
        "raw": ["P_X"],
        "low": ["P_PrX", "P_Pr2X", "P_Pr3X", "P_PsX", "P_Ps2X"],
        "high": ["P_XminusPrX", "P_PrXminusPr2X", "P_XminusPsX"],
    }

    missing = [c for group in cols.values() for c in group if c not in df.columns]
    if missing:
        print("Missing atlas columns", ds_slug, missing)
        continue

    raw_e = df[cols["raw"]].mean(axis=1)
    low_e = df[cols["low"]].mean(axis=1)
    high_e = df[cols["high"]].mean(axis=1)
    denom = raw_e + low_e + high_e + 1e-12

    df["_raw_family"] = raw_e / denom
    df["_low_family"] = low_e / denom
    df["_high_family"] = high_e / denom

    table5_rows.append({
        "dataset": ds_name,
        "ntest": len(df),
        "raw_percent": 100 * df["_raw_family"].mean(),
        "low_pass_percent": 100 * df["_low_family"].mean(),
        "high_pass_percent": 100 * df["_high_family"].mean(),
    })

    correct_col = None
    for cand in ["correct", "is_correct", "final_correct", "pred_correct", "y_correct"]:
        if cand in df.columns:
            correct_col = cand
            break

    if correct_col is not None:
        cc = df[correct_col].astype(int)
        high_correct = df.loc[cc == 1, "_high_family"].mean()
        high_wrong = df.loc[cc == 0, "_high_family"].mean()
        shift_rows.append({
            "dataset": ds_name,
            "wrong_minus_correct_highpass_pp": 100 * (high_wrong - high_correct),
            "highpass_correct_percent": 100 * high_correct,
            "highpass_wrong_percent": 100 * high_wrong,
        })

pd.DataFrame(table5_rows).to_csv(
    final_tables / "table5_signal_family_mixture.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(shift_rows).to_csv(
    final_tables / "figure8_highpass_error_shift.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Table 7 computational profile source
# ------------------------------------------------------------
eff_dir = ROOT / "results/efficiency"
eff_dir.mkdir(parents=True, exist_ok=True)

table7 = pd.DataFrame([
    ["Amazon-Computers", "GraphSAGE", 208.5, 292.9, 1.4, 78.71],
    ["Amazon-Photo", "GraphSAGE", 81.3, 136.4, 1.7, 88.76],
    ["Chameleon", "LINKX", 3.6, 62.1, 17.5, 72.48],
    ["Cornell", "GraphSAGE", 0.8, 4.5, 5.7, 75.41],
    ["Texas", "GraphSAGE", 0.8, 4.1, 5.2, 86.32],
    ["Wisconsin", "GraphSAGE", 0.9, 6.0, 6.8, 84.31],
], columns=[
    "dataset",
    "best_baseline",
    "baseline_time_sec",
    "wgsrc_time_sec",
    "time_ratio",
    "wgsrc_acc_percent"
])

table7.to_csv(eff_dir / "table7_computational_profile_source.csv", index=False, encoding="utf-8-sig")

print("Generated final derived CSVs.")
