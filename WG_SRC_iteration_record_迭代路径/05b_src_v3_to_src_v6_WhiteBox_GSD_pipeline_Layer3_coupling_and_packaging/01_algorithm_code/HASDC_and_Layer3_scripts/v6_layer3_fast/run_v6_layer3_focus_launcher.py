import os
import sys
import subprocess
from pathlib import Path

V6_L3_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = V6_L3_DIR.parent

SRC_V6 = Path(r"D:\桌面\MSR实验复现与创新\experiments_g1\claude_whitebox_g1_v2\src_v6")
DATA = Path(r"D:\桌面\MSR实验复现与创新\planetoid\data")
OUT = SCRIPTS_DIR / "results_v6_layer3_fast_gamma"

DATASETS = "chameleon,wisconsin"
REPEATS = "0,1,2"

env = os.environ.copy()
env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")

cmd = [
    sys.executable,
    str(V6_L3_DIR / "run_v6_layer3_focus.py"),

    "--src_v6", str(SRC_V6),
    "--data_root", str(DATA),
    "--out_dir", str(OUT),
    "--device", "auto",

    "--datasets", DATASETS,
    "--repeats", REPEATS,

    # 固定 base，只看 Layer3 强度
    "--lambda_grid", "0.1",
    "--d_pair_grid", "8:8",
    "--tau_grid", "5",

    # 只测 Layer3，不扫其他东西
    "--alpha_modes", "zero,force_pos,force_neg",
    "--gamma_grid", "0,100,1000,10000",
]

print("开始 v6 Layer3 快速强度测试")
print("V6_L3_DIR =", V6_L3_DIR)
print("SRC_V6 =", SRC_V6)
print("DATA =", DATA)
print("OUT =", OUT)
print("DATASETS =", DATASETS)
print("REPEATS =", REPEATS)

subprocess.run(cmd, check=True, env=env)

print("完成。结果在：", OUT)