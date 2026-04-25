import os
import sys
import subprocess
from pathlib import Path

V6_L3_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = V6_L3_DIR.parent

SRC_V6 = Path(r"D:\桌面\MSR实验复现与创新\experiments_g1\claude_whitebox_g1_v2\src_v6")
DATA = Path(r"D:\桌面\MSR实验复现与创新\planetoid\data")
OUT = SCRIPTS_DIR / "results_v6_layer3_fixed_base_fast"

# 这些是之前 smooth+Layer2 不好的异配图/网页图。
# 先跑三个最关键的；想全跑就改成：actor,chameleon,squirrel,cornell,texas,wisconsin
DATASETS = "actor,chameleon,wisconsin"
REPEATS = "0,1,2,3,4"

env = os.environ.copy()
env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")

cmd = [
    sys.executable,
    str(V6_L3_DIR / "run_v6_layer3_fixed_base_fast.py"),
    "--src_v6", str(SRC_V6),
    "--data_root", str(DATA),
    "--out_dir", str(OUT),
    "--datasets", DATASETS,
    "--repeats", REPEATS,

    # 只扫 Layer3：alpha_mode + gamma。
    # gamma 拉大是因为你当前日志里 cross 只有 1e-4 到 1e-3 量级，gamma<=1 基本改不了决策。
    "--alpha_modes", "zero,learned,force_pos,force_neg",
    "--gamma_grid", "0,1,10,100,1000",
]

print("开始运行 v6 固定 base 的 Layer3 快速调参")
print("V6_L3_DIR =", V6_L3_DIR)
print("SRC_V6 =", SRC_V6)
print("DATA =", DATA)
print("OUT =", OUT)
print("DATASETS =", DATASETS)
print("REPEATS =", REPEATS)
subprocess.run(cmd, check=True, env=env)
print("完成。结果在：", OUT)
