import os
import sys
import subprocess
from pathlib import Path

V8_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = V8_SCRIPT_DIR.parent.parent  # .../wgsrc_development_workspace
SRC_V8 = PROJECT_DIR / "src_v8"
DATA = Path(r"D:\桌面\MSR实验复现与创新\planetoid\data")
OUT = PROJECT_DIR / "scripts" / "results_v8_ch_smoke"


def has_required_files(p: Path) -> bool:
    return (p / 'out1_node_feature_label.txt').exists() and (p / 'out1_graph_edges.txt').exists()


def find_chameleon_root(data_root: Path) -> Path:
    candidates = [
        data_root / 'chameleon' / 'raw',
        data_root / 'chameleon',
    ]
    for c in candidates:
        if has_required_files(c):
            return c
    for p in data_root.rglob('out1_node_feature_label.txt'):
        parent = p.parent
        if 'chameleon' in str(parent).lower() and has_required_files(parent):
            return parent
    raise FileNotFoundError(f'在 {data_root} 下找不到 chameleon 原始目录')


def find_algo_py(src_v8: Path) -> Path:
    preferred = [
        src_v8 / 'whcp_v4_rnr_lowrank_chameleon.py',
        src_v8 / 'whcp_v4_rnr_lowrank_chameleon(1).py',
        src_v8 / 'main.py',
    ]
    for p in preferred:
        if p.exists():
            return p
    pys = [p for p in src_v8.glob('*.py') if not p.name.startswith('__')]
    if len(pys) == 1:
        return pys[0]
    if not pys:
        raise FileNotFoundError(f'{src_v8} 下没找到可运行的 .py 算法文件')
    raise FileNotFoundError(f'{src_v8} 下找到多个 .py 文件，请手动在脚本里指定算法文件: {[p.name for p in pys]}')


RAW_ROOT = find_chameleon_root(DATA)
ALGO = find_algo_py(SRC_V8)

print('V8_SCRIPT_DIR =', V8_SCRIPT_DIR)
print('PROJECT_DIR   =', PROJECT_DIR)
print('SRC_V8        =', SRC_V8)
print('ALGO          =', ALGO)
print('RAW_ROOT      =', RAW_ROOT)
print('OUT           =', OUT)

cmd = [
    sys.executable,
    str(ALGO),
    '--root', str(RAW_ROOT),
    '--out', str(OUT),
    '--repeats', '1',
    '--preprocess_mode', 'raw',
    '--feature_mode', 'rnr_plus_raw',
    '--rnr_variant', 'full',
    '--relations', 'sym',
    '--dim_list', '8',
    '--lambda_smooth_list', '0.05,0.1',
    '--ridge_alpha_list', '1',
    '--ridge_temperature_list', '1',
    '--residual_temperature_list', '1',
    '--alpha_sub_list', '1.0',
    '--beta_ridge_list', '0.5,1.0',
    '--gamma_geom_list', '0,0.25',
    '--lambda_list', '0,0.1',
    '--iter_list', '1',
    '--local_rank_list', '3',
    '--max_neighbors', '24',
    '--geometry_use', 'raw',
    '--tau_gate', '5',
    '--residual_norm', 'none',
    '--score_norm_mode', 'median',
    '--reliability_mode', 'degree_entropy',
    '--kappa', '5',
    '--fccp_temperature', '1.0',
]

subprocess.run(cmd, check=True)
print('v8 chameleon 冒烟测试完成。')


