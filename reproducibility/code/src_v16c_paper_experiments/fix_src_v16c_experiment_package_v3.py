# -*- coding: utf-8 -*-
"""
Hotfix v3 for src_v16c_paper_experiments.
Run this file inside src_v16c_paper_experiments directory:
    python fix_src_v16c_experiment_package_v3.py
It fixes duplicate keyword fields caused by **asdict(cfg) plus explicit repeated keys.
"""
from pathlib import Path

root = Path.cwd()
if not (root / 'scripts').exists():
    raise SystemExit('Please run this script inside src_v16c_paper_experiments, where scripts/ exists.')

patches = {
    'scripts/run_E05_ablation.py': [
        ("ablation=ab, feature_variant=variant, method='src_v16c_ablation',",
         "ablation=ab, method='src_v16c_ablation',"),
    ],
}

# Keep previous v2 alpha fix too, in case the local package is not fully patched.
for py in (root / 'scripts').glob('run_E*.py'):
    s = py.read_text(encoding='utf-8')
    s2 = s.replace(
        'alphas=json.dumps(list(cfg.alphas))',
        'alphas_json=json.dumps(list(cfg.alphas))'
    )
    if s2 != s:
        py.write_text(s2, encoding='utf-8')
        print('patched alpha duplicate:', py)

for rel, repls in patches.items():
    p = root / rel
    if not p.exists():
        print('skip missing:', rel)
        continue
    s = p.read_text(encoding='utf-8')
    original = s
    for old, new in repls:
        s = s.replace(old, new)
    if s != original:
        p.write_text(s, encoding='utf-8')
        print('patched:', rel)
    else:
        print('already ok:', rel)

print('Hotfix v3 done.')
