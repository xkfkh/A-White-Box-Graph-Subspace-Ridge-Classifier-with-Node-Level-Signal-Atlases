#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch src_v16c_paper_experiments package for duplicate cfg-field recording bugs.
Run from inside src_v16c_paper_experiments/.
"""
from pathlib import Path

root = Path.cwd()
if not (root / 'paperexp' / 'core.py').exists() or not (root / 'scripts').exists():
    raise SystemExit('请在 src_v16c_paper_experiments 目录下运行本脚本')

replacements = []
for p in (root / 'scripts').glob('run_E*.py'):
    s = p.read_text(encoding='utf-8')
    old = s
    s = s.replace("**asdict(cfg), alphas=json.dumps(list(cfg.alphas)),", "**asdict(cfg), alphas_json=json.dumps(list(cfg.alphas)),")
    s = s.replace("ablation=ab, feature_variant=variant, method='src_v16c_ablation',", "ablation=ab, method='src_v16c_ablation',")
    if s != old:
        p.write_text(s, encoding='utf-8')
        replacements.append(str(p.relative_to(root)))

core = root / 'paperexp' / 'core.py'
s = core.read_text(encoding='utf-8')
old = s
s = s.replace('"alphas": json.dumps(list(cfg.alphas))', '"alphas_json": json.dumps(list(cfg.alphas))')
if s != old:
    core.write_text(s, encoding='utf-8')
    replacements.append(str(core.relative_to(root)))

print('patched files:')
for x in replacements:
    print(' -', x)
if not replacements:
    print(' - none; package already patched')


