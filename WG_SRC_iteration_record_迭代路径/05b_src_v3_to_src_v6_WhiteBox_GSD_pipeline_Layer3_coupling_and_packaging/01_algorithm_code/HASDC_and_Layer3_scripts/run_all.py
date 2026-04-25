"""
White-Box Framework -- 统一运行入口 (V3, Layer3 反应版)
====================================

一键运行三个数据集.

用法:
    cd D:\\桌面\\MSR实验复现与创新\\experiments_g1\\claude_whitebox_g1_v2\\src_v3
    python run_all.py               # 运行全部
    python run_all.py cora          # 只运行 Cora
    python run_all.py citeseer      # 只运行 CiteSeer
    python run_all.py pubmed        # 只运行 PubMed
"""

import os, sys, json

# 路径设置: 把 src_v3 自身加入 sys.path，使 layer*.py / data_loader.py 可直接 import
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from framework import create_pipeline


def run_all(datasets=None, verbose=True):
    """运行指定数据集 (默认全部)."""
    if datasets is None:
        datasets = ['cora', 'citeseer', 'pubmed']

    results = {}
    for ds in datasets:
        pipe = create_pipeline(ds)
        results[ds] = pipe.run(verbose=verbose)

    # 汇总
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY -- White-Box Framework V3 (Reactive Layer3)")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'Pipeline':<24} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*52}")
    for ds in datasets:
        r = results[ds]
        print(f"  {ds:<12} {r['pipeline']:<24} {r['val']:>7.4f} {r['test']:>7.4f}")

    print(f"\n  Reference:")
    print(f"    GCN:   Cora ~0.815  CiteSeer ~0.715  PubMed ~0.790")
    print(f"    G1:    Cora ~0.680")
    print(f"    V2:    Cora ~0.812  CiteSeer ~0.711  PubMed ~0.764")
    print(f"{'='*60}")

    # 保存结果
    out_dir = os.path.join(_this_dir, '..', 'results_v3')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'framework_v3_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {os.path.abspath(out_path)}")

    return results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ds_list = [d.lower() for d in sys.argv[1:]]
    else:
        ds_list = None
    run_all(ds_list)
