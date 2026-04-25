"""
White-Box Framework -- 统一运行入口
====================================

一键运行三个数据集, 展示主体框架 + 插件的统一输出.

用法:
    python run_all.py               # 运行全部
    python run_all.py cora          # 只运行 Cora
    python run_all.py citeseer      # 只运行 CiteSeer
"""

import os, sys, json, time

# 路径设置: 确保 src_v1 可 import
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from framework import create_pipeline, DEFAULT_CONFIGS


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
    print("  FINAL SUMMARY -- White-Box Framework")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'Pipeline':<20} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*48}")
    for ds in datasets:
        r = results[ds]
        print(f"  {ds:<12} {r['pipeline']:<20} {r['val']:>7.4f} {r['test']:>7.4f}")

    print(f"\n  Reference:")
    print(f"    GCN:   Cora ~0.815  CiteSeer ~0.715  PubMed ~0.790")
    print(f"    G1:    Cora ~0.680")
    print(f"{'='*60}")

    # 保存结果
    out_dir = os.path.join(_src_dir, '..', 'results_v1')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'framework_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")

    return results


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ds_list = [d.lower() for d in sys.argv[1:]]
    else:
        ds_list = None
    run_all(ds_list)

