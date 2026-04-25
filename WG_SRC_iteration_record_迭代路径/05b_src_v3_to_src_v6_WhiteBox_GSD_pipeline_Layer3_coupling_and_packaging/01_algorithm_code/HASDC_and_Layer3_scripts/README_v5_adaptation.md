# scripts v5 adapted

放置方式：

```text
<project_root>/
  src_v3/
  src_v5/
  experiments_v5/
    scripts/
      *.py
```

也支持把 `scripts/` 直接放到 `<project_root>/scripts/`。脚本会向上搜索 `src_v5` / `srccv5` / `srcv5`。如果你的目录名不同，设置环境变量：

```bash
export WHITEBOX_SRC_V5=/absolute/path/to/src_v5
```

主要适配点：

1. 所有实验脚本优先导入 `src_v5`，不再导入 `src_v3`。
2. Layer 3 调用显式使用 v5 参数：`eta_react`、`activation_mode`、`activation_rho`、`aggregation_mode`、`max_activation_scale`。
3. `exp08_interpretability.py` 从 v5 的 `base_matrices` 取对角线作为可视化用方向权重，不参与预测，避免读取 v3 的 `base_weights` 旧字段。
4. `exp05` 和 `exp10` 中原本依赖 `src_v1` 的旧 Fisher / Layer4 对照改为可选：若没有 `src_v1`，脚本会在 JSON 中标记 `skipped`，不会伪造结果。
5. `whitebox_v5_adapter.py` 会在启动时自动修补当前 `src_v5/layer3_discriminative.py` 中确定性的重复 `elif aggregation_mode == 'linear':` 语法错误。该修补只删除空重复分支，不改变算法公式。

运行示例：

```bash
cd experiments_v5/scripts
python exp02_main_results.py
python run_all_experiments.py --only E02 E04 E08
```
