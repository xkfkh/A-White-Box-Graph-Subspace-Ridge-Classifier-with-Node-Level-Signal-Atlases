把本文件夹放到 scripts/v6_layer3_fast，然后运行：

cd /d "D:\桌面\MSR实验复现与创新\experiments_g1\claude_whitebox_g1_v2\scripts\v6_layer3_fast"
python run_v6_layer3_fixed_base_fast_launcher.py

这个版本固定非 Layer3 参数，每个 dataset/repeat 只 fit 一次，然后只扫 alpha_mode 和 gamma。
默认数据集：actor,chameleon,wisconsin；默认 repeats=0..4。
输出：scripts/results_v6_layer3_fixed_base_fast/
