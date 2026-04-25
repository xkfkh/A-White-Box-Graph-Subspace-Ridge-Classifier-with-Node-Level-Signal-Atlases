SRC-v16c submission package

Contents
--------
1. per_dataset/
   Final result folders for:
   - amazon-computers
   - amazon-photo
   - cornell
   - squirrel
   - texas
   - wisconsin

2. code/src_v16c_paper_experiments/
   Full experiment code used for reproduction.

3. run_one_dataset_reproduce.sh
   A simple helper script to reproduce one dataset.

4. logs/
   Running logs collected during experiments.

5. baselines/
   Baseline CSV files.

Notes
-----
- Texas uses the original formal result group only.
- Amazon-Computers and Squirrel use the fast validation grid.
- Texas / Cornell / Wisconsin / Amazon-Photo use the default validation grid.

Example: reproduce Texas
------------------------
bash run_one_dataset_reproduce.sh \
  ./code/src_v16c_paper_experiments \
  /path/to/planetoid/data \
  texas \
  default \
  10 \
  20260419 \
  ./reproduce_texas_out

Example: reproduce Amazon-Computers
-----------------------------------
bash run_one_dataset_reproduce.sh \
  ./code/src_v16c_paper_experiments \
  /path/to/planetoid/data \
  amazon-computers \
  fast \
  10 \
  20260419 \
  ./reproduce_amazon_computers_out

Example: reproduce Squirrel
---------------------------
bash run_one_dataset_reproduce.sh \
  ./code/src_v16c_paper_experiments \
  /path/to/planetoid/data \
  squirrel \
  fast \
  10 \
  20050224 \
  ./reproduce_squirrel_out
