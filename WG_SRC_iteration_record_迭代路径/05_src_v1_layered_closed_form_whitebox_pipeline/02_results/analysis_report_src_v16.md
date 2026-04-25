# src_v16 Analysis Report: reliability_antihub
Dataset: chameleon
Splits: 10

## Overall Results
- Reliability+AntiHub val_mean  = 0.7207 +/- 0.0183
- Reliability+AntiHub test_mean = 0.7055 +/- 0.0183
- Fixed-weight baseline val_mean  = 0.7112 +/- 0.0183
- Fixed-weight baseline test_mean = 0.7103 +/- 0.0155
- Improvement: -0.0048

## Per-Split Results
  split= 0  val=0.7229  test=0.6930  base_test=0.7018  delta=-0.0088
  split= 1  val=0.7023  test=0.7368  base_test=0.7434  delta=-0.0066
  split= 2  val=0.7449  test=0.6996  base_test=0.7061  delta=-0.0066
  split= 3  val=0.7435  test=0.7149  base_test=0.7039  delta=+0.0110
  split= 4  val=0.7311  test=0.7061  base_test=0.7149  delta=-0.0088
  split= 5  val=0.7188  test=0.7149  base_test=0.7171  delta=-0.0022
  split= 6  val=0.7051  test=0.6645  base_test=0.6776  delta=-0.0132
  split= 7  val=0.6831  test=0.6974  base_test=0.7105  delta=-0.0132
  split= 8  val=0.7325  test=0.7215  base_test=0.7105  delta=+0.0110
  split= 9  val=0.7229  test=0.7061  base_test=0.7171  delta=-0.0110

## Hyperparameter Choices
  split= 0  top_k=5000  dim=64  ridge_alpha=0.1  w_base=0.5  alpha_rel=0.5  beta_hub=0.0
  split= 1  top_k=6000  dim=64  ridge_alpha=0.1  w_base=0.3  alpha_rel=0.5  beta_hub=0.2
  split= 2  top_k=5000  dim=64  ridge_alpha=0.1  w_base=0.3  alpha_rel=1.0  beta_hub=0.0
  split= 3  top_k=5000  dim=64  ridge_alpha=0.1  w_base=0.5  alpha_rel=0.5  beta_hub=0.0
  split= 4  top_k=5000  dim=64  ridge_alpha=0.1  w_base=0.7  alpha_rel=0.5  beta_hub=0.0
  split= 5  top_k=6000  dim=64  ridge_alpha=0.5  w_base=0.7  alpha_rel=0.5  beta_hub=0.0
  split= 6  top_k=5000  dim=24  ridge_alpha=0.1  w_base=0.5  alpha_rel=1.5  beta_hub=0.0
  split= 7  top_k=5000  dim=24  ridge_alpha=0.1  w_base=0.5  alpha_rel=0.5  beta_hub=0.2
  split= 8  top_k=5000  dim=64  ridge_alpha=0.1  w_base=0.6  alpha_rel=0.5  beta_hub=0.0
  split= 9  top_k=3000  dim=48  ridge_alpha=0.1  w_base=0.7  alpha_rel=0.5  beta_hub=0.0

## Key Observations
- Dual-classifier reliability map provides condition-dependent fusion weights
- Anti-hub penalty targets hub classes in low-confidence, low-exposure conditions
- All mechanisms are closed-form and fully interpretable

