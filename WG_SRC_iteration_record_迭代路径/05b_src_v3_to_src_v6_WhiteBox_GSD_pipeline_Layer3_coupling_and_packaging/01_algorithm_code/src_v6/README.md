# HA-SDC：同异配自适应带符号双子空间交联算法

HA-SDC 是一个白盒、无需神经网络训练的图节点分类器。它同时保留平滑通道和原始通道，对每个类别建立两个子空间，并用验证集估计类别级门控与带符号子空间交联。

## 1. 安装

```bash
pip install -e .
```

依赖：`numpy`、`scipy`。

## 2. 最小使用示例

```python
import numpy as np
from scipy import sparse
from ha_sdc import HASDC

# A: n x n 邻接矩阵；X: n x D 特征矩阵；y: 长度 n 的标签数组
# train_idx: 训练节点下标；val_idx: 验证节点下标

model = HASDC(
    lambda_smooth=1.0,
    d_s=8,
    d_r=8,
    tau_gate=5.0,
    gamma=0.1,
    laplacian="normalized",
)

model.fit(A, X, y, train_idx, val_idx)
pred = model.predict(A, X)
residual = model.decision_function(A, X)  # 越小越像该类别
state = model.get_state()

print("classes:", state.classes)
print("gate:", state.gate)
print("alpha matrix:\n", state.alpha)
print("node explanation:", model.explain_node(A, X, node_id=0))
```

如果你的训练标签和验证标签是分开的，也可以直接传字典：

```python
train_labels = {0: "A", 3: "B", 8: "A"}
val_labels = {1: "A", 4: "B"}

model.fit_from_labels(A, X, train_labels, val_labels)
pred = model.predict(A, X)
```

## 3. 数学定义

设图有 `n` 个节点，每个节点有 `D` 维原始特征。原始特征矩阵为：

\[
X \in \mathbb{R}^{n\times D}
\]

邻接矩阵为：

\[
A \in \mathbb{R}^{n\times n}
\]

图拉普拉斯矩阵为：

\[
L
\]

### Layer 1：双通道表示构造

平滑通道：

\[
Z = (I + \lambda L)^{-1}X
\]

其中：

- `I` 是 `n x n` 单位矩阵；
- `lambda_smooth = λ` 是图平滑强度；
- `Z` 是平滑后的节点特征矩阵。

原始通道直接使用：

\[
X
\]

### Layer 2：类别双子空间

对类别 `c`，在平滑通道中取训练样本集合：

\[
\{z_i: y_i=c, i\in \mathcal{T}\}
\]

类别平滑中心：

\[
\mu_{c}^{s}=\frac{1}{N_c}\sum_{i:y_i=c}z_i
\]

中心化后做 SVD，取前 `d_s` 个方向形成平滑子空间基：

\[
B_{c}^{s}\in\mathbb{R}^{D\times d_s}
\]

对任意节点 `i`，其平滑子空间坐标为：

\[
u_{c}^{s}(i)=(z_i-\mu_c^s)^T B_c^s
\]

投影证据为：

\[
E_{c}^{s}(i)=\|u_c^s(i)\|_2^2
\]

平滑残差为：

\[
r_{c}^{s}(i)=\|z_i-\mu_c^s\|_2^2-E_c^s(i)
\]

原始通道同理得到：

\[
\mu_c^r,
\quad B_c^r,
\quad u_c^r(i),
\quad r_c^r(i)
\]

### Layer 3：同异配门控

在验证集上分别用平滑残差和原始残差分类，得到类别 `c` 的验证准确率：

\[
Acc_c^s,
\quad Acc_c^r
\]

平滑优势：

\[
h_c = Acc_c^s - Acc_c^r
\]

门控系数：

\[
g_c = \sigma(\tau h_c)=\frac{1}{1+\exp(-\tau h_c)}
\]

其中 `τ = tau_gate`。基础双通道残差：

\[
r_c^{base}(i)=g_c r_c^s(i)+(1-g_c)r_c^r(i)
\]

### Layer 3：带符号子空间交联

类别 `c` 的平滑子空间和类别 `c'` 的原始子空间之间的交联矩阵为：

\[
M_{c,c'} = (B_c^s)^T B_{c'}^r
\]

平均重叠强度为：

\[
o_{c,c'}=\frac{\|M_{c,c'}\|_F^2}{\min(d_s,d_r)}
\]

节点 `i` 对类别对 `(c,c')` 的交联激活为：

\[
q_{c,c'}(i)=\frac{\|M_{c,c'}u_{c'}^r(i)\|_2^2}{d_s}
\]

带符号交联系数：

\[
\alpha_{c,c'}\in\mathbb{R}
\]

本实现中，`alpha` 的符号由验证集白盒估计：

- 若对类别 `c`，加入正向调制后验证准确率高于负向调制，则 `alpha > 0`；
- 若负向调制更好，则 `alpha < 0`；
- 若二者差距不足 `alpha_min_acc_delta`，则 `alpha = 0`。

`alpha` 的绝对值使用平均重叠强度 `o_{c,c'}`。

交联调制项：

\[
\Delta_c(i)=\sum_{c'\neq c}\alpha_{c,c'}q_{c,c'}(i)
\]

最终残差：

\[
r_c^{final}(i)=r_c^{base}(i)-\gamma\Delta_c(i)
\]

预测标签：

\[
\hat y_i=\arg\min_c r_c^{final}(i)
\]

## 4. 参数建议

| 参数 | 作用 | 建议 |
|---|---|---|
| `lambda_smooth` | 图平滑强度 | 同配图可大一些，如 1 到 10；异配图可小一些，如 0.1 到 1 |
| `d_s` | 平滑子空间维度 | 通常取 4、8、16 |
| `d_r` | 原始子空间维度 | 通常取 4、8、16 |
| `tau_gate` | 门控 sigmoid 陡峭度 | 3 到 10 |
| `gamma` | 交联调制强度 | 从 0.01、0.05、0.1、0.5 网格搜索 |
| `alpha_min_acc_delta` | alpha 非零所需准确率差异 | 验证集小时可设 0.05，验证集大时可设 0 |

## 5. 输出解释

`model.get_state()` 返回：

- `classes`：类别列表；
- `smooth_subspaces`：每个类别的平滑子空间；
- `raw_subspaces`：每个类别的原始子空间；
- `gate`：每个类别的平滑门控 `g_c`；
- `smooth_accuracy_by_class`：验证集上平滑通道分类准确率；
- `raw_accuracy_by_class`：验证集上原始通道分类准确率；
- `overlap`：类别平滑子空间与其他类别原始子空间的重叠强度矩阵；
- `alpha`：带符号交联系数矩阵。

`model.explain_node(A, X, node_id)` 返回该节点对每个类别的：

- 平滑残差；
- 原始残差；
- 门控；
- 基础残差；
- 交联调制项；
- 最终残差。

## 6. 注意事项

1. 这是一个转导式图分类器：平滑步骤使用整张图的 `A` 和 `X`。
2. 如果新节点加入图，需要重新给出新的 `A` 和 `X` 并重新计算平滑特征。
3. 特征尺度会影响残差大小，建议先对 `X` 做标准化。
4. 验证集过小会导致 `g_c` 和 `alpha` 不稳定，可以调小 `tau_gate`，或设置 `alpha_min_acc_delta=0.05`。
