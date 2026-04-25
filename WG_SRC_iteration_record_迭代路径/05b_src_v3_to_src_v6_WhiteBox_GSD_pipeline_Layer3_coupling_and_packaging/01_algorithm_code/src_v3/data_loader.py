"""
数据加载模块：支持 Cora / CiteSeer / PubMed 三个 Planetoid 数据集

主接口：
    load_dataset(dataset, data_dir=DATA_DIR)
    -> features, labels, adj_norm, lap, train_idx, val_idx, test_idx, num_classes

兼容旧接口（保留）：
    load_planetoid(dataset_name, data_dir)
    -> features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx

返回值说明：
    features   : [N, D]  float32 numpy，行 L1 归一化后的节点特征
    labels     : [N]     int64  numpy，节点类别标签
    adj_norm   : [N, N]  float32 numpy，D^{-1/2}(A+I)D^{-1/2}（含自环）
    lap        : [N, N]  float32 numpy，I - adj_norm（归一化拉普拉斯）
    train_idx  : np.ndarray int，训练节点索引（每类 20 个，共 20*C 个）
    val_idx    : np.ndarray int，验证节点索引（500 个）
    test_idx   : np.ndarray int，测试节点索引（1000 个）
    num_classes: int

CiteSeer 特殊处理：
    test.index 中存在孤立节点，索引范围（最大 3326）超出 allx+tx 行数（3312），
    超出部分用零特征填充，标签设为 -1（无效标签）。
    test_idx 只保留 labels >= 0 的节点，不参与评估。

PubMed 特殊处理：
    allx/tx 为 scipy sparse（TF-IDF 浮点），调用 todense() 转换。
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DATA_DIR = "D:/桌面/MSR实验复现与创新/planetoid/data"


# ─────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────

def _load_raw(data_dir, dataset):
    """读取 Planetoid 原始 pickle 文件，返回 7 个对象 + test_idx_raw。"""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs = []
    for n in names:
        path = os.path.join(data_dir, f'ind.{dataset}.{n}')
        with open(path, 'rb') as f:
            objs.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_path = os.path.join(data_dir, f'ind.{dataset}.test.index')
    test_idx_raw = [int(line.strip()) for line in open(test_idx_path)]
    return x, y, tx, ty, allx, ally, graph, test_idx_raw


def _to_dense_f32(mat):
    """将 scipy sparse 或 numpy matrix 转为 float32 ndarray。"""
    if sp.issparse(mat):
        return np.array(mat.todense(), dtype=np.float32)
    return np.array(mat, dtype=np.float32)


def _row_normalize(features):
    """
    行 L1 归一化：每行除以其绝对值之和。
    零行保持为零（避免除零）。
    """
    rowsum = np.abs(features).sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return features / rowsum


def _build_adj_sparse(graph, n):
    """
    从邻接表 dict 构造对称稀疏邻接矩阵（不含自环）。
    graph: {node_id: [neighbor_id, ...]}
    """
    rows, cols = [], []
    for src, neighbors in graph.items():
        for dst in neighbors:
            rows.append(src)
            cols.append(dst)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A


def _normalize_adj_sparse(A, n):
    """
    计算 adj_norm = D^{-1/2}(A+I)D^{-1/2} 和 lap = I - adj_norm。

    数学说明
    --------
    设 A~ = A + I（加自环），D~ = diag(A~ 1)（度矩阵）
    adj_norm_{ij} = A~_{ij} / sqrt(D~_{ii} * D~_{jj})
    lap = I - adj_norm

    几何意义：
      adj_norm 是对称归一化图传播算子，特征值在 [0,2]；
      lap 的特征值在 [0,2]，0 对应连通分量常数向量，
      2 对应二部图振荡模式。

    小数值例子（3 节点链 0-1-2）：
      A~ = [[1,1,0],[1,2,1],[0,1,1]]，D~ = diag(2,4,2)
      adj_norm[0,1] = 1/sqrt(2*4) = 0.354
      adj_norm[1,1] = 2/4 = 0.5
    """
    A_tilde = A + sp.eye(n, format='csr', dtype=np.float32)
    deg = np.array(A_tilde.sum(axis=1), dtype=np.float32).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
    adj_norm = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
    lap = sp.eye(n, format='csr', dtype=np.float32) - adj_norm
    return adj_norm.toarray().astype(np.float32), lap.toarray().astype(np.float32)


def _build_adj_dense(graph, n):
    """从邻接表构建稠密对称邻接矩阵（含自环），供旧接口使用。"""
    adj = np.zeros((n, n), dtype=np.float32)
    for node, neighbors in graph.items():
        for nb in neighbors:
            adj[node, nb] = 1.0
            adj[nb, node] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def _normalize_adj_dense(adj):
    """旧接口用：adj_norm = D^{-1/2} A D^{-1/2}，lap = I - adj_norm。"""
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    n = adj.shape[0]
    lap = np.eye(n, dtype=np.float32) - adj_norm
    return adj_norm.astype(np.float32), lap.astype(np.float32)


def _make_standard_splits(labels, num_classes):
    """
    标准 Planetoid 划分：
      train : 每类前 20 个有效节点，共 20 * num_classes 个
      val   : train 之后连续 500 个节点（索引 train_size ~ train_size+500）
    test_idx 由调用方传入（来自 test.index 文件）。
    """
    train_idx = []
    count = {c: 0 for c in range(num_classes)}
    for i, lbl in enumerate(labels):
        c = int(lbl)
        if c in count and count[c] < 20:
            train_idx.append(i)
            count[c] += 1
        if sum(count.values()) == 20 * num_classes:
            break
    train_size = len(train_idx)
    val_idx = list(range(train_size, train_size + 500))
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


# ─────────────────────────────────────────────────────────────
# 新主接口：load_dataset
# ─────────────────────────────────────────────────────────────

def load_dataset(dataset: str, data_dir: str = DATA_DIR):
    """
    加载 Planetoid 数据集，返回标准接口。

    参数
    ----
    dataset  : 'cora' | 'citeseer' | 'pubmed'（大小写不敏感）
    data_dir : 数据目录，默认 DATA_DIR

    返回
    ----
    features   : np.ndarray [N, D] float32，行 L1 归一化
    labels     : np.ndarray [N]    int64
    adj_norm   : np.ndarray [N, N] float32，D^{-1/2}(A+I)D^{-1/2}
    lap        : np.ndarray [N, N] float32，I - adj_norm
    train_idx  : np.ndarray int64，20*num_classes 个训练节点
    val_idx    : np.ndarray int64，500 个验证节点
    test_idx   : np.ndarray int64，1000 个测试节点
    num_classes: int
    """
    dataset = dataset.lower()
    assert dataset in ('cora', 'citeseer', 'pubmed'), \
        f"不支持的数据集: {dataset}，请选择 cora / citeseer / pubmed"

    x, y, tx, ty, allx, ally, graph, test_idx_raw = _load_raw(data_dir, dataset)
    test_idx_sorted = sorted(test_idx_raw)

    allx_d = _to_dense_f32(allx)   # [n_allx, D]
    tx_d   = _to_dense_f32(tx)     # [n_tx,   D]
    ally_d = _to_dense_f32(ally)   # [n_allx, C]
    ty_d   = _to_dense_f32(ty)     # [n_tx,   C]

    if dataset == 'citeseer':
        # CiteSeer 特殊处理（修复 v3）
        # ================================
        # 问题：test_idx 最大值(3326) > allx+tx 行数(3312)
        #   超出的 ~15 个孤立节点在 tx 中没有对应行。
        #
        # 标准 Planetoid 处理方式（与 PyG/DGL 一致）：
        #   1. N = max(test_idx_sorted) + 1 = 3327
        #   2. 初始化 N×D 零特征矩阵 + N 个 -1 标签
        #   3. 前 n_allx 行填入 allx/ally
        #   4. 用 test_idx_raw[i] → tx[i] / ty[i] 做正确映射
        #      （raw 保持 tx 存储顺序，不能用 sorted！）
        #   5. 超出范围的孤立节点保持零特征/-1 标签
        N = max(test_idx_sorted) + 1
        D = allx_d.shape[1]
        C = ally_d.shape[1]
        n_allx = allx_d.shape[0]
        n_tx   = tx_d.shape[0]

        features_full = np.zeros((N, D), dtype=np.float32)
        labels_int = np.full(N, -1, dtype=np.int64)

        # allx 部分
        features_full[:n_allx] = allx_d
        labels_int[:n_allx] = np.argmax(ally_d, axis=1)

        # tx/ty 部分：使用 test_idx_raw（保持 tx 行顺序）
        # test_idx_raw[i] 是 tx[i] 对应的图节点编号
        for i in range(n_tx):
            idx = test_idx_raw[i]
            if idx < N:
                features_full[idx] = tx_d[i]
                labels_int[idx] = np.argmax(ty_d[i])

        # 行 L1 归一化
        features_full = _row_normalize(features_full)

        num_classes = C

        # 图结构
        A = _build_adj_sparse(graph, N)
        adj_norm, lap = _normalize_adj_sparse(A, N)

        # 标准 Planetoid 固定划分
        n_train = x.shape[0]  # CiteSeer: 120
        train_idx = np.arange(n_train, dtype=np.int64)
        val_idx = np.arange(n_train, n_train + 500, dtype=np.int64)

        # test_idx 只保留有真实标签的节点（排除 label == -1 的孤立节点）
        test_idx = np.array([idx for idx in test_idx_sorted
                             if labels_int[idx] >= 0], dtype=np.int64)

        return features_full, labels_int, adj_norm, lap, train_idx, val_idx, test_idx, num_classes

    else:
        # Cora / PubMed：标准拼接 + 重排
        features_full = np.vstack([allx_d, tx_d])          # [n_allx+n_tx, D]
        labels_full   = np.vstack([ally_d, ty_d])           # [n_allx+n_tx, C]
        # 按 test_idx_raw 原始顺序重排（消除测试集乱序）
        features_full[test_idx_raw] = features_full[test_idx_sorted]
        labels_full[test_idx_raw]   = labels_full[test_idx_sorted]

    # 行 L1 归一化
    features_full = _row_normalize(features_full)

    # one-hot → int 标签
    labels = np.argmax(labels_full, axis=1).astype(np.int64)
    num_classes = labels_full.shape[1]

    # 图结构
    N = features_full.shape[0]
    A = _build_adj_sparse(graph, N)
    adj_norm, lap = _normalize_adj_sparse(A, N)

    # 标准 Planetoid 固定划分：
    #   train = range(x.shape[0])       Cora=140, PubMed=60
    #   val   = range(x.shape[0], x.shape[0] + 500)
    n_train = x.shape[0]
    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + 500, dtype=np.int64)
    test_idx = np.array(test_idx_sorted[:1000], dtype=np.int64)

    return features_full, labels, adj_norm, lap, train_idx, val_idx, test_idx, num_classes


# ─────────────────────────────────────────────────────────────
# Torch 转换
# ─────────────────────────────────────────────────────────────

def to_torch(features, labels, adj_norm, lap, train_idx, val_idx, test_idx):
    """
    将 load_dataset 返回的 numpy 数组转为 torch tensor。

    返回
    ----
    features_t  : torch.FloatTensor  [N, D]
    labels_t    : torch.LongTensor   [N]
    adj_norm_t  : torch.FloatTensor  [N, N]
    lap_t       : torch.FloatTensor  [N, N]
    train_idx_t : torch.LongTensor
    val_idx_t   : torch.LongTensor
    test_idx_t  : torch.LongTensor
    """
    import torch
    return (
        torch.tensor(features,  dtype=torch.float32),
        torch.tensor(labels,    dtype=torch.long),
        torch.tensor(adj_norm,  dtype=torch.float32),
        torch.tensor(lap,       dtype=torch.float32),
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx,   dtype=torch.long),
        torch.tensor(test_idx,  dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────
# 掩码工具
# ─────────────────────────────────────────────────────────────

def make_masks(n, train_idx, val_idx, test_idx):
    """
    生成布尔掩码 np.ndarray，shape (N,)。

    参数
    ----
    n         : 节点总数
    train_idx : 训练节点索引
    val_idx   : 验证节点索引
    test_idx  : 测试节点索引

    返回
    ----
    train_mask, val_mask, test_mask : np.ndarray bool [N]
    """
    train_mask = np.zeros(n, dtype=bool)
    val_mask   = np.zeros(n, dtype=bool)
    test_mask  = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True
    return train_mask, val_mask, test_mask


# ─────────────────────────────────────────────────────────────
# 旧接口兼容（保留，供已有代码调用）
# ─────────────────────────────────────────────────────────────

def load_planetoid(dataset_name, data_dir):
    """
    旧接口：返回 8 个值（含稠密 adj）。
    新代码请使用 load_dataset()。
    """
    name = dataset_name.lower()
    assert name in ('cora', 'citeseer', 'pubmed')

    x, y, tx, ty, allx, ally, graph, test_idx_raw = _load_raw(data_dir, name)
    test_idx_sorted = sorted(test_idx_raw)

    n_train = x.shape[0]  # Cora=140, CiteSeer=120, PubMed=60

    if name == 'citeseer':
        N = max(test_idx_sorted) + 1
        D = allx.shape[1]
        features = np.zeros((N, D), dtype=np.float32)
        allx_d = _to_dense_f32(allx)
        tx_d   = _to_dense_f32(tx)
        features[:allx_d.shape[0]] = allx_d
        # 修复 v3: 使用 test_idx_raw（保持 tx 行顺序）
        n_tx = tx_d.shape[0]
        for i in range(n_tx):
            idx = test_idx_raw[i]
            if idx < N:
                features[idx] = tx_d[i]
        # 标签初始化为 -1（无效），不是全零
        labels = np.full(N, -1, dtype=np.int64)
        ally_d = _to_dense_f32(ally)
        ty_d   = _to_dense_f32(ty)
        labels[:ally_d.shape[0]] = np.argmax(ally_d, axis=1)
        for i in range(n_tx):
            idx = test_idx_raw[i]
            if idx < N:
                labels[idx] = np.argmax(ty_d[i])
        train_idx = list(range(n_train))
        val_idx   = list(range(n_train, n_train + 500))
        # test_idx 只保留有真实标签的节点
        test_idx = [idx for idx in test_idx_sorted if labels[idx] >= 0]
    else:
        allx_d = _to_dense_f32(allx)
        tx_d   = _to_dense_f32(tx)
        ally_d = _to_dense_f32(ally)
        ty_d   = _to_dense_f32(ty)
        features = np.vstack([allx_d, tx_d])
        labels_mat = np.vstack([ally_d, ty_d])
        features[test_idx_raw]   = features[test_idx_sorted]
        labels_mat[test_idx_raw] = labels_mat[test_idx_sorted]
        labels = np.argmax(labels_mat, axis=1).astype(np.int64)
        N = features.shape[0]
        train_idx = list(range(n_train))
        val_idx   = list(range(n_train, n_train + 500))
        test_idx = test_idx_sorted[:1000]

    adj = _build_adj_dense(graph, N)
    adj_norm, lap = _normalize_adj_dense(adj)
    return features, labels, adj, adj_norm, lap, train_idx, val_idx, test_idx
