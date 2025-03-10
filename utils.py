import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import scipy.io as sio
import random
import dgl
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, roc_curve


###############################################
# Forked from GRAND-Lab/CoLA                  #
###############################################

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret


def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks


def micro_f1(logits, labels):
    preds = torch.round(nn.Sigmoid()(logits))
    preds = preds.long()
    labels = labels.long()

    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_to_dict(adj, hop=1, min_len=8):
    adj = np.array(adj.todense(), dtype=np.float64)
    num_node = adj.shape[0]
    # adj += np.eye(num_node)

    adj_diff = adj
    if hop > 1:
        for _ in range(hop - 1):
            adj_diff = adj_diff.dot(adj)

    dict = {}
    for i in range(num_node):
        dict[i] = []
        for j in range(num_node):
            if adj_diff[i, j] > 0:
                dict[i].append(j)

    final_dict = dict.copy()

    for i in range(num_node):
        while len(final_dict[i]) < min_len:
            final_dict[i].append(random.choice(dict[random.choice(dict[i])]))
    return dict


def loss_func_rec(X, X_rec, motifs_feat, S_hat, beta):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X - X_rec, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    # attribute_cost = torch.mean(attribute_reconstruction_errors)

    # high-order structure reconstruction loss
    diff_structure = torch.pow(motifs_feat - S_hat, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    # structure_cost = torch.mean(structure_reconstruction_errors)

    cost = beta * attribute_reconstruction_errors + (1 - beta) * structure_reconstruction_errors
    return cost


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    data = sio.loadmat("./dataset_with_tif/{}_tif.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    feat = data['Attributes'] if ('Attributes' in data) else data['X']
    adj = data['Network'] if ('Network' in data) else data['A']
    motifs = data['Motif']
    # print(type(adj))
    # constrasive
    adj_con = sp.csr_matrix(adj)
    feat_con = sp.lil_matrix(feat)
    motifs_con = sp.lil_matrix(motifs)

    # reconstruction
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj.toarray()
    feat = feat.toarray()
    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None
    # adj, feat, motifs for reconstruction
    # adj_con, feat_con, motifs_con for contrasive
    return adj_con, feat_con, motifs_con, adj, feat, motifs, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1.0,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)

        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
        # print(type(subv))
    return subv


def auc_roc(scores, target):
    # print('scores',scores)
    # print('target',target)
    return roc_auc_score(target, scores)


def auc_pr(scores, target):
    # print('scores',scores)
    precision, recall, _ = precision_recall_curve(target, scores)
    auc_pr = auc(recall, precision)
    return auc_pr


def precision(scores, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    pred = scores.argsort()[-maxk:][::-1]

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    correct = target[pred]
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(float).sum(0)
        res.append(correct_k * (1.0 / k))
    return res


def recall(scores, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    anomaly_size = target.sum()

    #    _, pred = scores.view(1,-1).topk(maxk, 1, True, True)
    pred = scores.argsort()[-maxk:][::-1]

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    correct = target[pred]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(float).sum(0)
        # print(correct_k)
        res.append(correct_k * (1.0 / anomaly_size))
    return res



