import torch
import numpy as np
import torch
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from copy import deepcopy
from tqdm import tqdm
from colorama import init, Fore, Style
from loguru import logger
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, auc
import torch

device = 'cpu'

def get_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor):
    y_pred = y_pred_probs > 0.5
    P, R = (
        precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1),
    )
    precision, recall, _ = precision_recall_curve(
        y_true, y_pred_probs.detach().numpy(), pos_label=1
    )
    pr_auc = auc(recall, precision)
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return (pr_auc, roc_auc_score(y_true, y_pred_probs.detach().numpy()), F1, P, R)


def write_metrics(
    y_trues: torch.Tensor, y_probs: torch.Tensor, writer, epc, av_loss, train_category: bool
):
    all_metrics = get_metrics(y_trues, y_probs)
    pr_auc, roc_auc, f1, prec, rec = all_metrics
    metrics = {"PR-AUC": pr_auc, "ROC-AUC": roc_auc, "F1": f1, "PREC": prec, "rec": rec, 'avg_loss': av_loss}
    istrain = "TRAIN" if train_category else "TEST"
    for metric_name, metric in metrics.items():
        writer.add_scalar(f"all/{metric_name}_{istrain}", metric, epc)
    return all_metrics


def most_recent_object(exp_dir):
    exp_dirs = list(Path(exp_dir).glob("*"))
    exp_dir = max(exp_dirs, key=lambda d: d.stat().st_ctime)
    logger.info(f"Using newest experiment directory: {exp_dir}")
    return Path(exp_dir)


def fidelities(y_pred, y_mask, y_imask, y):
    """Phenominal fidelity+ and Fidelity- (expects THRESHOLDED values)"""
    fp = ((y_pred == y).float() - (y_imask == y).float()).abs().mean()
    fm = ((y_pred == y).float() - (y_mask == y).float()).abs().mean()
    return fp, fm


def train_graph(model, train_graph, optimizer, loss_fn, y_train):
    model.train()
    optimizer.zero_grad()
    G = train_graph.to(device)
    out = model(G.x.to(device), G.edge_index.to(device))
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    return loss.item(), out, y_train


def eval_graph(model, test_graph, loss_fn, y_test):
    with torch.no_grad():
        out = model(test_graph.x.to(device), test_graph.edge_index.to(device))
    y_test
    return (loss_fn(out, y_test), out, y_test)


def graph_encode(data, edge_cols: list, linegraph: bool, target_col: str = None):
    assert target_col in data.columns
    # keep label separate and explicit
    labels = data[target_col].to_numpy(dtype=np.int64)  # shape (E,)
    cols = [c for c in data.columns if c != target_col] + [target_col]
    data = data[cols]

    # feature columns = all columns except edge_cols and target_col
    feature_cols = [c for c in data.columns if c not in edge_cols + [target_col]]
    edge_features = data[feature_cols].to_numpy(dtype=np.float32)  # (E, F)
    edge_labels = torch.tensor(labels, dtype=torch.long)  # (E,)

    nodes = pd.concat([data["src"], data["dst"]]).unique()
    node_map = {n: i for i, n in enumerate(nodes)}

    src = data[edge_cols[0]].map(node_map).to_numpy()
    dst = data[edge_cols[1]].map(node_map).to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)  # (2, E)

    G = Data(
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        num_nodes=len(nodes),
    )

    if linegraph:
        # DO NOT make bidirectional yet; call LineGraph on the original directed edges
        # LineGraph will create nodes representing each original edge in the same order,
        # so edge_labels align with the new nodes.
        G = LineGraph()(G)

        # after LineGraph, G.x contains the edge_attr (node features for line-graph)
        # and node-count equals original number of edges. Attach labels explicitly:
        G.y = edge_labels.clone()  # shape matches G.x rows

        # If you want undirected line-graph edges (bi-directional adjacency), do it now:
        G.edge_index = torch.cat([G.edge_index, G.edge_index.flip(0)], dim=1)

    else:
        # For non-linegraph case you probably want node features; set them explicitly if needed
        # e.g., create node features or put edge features somewhere else.

        G.y = edge_labels.clone()

    return G, node_map
