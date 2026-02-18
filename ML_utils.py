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

import torch
import torch.nn.functional as F

DEBUG = 0
init()  # colour logs
device = "cpu"

def debug(message, **kwargs):
    log(message, **kwargs, debug=True, c='red')


def log(message, c=None, debug=False, lstrip=False):
    fore_col = Fore.__getattribute__(c.upper()) if c else ""
    reset = Style.RESET_ALL if c else ""
    if lstrip:
        new_message = ""
        for line in message.split('\n'):
            if len(line.strip()) < 1:
                continue 
            new_message += line.lstrip() + '\n'
        message = new_message[:-1] # remove last line break

    if not DEBUG and debug:
        return # skip without debug option
    
    debug_prefix = "DEBUG" if debug and DEBUG else ""
    print(f"{fore_col} > {debug_prefix} LOG: {message} {reset}", flush=True)



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


def epoch(
    model, flows, loss_fn, optimizer=None, evaluate_instead=False, window_size=500
):
    """
    Assumes 'flows' has columns src and dst for edge construction,
    and 'Attack' for labels (float).
    """
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_flows = len(flows)
    losses, ys, ypreds = [], [], []
    for i, start in enumerate(
        tqdm(range(0, n_flows - 1, window_size)),
    ):
        window_flows = flows.iloc[start : start + window_size]
        window_graph, node_map = graph_encode(
            window_flows,
            linegraph=True,  # ! uses linegraphs
            edge_cols=["src", "dst"],
            target_col="Attack",
        )

        if evaluate_instead:
            loss, out, y = eval_graph(model, window_graph, loss_fn=loss_fn)
        else:
            loss, out, y = train_graph(model, window_graph, optimizer, loss_fn=loss_fn)

        losses.append(loss)
        ys.append(y)
        ypreds.append(out.argmax(dim=1))

    return losses, sum(losses) / len(losses), ys, ypreds


def yield_subgraphs(flows, window, target_col="Attack"):
    for start in range(0, len(flows) - 1, window):
        window_flows = flows.iloc[start : start + window].copy()
        debug(f'window computed [{start}:{start+window}]. mal flows: {sum(window_flows.Attack)}')
        # y = deepcopy(window_flows.Attack.values)
        # window_flows.drop('Attack', axis=1, inplace=True)
        window_graph, _ = graph_encode(
            window_flows,
            linegraph=True,
            edge_cols=["src", "dst"],
            target_col=target_col,
        )
        yield window_graph


def graph_encode(data, edge_cols: list, linegraph: bool, target_col: str = None):
    """convert flows df to pyG graph with G.x[-1] as target"""

    assert target_col in data.columns
    cols = [c for c in data.columns if c != target_col] + [target_col]
    data = data[cols]

    attrs = [
        c
        for c in data.columns
        if c not in edge_cols 
    ]

    x = data[attrs].to_numpy(dtype=np.float32)
    edge_attr = torch.tensor(x, dtype=torch.float)

    nodes = pd.concat([data["src"], data["dst"]]).unique()
    node_map = {n: i for i, n in enumerate(nodes)}

    src_name, dst_name = edge_cols
    src = data[src_name].map(node_map).to_numpy()
    dst = data[dst_name].map(node_map).to_numpy()

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    G = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(nodes),
    )

    # make bidirectional
    edge_index = G.edge_index
    edge_attr = G.edge_attr
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    G.edge_index = edge_index
    G.edge_attr = edge_attr

    if linegraph:
        G = LineGraph()(G)

        # ensure linegraph is also birectional
        edge_index = G.edge_index
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        G.edge_index = edge_index


    return G, node_map
