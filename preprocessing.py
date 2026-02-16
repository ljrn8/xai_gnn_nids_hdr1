import torch
import numpy as np
import torch
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph

def yield_subgraphs(flows, window, le=None):
    for start in range(0, len(flows) - 1, window):
        window_flows = flows.iloc[start:start + window].copy()
        y = window_flows.Attack
        window_flows.drop('Attack', axis=1, inplace=True)
        window_graph, _ = graph_encode(
            window_flows,
            linegraph=True,
            edge_cols=['src', 'dst'],
        )
        if le:
            yield window_graph, torch.LongTensor(le.transform(window_flows.Attack))
        else:
            yield window_graph, torch.LongTensor(y.values)


def graph_encode(data, edge_cols: list, 
                 linegraph: bool, 
                #  target_col: str
                ):
    """ Convert flow dataframe  (all cols numerical) 
    to flow graph """

    # ----- build edge features -----
    attrs = [c for c in data.columns if c not in 
             edge_cols 
            #  + [target_col]
             ]

    x = data[attrs].to_numpy(dtype=np.float32)
    edge_attr = torch.tensor(x, dtype=torch.float)

    # edge_y = torch.tensor(
        # data[target_col].values, dtype=torch.long
    # )

    nodes = pd.concat([data['src'], data['dst']]).unique()
    node_map = {n: i for i, n in enumerate(nodes)}

    src_name, dst_name = edge_cols
    src = data[src_name].map(node_map).to_numpy()
    dst = data[dst_name].map(node_map).to_numpy()

    edge_index = torch.tensor(
        np.stack([src, dst]), dtype=torch.long
    )

    G = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        # edge_y=edge_y,
        num_nodes=len(nodes)
    )

    if linegraph:
        G = LineGraph()(G)

    return G, node_map





