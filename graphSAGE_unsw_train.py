import pandas as pd
from preprocessing import graph_encode
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE

device = 'cpu'

model = GraphSAGE(
    49,
    hidden_channels=256,
    out_channels=5,
    num_layers=3,
).to(device)


flows = pd.read_csv('interm\unsw_nb15_processed.csv')

n_flows = len(flows)
size = 500
for i, start in enumerate(
    tqdm(range(0, n_flows - 1, size))
):
    window_flows = flows.iloc[start:start + size]
    window_graph, node_map = graph_encode(window_flows, 
                       linegraph=True,  # ! uses linegraphs
                       edge_cols=['src', 'dst'], 
                       target_col='Attack')
    

