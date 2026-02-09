import pandas as pd
from preprocessing import graph_encode
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

device = 'cpu'

def train_graph(model, train_graph):
    model.train()
    optimizer.zero_grad()
    G = train_graph.to(device)   
    out= model(G.x.to(device), G.edge_index.to(device))
    loss = F.cross_entropy(out, G.Attack)
    loss.backward()
    optimizer.step()
    return loss.item()
        
def eval_graph(model, test_graph):
    model.eval()
    with torch.no_grad():
        out = model(test_graph.x.to(device), test_graph.edge_index.to(device))
    return F.cross_entropy(out, test_graph.Attack)


# -- Script ---

print('Loading data...')
flows = pd.read_csv('interm/unsw_nb15_processed.csv')
n_flows = len(flows)
size = 500

model = GraphSAGE(
    49,
    hidden_channels=256,
    out_channels=2,
    num_layers=3,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for i, start in enumerate(
    tqdm(range(0, n_flows - 1, size))
):
    window_flows = flows.iloc[start:start + size]
    window_graph, node_map = graph_encode(
                        window_flows, 
                       linegraph=True,  # ! uses linegraphs
                       edge_cols=['src', 'dst'], 
                       target_col='Attack')
    
    train_graph = Data(x=window_graph.x, edge_index=window_graph.edge_index, Attack=window_graph.Attack)
    loss = train_graph(model, train_graph)
    print(f'Window {i}, Loss: {loss}')
    

