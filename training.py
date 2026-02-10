from preprocessing import graph_encode
from tqdm import tqdm
import torch
import torch.nn.functional as F

device = 'cpu'

def train_graph(model, train_graph, optimizer, loss_fn, target_col='edge_y'):
    model.train()
    optimizer.zero_grad()
    G = train_graph.to(device)   
    out = model(G.x.to(device), G.edge_index.to(device))
    y = getattr(G, target_col).long()
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    return loss.item(), out, y
        
def eval_graph(model, test_graph, loss_fn, target_col='edge_y'):
    model.eval()
    with torch.no_grad():
        out = model(test_graph.x.to(device), test_graph.edge_index.to(device))
    y = getattr(test_graph, target_col)
    return (
        loss_fn(out, y), 
        out, y
    )

def epoch(model, flows, loss_fn, optimizer=None, 
          evaluate_instead=False, window_size=500):
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
        window_flows = flows.iloc[start:start + window_size]
        window_graph, node_map = graph_encode(
                                window_flows, 
                                linegraph=True,  # ! uses linegraphs
                                edge_cols=['src', 'dst'], 
                                target_col='Attack')
        
        if evaluate_instead:
            loss, out, y = eval_graph(model, window_graph, loss_fn=loss_fn)
        else:
            loss, out, y = train_graph(model, window_graph, optimizer, loss_fn=loss_fn)

        losses.append(loss)
        ys.append(y.cpu())
        ypreds.append(out.argmax(dim=1).cpu())

    return losses, sum(losses) / len(losses), ys, ypreds