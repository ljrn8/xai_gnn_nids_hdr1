from preprocessing import graph_encode
from tqdm import tqdm
import torch
import torch.nn.functional as F

device = 'cpu'

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
    return (
        loss_fn(out, y_test), 
        out, y_test
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
        ys.append(y)
        ypreds.append(out.argmax(dim=1))

    return losses, sum(losses) / len(losses), ys, ypreds