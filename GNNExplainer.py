## GNNExplainer runs
# sys args: <test_flows location> <model location (pkl)>
# EG python GNNExplainer.py interm/unsw_nb15_processed_test.csv interm/runs/EGraphSAGE_anomdetection_UNSW_graphsage_20260303_002358/best_model.pkl


from ML_utils import yield_subgraphs
import pandas as pd
from torch import nn
import torch
from tqdm import tqdm
import sys
import pickle
from loguru import logger
from pathlib import Path
from EGraphSAGE import EGraphSAGE
import numpy as np

device = 'cpu'

# setup model and data
assert len(sys.argv) > 2
logger.info('loading data')

# !NOTE using prototype smaller dataset for debugging
test_flows = pd.read_csv(Path(sys.argv[1]), nrows=100_000)

model_dir = Path(sys.argv[2])
logger.info('loading model ..')
with open(model_dir, 'rb') as f:
    model = pickle.load(f)

model.to(device)

class GNNExplainer(nn.Module):
    def __init__(self, feature_reg_weight, 
                 edge_reg_weight, **kwargs):
        
        super().__init__(**kwargs)
        self.feature_weight = feature_reg_weight
        self.edge_weight = edge_reg_weight
        # (n_edges, n_features)

    def entropy_regularization(self, soft_mask):
        n_edges, n_features = soft_mask.shape
        # clamp to avoid log(0)

        reg = 1e-8
        soft_mask = soft_mask.clamp(reg, 1 - reg)

        # edge entropy
        edge_mean_importances = torch.stack([
            torch.mean(soft_mask[i, :])
            for i in range(n_edges)
        ])
        edge_reg = self.edge_weight * torch.sum(edge_mean_importances * torch.log(edge_mean_importances))

        # feature entropy
        feature_entropies = torch.stack([
            -torch.sum(soft_mask[i, :] * torch.log(soft_mask[i, :]))
            for i in range(n_edges)
        ])
        feature_reg = self.feature_weight * torch.mean(feature_entropies)

        return edge_reg, feature_reg

    def fit(self, model, test_flows, epochs,
                 window, lr=0.01, loss_f=torch.nn.BCELoss()):

        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False


        for i, G in enumerate(yield_subgraphs(test_flows, window, linegraph=False)):
            num_features = len(test_flows.columns) - 3 # src dst Attack
            num_nodes = G.edge_index.max().item() + 1

            # initialize edge mask with requires_grad
            learned_mask = nn.Parameter(torch.randn(G.edge_attr.shape).to(device), 
                                        requires_grad=True)
            optimizer = torch.optim.Adam([learned_mask], lr=lr)
            losses, edge_regularization, feature_regularization = [], [], []

            for epc in range(1, 1 + epochs):

                edge_weights = torch.sigmoid(learned_mask)
                optimizer.zero_grad()

                masked_edge_attr = G.edge_attr * edge_weights

                # F(G) — original predictions
                with torch.no_grad():
                    y_pred, _ = model.forward(G.edge_attr, G.edge_index, 
                                              node_attr=torch.ones(size=(num_nodes, num_features)).to(device))
                    y_pred = torch.sigmoid(y_pred)

                # f(G_S) — masked predictions
                masked_y_pred, _ = model.forward(masked_edge_attr, G.edge_index, 
                                                 node_attr=torch.ones(size=(num_nodes, num_features)).to(device))
                masked_y_pred = torch.sigmoid(masked_y_pred)

                loss = loss_f(masked_y_pred, y_pred)
                er, fr = self.entropy_regularization(edge_weights)
                total_loss = loss + er + fr

                total_loss.backward()
                optimizer.step()

                losses.append(loss.detach())
                edge_regularization.append(er.detach())
                feature_regularization.append(fr.detach())


            losses_out = torch.stack(losses).cpu().numpy()
            edge_reg_out = torch.stack(edge_regularization).cpu().numpy()
            feature_reg_out = torch.stack(feature_regularization).cpu().numpy()

            yield (window, learned_mask, losses_out, edge_reg_out, feature_reg_out)


gnne = GNNExplainer(
    edge_reg_weight=1e-2,
    feature_reg_weight=1e-2
)

EPOCHS = 2
WINDOW = 10_000
LR = 0.01

# convert test_flows Attack to binary
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()

for i, (window, learned_parameters, losses_out, edge_reg_out, feature_reg_out) in enumerate(gnne.fit(
    model=model, test_flows=test_flows, epochs=EPOCHS, window=WINDOW, lr=LR, loss_f=torch.nn.BCELoss()
)):
    logger.info(
        f'LEARNED WINDOW: {i+1} | '
        f"av loss for mask: {np.mean(losses_out):.5f} | "
        f"av edge regularizatino: {np.mean(edge_reg_out):.5f} | "
        f"av feature regularization: {np.mean(feature_reg_out):.5f}"
        f"mask grad={learned_parameters.grad.norm():.5f}"
    )
    
