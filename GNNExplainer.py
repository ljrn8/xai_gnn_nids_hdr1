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
# assert len(sys.argv) > 2
test_f = Path('interm/unsw_nb15_processed_test.csv')
model_dir = Path('interm/runs/EGraphSAGE_anomdetection_UNSW_graphsage_20260303_033347/best_model.pkl')

logger.info('loading data')

# !NOTE using prototype smaller dataset for debugging
test_flows = pd.read_csv(test_f, nrows=100_000)

logger.info('loading model ..')
with open(model_dir, 'rb') as f:
    model = pickle.load(f)

model.to(device)

class GNNExplainer(nn.Module):
    def __init__(self, feature_reg_weight=0.1, edge_reg_weight=0.1, l1_norm_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_reg_weight = feature_reg_weight
        self.edge_reg_weight = edge_reg_weight
        self.l1_norm_weight = l1_norm_weight

    def elementwise_entropy(x):
        return -x * torch.log2(x) - (1-x)*torch.log2(1-x)

    def regularization(self, edge_mask, feature_mask):
        reg = 0
        reg += self.edge_reg_weight * torch.mean([self.elementwise_entropy(e) for e in edge_mask])
        reg += self.feature_reg_weight * torch.mean([self.elementwise_entropy(e) for e in feature_mask])
        reg += self.l1_norm_reg * (torch.mean(edge_mask) + torch.mean(feature_mask) / 2) 
        return reg
    
    def sample_from_empirical(edge_attr, feature_bank):
        n_edges, n_features = edge_attr.shape
        N = feature_bank.shape[0]
        idx = torch.randint(0, N, (n_edges, n_features))
        Z = feature_bank[idx, torch.arange(n_features).unsqueeze(0)]
        return Z  # (n_edges, n_features)

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
            feature_mask = torch.randn((num_features), requires_grad=True, device=device)
            edge_mask = torch.randn((G.edge_attr.shape[0]), requires_grad=True, device=device)
            optimizer = torch.optim.Adam([edge_mask, feature_mask], lr=lr)
            losses, regularization_penalties = [], []

            for epc in range(1, 1 + epochs):
                edge_weights = torch.sigmoid(edge_mask)
                feature_weights = torch.sigmoid(feature_mask)
                
				# impirical marginal distribution sampling for noise features (Z)
                optimizer.zero_grad()
                Z = self.sample_from_empirical(G.edge_attr, 
                                          feature_bank=test_flows[test_flows.columns.difference(['src', 'dst', 'Attack'])].values)

                # reparameterization trick
                feature_masked_edge_attr = Z + ((G.edge_attr - Z) * feature_weights)
                fully_masked_edge_attr = G.edge_attr * edge_weights.unsqueeze(1)

                # F(G) — original predictions
                with torch.no_grad():
                    y_pred, _ = model.forward(G.edge_attr, G.edge_index,
                                              node_attr=torch.ones(size=(num_nodes, num_features)).to(device))
                    y_pred = torch.sigmoid(y_pred)

                # f(G_S) — masked predictions
                masked_y_pred, _ = model.forward(fully_masked_edge_attr, G.edge_index,
                                                 node_attr=torch.ones(size=(num_nodes, num_features)).to(device))
                masked_y_pred = torch.sigmoid(masked_y_pred)

                loss = loss_f(masked_y_pred, y_pred)
                reg = self.entropy_regularization(edge_weights)
                total_loss = loss + reg

                total_loss.backward()
                optimizer.step()

                losses.append(loss.detach())
                regularization_penalties.append(reg.detach())

                logger.info(
                    f'epoch: {epc} | '
                    f"av loss for mask: {np.mean(losses):.5f} | "
                    f"edge mask average value: {torch.mean(edge_mask):.5f} | "
                    f"feature mask average value: {torch.mean(feature_mask):.5f}"
                    f"regularization penalty: {reg:.5f} | "
                )

            losses_out = torch.stack(losses).cpu().numpy()
            regularization_penalties_out = torch.stack(regularization_penalties).cpu().numpy()

            yield (window, edge_mask, feature_mask, losses_out, regularization_penalties_out)


gnne = GNNExplainer(
    edge_reg_weight=1e-2,
    feature_reg_weight=1e-2
)

EPOCHS = 10
WINDOW = 10_000
LR = 0.01

# convert test_flows Attack to binary
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()

for i, (window, edge_mask, feature_mask, losses_out, regularization_penalties_out) in enumerate(gnne.fit(
    model=model, test_flows=test_flows, epochs=EPOCHS, window=WINDOW, lr=LR, loss_f=torch.nn.BCELoss()
)):
    logger.info(
        f'LEARNED WINDOW: {i+1} | '
        f"av loss for mask: {np.mean(losses_out):.5f} | "
    )

