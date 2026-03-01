## GNNExplainer runs
# sys args: <test_flows location> <model location (pkl)>
# EG: python GNNExplainer.py interm/unsw_nb15_processed_test.csv interm/runs/EGraphSAGE_anomdetection_UNSW_graphsage_20260223_085157/best_model.pkl

import pandas as pd
import numpy as np
import torch
from torch import nn
from ML_utils import yield_subgraphs
import sys
import pickle
from loguru import logger
from pathlib import Path

device = 'cpu'

# setup model and data
assert len(sys.argv) > 2
logger.info('loading data')
test_flows = pd.read_csv(Path(sys.argv[1]))
model_dir = Path(sys.argv[2])
logger.info('loading model ..')
with open(model_dir, 'rb') as f:
    model = pickle.load(f)

model.to(device)

# TODO use this smaller dataset for debugging
test_flows_prototype = test_flows.iloc[:100_000]
test_flows_prototype.to_csv('interm/unswnb15_protoype_test_partition.csv', index=False)

class GNNExplainer(nn.Module):
    def __init__(self, edge_attr_shape, feature_reg_weight, edge_reg_weight, **kwargs):
        super().__init__(**kwargs)
        self.feature_weight = feature_reg_weight
        self.edge_weight = edge_reg_weight
        # (n_edges, n_features)
        assert edge_attr_shape[0] > edge_attr_shape[1], (
            f"Expected (n_features, n_edges) but got shape where dim0 >= dim1: {edge_attr_shape}"
        )
        self.edge_mask = nn.Parameter(torch.zeros(edge_attr_shape))

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

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epc in range(1, 1 + epochs):
            losses, edge_regularization, feature_regularization = [], [], []
            for i, G in enumerate(yield_subgraphs(test_flows, window, linegraph=False)):
                optimizer.zero_grad()

                edge_weights = torch.sigmoid(self.edge_mask)
                masked_edge_attr = G.edge_attr * edge_weights

                # F(G) — original predictions
                with torch.no_grad():
                    y_pred, _ = model(G.edge_attr, G.edge_index, num_nodes=G.num_nodes)
                    y_pred = torch.sigmoid(y_pred)

                # f(G_S) — masked predictions,
                masked_y_pred, _ = model(masked_edge_attr, G.edge_index, num_nodes=G.num_nodes)
                masked_y_pred = torch.sigmoid(masked_y_pred)

                loss = loss_f(masked_y_pred, y_pred)
                er, fr = self.entropy_regularization(edge_weights)
                total_loss = loss + er + fr

                total_loss.backward()
                optimizer.step()

                losses.append(loss.detach())
                edge_regularization.append(er.detach())
                feature_regularization.append(fr.detach())

                del G

            losses_out = torch.stack(losses).cpu().numpy()
            edge_reg_out = torch.stack(edge_regularization).cpu().numpy()
            feature_reg_out = torch.stack(feature_regularization).cpu().numpy()

            yield (losses_out, edge_reg_out, feature_reg_out)


gnne = GNNExplainer(
    edge_attr_shape=(len(test_flows), len(test_flows.columns) - 3),
    edge_reg_weight=1e-2,
    feature_reg_weight=1e-2
)

epochs = 2
window = 10_000

for i, (losses_out, edge_reg_out, feature_reg_out) in enumerate(gnne.fit(
    model, test_flows, epochs, window
)):
    logger.info(
        f'EPOCH: {i+1}/{epochs} | '
        f"av loss for mask: {torch.mean(losses_out)} | "
        f"av edge regularizatino: {torch.mean(edge_reg_out)} | "
        f"av feature regularization: {torch.mean(feature_reg_out)}"
    )
    
