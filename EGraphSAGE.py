import os
from torch import nn
import torch
from loguru import logger
from torch_scatter import scatter_mean
from ML_utils import yield_subgraphs
import numpy as np


class EGraphSAGE(nn.Module):
    """for binary classification"""

    def __init__(self, hidden_channels: list, num_features):
        super(EGraphSAGE, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.num_features = num_features
        self.channels = [num_features] + hidden_channels
        self.layers = nn.ModuleList()

        # add E-SAGE layers
        for i in range(len(self.channels) - 1):
            self.layers.append(
                EdgeSAGELayer(
                    edge_features=self.num_features,
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                )
            )

        # add final binary linear layer
        self.layers.append(nn.Linear(self.channels[-1] * 2, 1))

    def forward(self, edge_attr, edge_index, node_attr):
        for i, channel in enumerate(self.channels):

            # linear
            if i == len(self.channels) - 1:

                # concat node embeddings for each edge
                src, dst = edge_index
                edge_embs = torch.cat([node_attr[src], node_attr[dst]], dim=1)

                # hold embeddings
                embeddings = edge_embs.clone().detach()

                # final binary MLP layer
                linear_layer = self.layers[i]
                edge_attr = linear_layer(edge_embs)

            # SAGE
            else:
                SAGE_layer = self.layers[i]
                node_attr = SAGE_layer(
                    edge_attr,
                    edge_index,
                    node_attr=node_attr,
                )

        return edge_attr.view(-1), embeddings

    def train_flows(
        self,
        flows,
        criterion,
        window,
        optimizer,
        train=True,
        device="cpu",
    ):
        """
        - expects columns in flows: src, dst and Attack, and all numeric
        - expects Attack to 0 (ben) and 1 (mal) for training
        - criterion must work on sigmoid probabilities (binary)
        """
        losses, y_trues, y_probs, embeddings = [], [], [], []
        for i, G in enumerate(yield_subgraphs(flows, window, linegraph=False)):
            if train:
                optimizer.zero_grad()

            n_nodes = G.edge_index.max().item() + 1
            logits, emb = self.forward(
                G.edge_attr,
                G.edge_index,
                node_attr=torch.ones(size=(n_nodes, self.num_features)).to(device),
            )
            probs = torch.sigmoid(logits)
            y = G.y.to(device).float()
            loss = criterion(logits, y)

            if i % 100 == 0:
                logger.debug(
                    f"WINDOW index={i}: edge index shape {G.edge_index.shape} edge_attr shape {G.edge_attr.shape}"
                )
                logger.debug(f"y shape = {y.size()}")
                logger.debug(f"y unique = {np.unique(y, return_counts=True)}")

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            y_trues.append(y)
            y_probs.append(probs)
            embeddings.append(emb)
            del G

        y_trues = torch.cat(y_trues).cpu().detach().numpy()
        y_probs = torch.cat(y_probs).cpu().detach().numpy()
        avg_loss = np.mean(losses)
        logger.debug(
            f"BATCH COMPLETE: train y_trues.shape, sum {y_trues.shape}"
            f"{y_trues.sum()} | y_preds.shape, sum {y_probs.shape}, {y_probs.sum()}"
        )
        return (avg_loss, y_trues, y_probs, embeddings)


class EdgeSAGELayer(nn.Module):
    def __init__(self, edge_features, in_channels, out_channels):
        super().__init__()
        self.W = nn.Linear(in_channels + edge_features, out_channels)

    def forward(self, edge_attr, edge_index, node_attr):
        # edge_attr: (E, F)
        # edge_index: (2, E)
        num_nodes = edge_index.max().item() + 1

        target_indices = edge_index[0, :].reshape(-1)
        edge_aggregated_mean = scatter_mean(
            edge_attr, target_indices, dim=0, dim_size=num_nodes  # (W, 10)  # (W)
        )

        node_edge_concat_embs = torch.cat([node_attr, edge_aggregated_mean], dim=1)
        new_node_embeddings = torch.sigmoid(self.W(node_edge_concat_embs))

        assert new_node_embeddings.shape[0] == node_attr.shape[0]
        return new_node_embeddings
