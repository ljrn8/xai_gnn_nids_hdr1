from torch import nn
import torch
from loguru import logger
from torch_scatter import scatter_mean
from ML_utils import yield_subgraphs
import numpy as np

class EGraphSAGE(nn.Module):
    """for binary classification"""

    def __init__(self, hidden_channels: list, num_features):
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.num_features = num_features
        self.channels = [num_features] + hidden_channels
        for i in range(len(self.channels) - 1):
            self.layers.append(
                EdgeSAGELayer(
                    in_channels=hidden_channels[i], 
                    out_channels=hidden_channels[i + 1]
                )
            )
        self.layers.append(nn.Linear(hidden_channels[i - 1] * 2, 1))  # binary for now
        super().__init__()

    def forward(self, edge_attr, edge_index, node_attr):
        """Node attr should be all 1s"""
        for i, layer in enumerate(self.num_layers):
            if i == len(self.num_layers):
                embeddings = edge_attr.copy()
            edge_attr = layer(node_attr, edge_attr, edge_index)

        return torch.sigmoid(edge_attr), embeddings

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
        - expects columns in flows: src, dst and Attack
        - expects attack labels to 0 (ben) and 1 (mal) for training
        - criterion must work on sigmoid probs (binary)
        """
        losses, y_trues, y_probs = [], [], []
        for i, G in enumerate(yield_subgraphs(flows, window, linegraph=False)):
            if train:
                optimizer.zero_grad()

            probs, embeddings = self.forward(G.edge_attr, G.edge_index, node_attr=torch.ones([]))
            y = G.y.to(device)
            loss = criterion(probs, y)

            if i % 20 == 0:
                logger.debug(
                    f"window G{i} edge index shape {G.edge_index.shape} x shape {G.x.shape}"
                )
                logger.debug(f"y train shape = {y.shape}")
                logger.debug(f"y train unique = {np.unique(y, return_counts=True)}")

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            y_trues.append(y)
            y_probs.append(probs)
            del G

        y_trues = torch.cat(y_trues).cpu().numpy()
        y_probs = torch.cat(y_probs).cpu().numpy()
        avg_loss = np.mean(losses)
        logger.debug(
            f"train y_trues.shape, sum {y_trues.shape}, {y_trues.sum()} | y_preds.shape, sum {y_probs.shape},{y_probs.sum()}",
        )
        return avg_loss, y_trues, y_probs, embeddings


class EdgeSAGELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """x should be 2 * feature size (raw vector concatentation for each layer)"""
        super().__init__()
        self.W = nn.Linear(in_channels * 2, out_channels)

    def forward(self, node_attr, edge_attr, edge_index):
        # edge_attr: (E, F)
        # edge_index: (2, E)
        num_nodes = edge_index.max().item() + 1
        src, dst = edge_index

        assert edge_attr.shape[1] > edge_attr.shape[0]
        assert edge_index.shape[0] > edge_index.shape[1]
        target_indices = edge_index[0, :]
        edge_aggregated_mean = scatter_mean(
            edge_attr, target_indices, dim=0, dim_size=num_nodes
        )
        node_embeddings = torch.sigmoid(
            torch.matmul(self.W, (node_attr + edge_aggregated_mean) / 2)
        )

        # concatenate adjacent node embeddings to form edge embeddings
        edge_embeddings = torch.cat(edge_index[src], edge_index[dst])
        return edge_embeddings
