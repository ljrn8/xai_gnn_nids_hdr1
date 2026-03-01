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
                    in_channels=self.channels[i], out_channels=self.channels[i + 1]
                )
            )

        # add final binary linear layer
        self.layers.append(nn.Linear(self.channels[-1], 1))  

    def forward(self, edge_attr, edge_index, num_nodes):
        for i, channel in enumerate(self.channels):
            # ends with linear
            if i == len(self.channels) - 1:
                embeddings = edge_attr.clone().detach()
                linear_layer = self.layers[i]
                edge_attr = linear_layer(edge_attr)

            # SAGE
            else:
                SAGE_layer = self.layers[i]  # starts with SAGE
                edge_attr = SAGE_layer(
                    edge_attr,
                    edge_index,
                    node_attr=torch.ones([num_nodes, channel]),
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

            logits, emb = self.forward(G.edge_attr, G.edge_index, num_nodes=G.num_nodes)
            probs = torch.sigmoid(logits)
            y = G.y.to(device).float()
            loss = criterion(logits, y)

            if i % 100 == 0:
                logger.debug(
                    f"WINDOW index={i}: edge index shape {G.edge_index.shape} edge_attr shape {G.edge_attr.shape}"
                )
                # logger.debug(f"probs.shape = {probs.shape}")
                # logger.debug(f"y_probs unique = {np.unique(probs, return_counts=True)}")
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, edge_attr, edge_index, 
                node_attr, edge_weights=None):
        
        # edge_attr: (E, F)
        # edge_index: (2, E)
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.shape[1])
        assert edge_weights.shape[0] == edge_index.shape[1]

        num_nodes = edge_index.max().item() + 1
        src, dst = edge_index

        target_indices = edge_index[0, :].reshape(-1)
        edge_aggregated_mean = scatter_mean(
            edge_attr, target_indices, dim=0, dim_size=num_nodes  # (W, 10)  # (W)
        )

        mean_sum_embedds = node_attr + edge_aggregated_mean / 2
        node_embeddings = torch.sigmoid(self.W(mean_sum_embedds))

        # concatenate adjacent node embeddings to form edge embeddings
        edge_embeddings = (node_embeddings[src] + node_embeddings[dst]) / 2
        assert edge_embeddings.shape[0] == edge_attr.shape[0]
        return edge_embeddings
