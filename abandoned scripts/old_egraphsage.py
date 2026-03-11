import os
from torch import nn
import torch
from loguru import logger
from torch_scatter import scatter_mean
from ML_utils import yield_subgraphs
import numpy as np
from torch import Tensor


class EGraphSAGE(nn.Module):
    """for binary flow classification"""

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

    def forward(
        self, edge_attr, edge_index, node_attr=None, edge_weight=None, node_weight=None
    ):
        if not node_attr:
            num_nodes = edge_index.max().item() + 1
            # initialise node attr as mean of neighbouring edge features
            node_attr = scatter_mean(
                (
                    edge_attr
                    if edge_weight is None
                    else edge_attr * edge_weight.unsqueeze(1)
                ),
                edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )

            # apply node weight to node initials
            if node_weight is not None:
                node_attr = node_attr * node_weight.unsqueeze(1)

        for i, channel in enumerate(self.channels):
            # linear
            if i == len(self.channels) - 1:
                # concat node embeddings for each edge
                src, dst = edge_index
                node_embeddings = node_attr.clone()
                edge_embs = torch.cat([node_attr[src], node_attr[dst]], dim=1)
                # hold embeddings
                edge_embeddings = edge_embs.clone()
                # final binary MLP layer
                linear_layer = self.layers[i]
                edge_attr = linear_layer(edge_embs)

            # SAGE
            else:
                SAGE_layer = self.layers[i]
                node_attr = SAGE_layer(
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    node_attr=node_attr,
                    edge_weight=edge_weight,
                    node_weight=node_weight,
                )

        return edge_attr.view(-1), edge_embeddings, node_embeddings

    def pass_flowgraph(
        self,
        G,
        criterion,
        optimizer,
        edge_weight=None,
        node_weight=None,
        train=True,
        device="cpu",
        debug=True,
    ):
        if train:
            optimizer.zero_grad()
        else:
            self.eval()

        n_nodes = G.edge_index.max().item() + 1
        logits, emb, node_emb = self.forward(
            G.edge_attr, G.edge_index, edge_weight=edge_weight, node_weight=node_weight
        )
        probs = torch.sigmoid(logits)
        y = G.y.to(device).float()
        loss = criterion(logits, y)
        if debug:
            logger.debug(
                f"Graph pass done: edge index shape {G.edge_index.shape} edge_attr shape {G.edge_attr.shape}"
            )
            logger.debug(f"y shape = {y.size()}")
            logger.debug(f"y unique = {np.unique(y, return_counts=True)}")
            logger.debug(f"probs average = {torch.mean(probs)}")

        if train:
            loss.backward()
            optimizer.step()

        return loss, y, probs, emb

    def _pass_flowgraph_windows(
        self,
        flows,
        criterion,
        window,
        optimizer,
        train=True,
        device="cpu",
    ):
        """!! Old method do not use
        - expects columns in flows: src, dst and Attack, and all numeric
        - expects Attack to 0 (ben) and 1 (mal) for training
        - criterion must work on sigmoid probabilities (binary)
        """
        losses, y_trues, y_probs, embeddings = [], [], [], []
        for i, G in enumerate(yield_subgraphs(flows, window, linegraph=False)):
            loss, y, probs, emb = self.pass_flowgraph(
                G, criterion, optimizer, train=train, device=device, debug=i % 50 == 0
            )
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
    """
    Produces new node embeddings based off Aggregated edge features
    """

    def __init__(self, edge_features: int, in_channels: int, out_channels: int):
        super().__init__()
        self.W = nn.Linear(in_channels + edge_features, out_channels)

    def forward(
        self, edge_index, edge_attr, node_attr, edge_weight=None, node_weight=None
    ):
        # edge_attr: (E, F)
        # edge_index: (2, E)
        assert edge_index.shape[0] == 2, "this is edge index (changed order recently)"

        num_nodes = edge_index.max().item() + 1

        # apply edge weights BEGORE neighbourhood aggregation
        if edge_weight is not None:
            assert (
                edge_weight.shape[0] == edge_attr.shape[0]
            ), f"bad weight shape in forward, not {edge_attr.shape[0]}"
            edge_attr = edge_attr * edge_weight.unsqueeze(1)

        target_indices = edge_index[0, :].reshape(-1)
        edge_aggregated_mean = scatter_mean(
            edge_attr, target_indices, dim=0, dim_size=num_nodes  # (W, 10)  # (W)
        )

        node_edge_concat_embs = torch.cat([node_attr, edge_aggregated_mean], dim=1)
        new_node_embeddings = torch.relu(self.W(node_edge_concat_embs))

        # apply node weights AFTER neighbourhood aggregation
        if node_weight is not None:
            assert (
                node_weight.shape[0] == new_node_embeddings.shape[0]
            ), f"bad weight shape in forward, not {new_node_embeddings.shape[0]}"
            new_node_embeddings = new_node_embeddings * node_weight.unsqueeze(1)

        assert new_node_embeddings.shape[0] == node_attr.shape[0]
        return new_node_embeddings
