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

    def __init__(self, layer_sizes: list, flow_features, output_dim=1):
        super(EGraphSAGE, self).__init__()
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_features = flow_features
        self.channels = [flow_features] + layer_sizes
        self.layers = nn.ModuleList()

        # Edge and Node embedders for each layer
        for i, _ in enumerate(self.channels[:-1]):
            self.layers.append(
                PairedSAGELayer(
                    edge_features=flow_features,
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                )
            )

        # final binary linear layer
        # x2 for node concatentation to re-form edge embeddings
        self.layers.append(nn.Linear(self.channels[-1] * 2, output_dim))

    def init_node_embeddings(
        self, edge_index, edge_attr, node_weight=None, edge_weight=None
    ):
        num_nodes = edge_index.max().item() + 1

        # initialise node attr as mean of neighbouring edge features
        node_attr = scatter_mean(
            edge_attr if edge_weight is None else edge_attr * edge_weight.unsqueeze(1),
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
        )

        # apply node weight to node initials
        if node_weight is not None:
            node_attr = node_attr * node_weight.unsqueeze(1)

        return node_attr

    def forward(
        self, edge_attr, edge_index, node_attr=None, edge_weight=None, node_weight=None
    ):
        if node_attr is None:
            node_attr = self.init_node_embeddings(
                edge_index, edge_attr, node_weight, edge_weight
            )

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
                SAGE_pair_layer = self.layers[i]
                _, node_attr = SAGE_pair_layer(
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

        # apply node weights to neighbourhood aggregation
        # (node attr already has the mask applied)
        if node_weight is not None:
            edge_aggregated_mean = edge_aggregated_mean * node_weight.unsqueeze(1)

        node_edge_concat_embs = torch.cat([node_attr, edge_aggregated_mean], dim=1)
        new_node_embeddings = torch.relu(self.W(node_edge_concat_embs))

        assert new_node_embeddings.shape[0] == node_attr.shape[0]
        return new_node_embeddings


class SAGELayer(nn.Module):
    """
    Produces new node embeddings based off aggregated node neighbourhood
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # x2 for (node neighbour + previous node) embeddings
        self.W = nn.Linear(in_channels * 2, out_channels)

    def forward(self, edge_index, node_attr, node_weight=None):
        # edge_attr: (E, F)
        # edge_index: (2, E)
        num_nodes = edge_index.max().item() + 1
        src, dst = edge_index

        # gather source node features for each edge
        src_features = node_attr[src]  # (E, in_channels)

        # mean aggregate neighbour (src) features into each dst node
        aggregated = scatter_mean(
            src_features, dst, dim=0, dim_size=num_nodes
        )  # (N, in_channels)

        # apply node weight to neighbourhood aggregation 
        # Node attr alread has been masked out
        if node_weight is not None:
            aggregated = aggregated * node_weight.unsqueeze(1)

        # concatenate each node's own features with its aggregated neighbourhood
        node_concat = torch.cat([node_attr, aggregated], dim=1)  # (N, in_channels * 2)

        # apply linear transform
        new_node_embeddings = torch.relu(self.W(node_concat))  # (N, out_channels)

        return new_node_embeddings


class PairedSAGELayer(nn.Module):
    """
    Edge aggregation for node embdeddings, followed by node neighbourhood aggregation
    """

    def __init__(self, edge_features: int, in_channels: int, out_channels: int):
        super().__init__()
        self.edge_SAGE = EdgeSAGELayer(edge_features, in_channels, in_channels)
        self.node_SAGE = SAGELayer(in_channels, out_channels)

    def forward(
        self, edge_index, edge_attr, node_attr, edge_weight=None, node_weight=None
    ):
        concated_node_embeddings = self.edge_SAGE.forward(
            edge_attr=edge_attr,
            edge_index=edge_index,
            node_attr=node_attr,
            edge_weight=edge_weight,
            node_weight=node_weight,
        )

        neighbour_concatenated_node_embeddings = self.node_SAGE.forward(
            node_attr=concated_node_embeddings,
            edge_index=edge_index,
        )

        return concated_node_embeddings, neighbour_concatenated_node_embeddings
