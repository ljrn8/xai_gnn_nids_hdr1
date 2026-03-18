import os
from torch import nn
import torch
from loguru import logger
from torch_scatter import scatter_mean
from ML_utils import graph_encode
import numpy as np
from torch import Tensor
import pickle
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from ML_utils import write_metrics
from tqdm import tqdm
import itertools


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
        train_now=True,
        device="cpu",
        debug=True,
    ):
        """Pass a single flow graph, for training, later training or evaluation
        Returns: loss (val), y (Tensor), probs (Tensor), embeddings (Tensor)
        """
        if train_now:
            self.train()
            optimizer.zero_grad()

        logits, emb, _ = self.forward(
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

        if train_now:
            loss.backward()
            optimizer.step()

        return loss, y, probs, emb

    def pass_flow_windows(
        self, flow_generator, n_windows=None, optimizer=None, train=True, **kwargs
    ):
        """pass flow graphs from flow generator, with kwargs for pass_flowgraph using batch gradient descent (*NOT SGD)
        Returns (Tensors): Average loss, hstacked losses per windows, y and output probabilities
        """
        if train:
            self.train()
            optimizer.zero_grad()

        window_losses, ys, probs = [], [], []
        for train_window in tqdm(flow_generator, total=n_windows):
            G, _ = graph_encode(
                train_window,
                edge_cols=["src", "dst"],
                linegraph=False,
                target_col="Attack",
            )
            loss, y, prob, _ = self.pass_flowgraph(
                G=G, train_now=False, optimizer=optimizer, **kwargs
            )
            window_losses.append(loss)
            ys.append(y)
            probs.append(prob)
            # acc gradients, dont update untill afterwards
            if train:
                loss.backward()

            del G

        if train:
            optimizer.step()

        av_loss = torch.hstack(window_losses).mean()
        return [av_loss] + [torch.hstack(m) for m in (window_losses, ys, probs)]

    def train_flows(
        self,
        train_flow_generator,
        test_flow_generator,
        criterion,
        optimizer,
        epochs,
        experiment_summary: dict,
        experimental_directory: Path,
        n_train_windows=None,
    ):
        """Self explanatory"""
        writer = SummaryWriter(log_dir=experimental_directory)

        # ----------------------------------------------------------------
        # ----------------------- TRAINING LOOP --------------------------
        test_iterators = itertools.tee(test_flow_generator, epochs)
        train_iterators = itertools.tee(train_flow_generator, epochs)

        for epc, test_flow_iter, train_flow_iter in zip(
            range(1, epochs - 1), test_iterators, train_iterators
        ):
            # ----- TRAIN -----
            logger.info("training...")
            self.train()
            av_loss, window_losses, y, probs = self.pass_flow_windows(
                train_flow_iter,
                criterion=criterion,
                optimizer=optimizer,
                train=True,
                n_windows=n_train_windows,
            )

            train_pr_auc, train_roc_auc, train_f1, prec, rec = write_metrics(
                y, probs, writer, epc, av_loss, train_category=True
            )
            writer.add_scalar(f"PosRate/Train/MeanProb", torch.mean(probs), epc)
            writer.add_histogram(f"Probs/Train/probs_hist", probs, epc)

            # ---- TEST -----
            logger.info("testing...")
            self.eval()
            with torch.no_grad():
                av_test_loss, test_window_losses, test_y, test_probs = (
                    self.pass_flow_windows(
                        flow_generator=test_flow_iter,
                        criterion=criterion,
                        optimizer=None,
                        train=False,
                    )
                )
                pr_auc, roc_auc, f1, prec, rec = write_metrics(
                    test_y, test_probs, writer, epc, av_test_loss, train_category=False
                )
                writer.add_scalar(f"PosRate/Test/MeanProb", torch.mean(test_probs), epc)
                writer.add_histogram(f"Probs/Test/probs_hist", test_probs, epc)

            logger.info(
                f"Epoch {epc:02d} \n"
                f"Train Av Window Loss: {av_loss:.4f} \n"
                f"Train ROC AUC: {train_roc_auc:.4f} \n"
                f"Train PR AUC: {train_pr_auc:.4f} \n"
                f"Train F1: {train_f1:.4f} \n"
                "--\n"
                f"Test Av Window loss: {av_test_loss:.4f} \n"
                f"Test ROC AUC: {roc_auc:.4f} \n"
                f"Test PR AUC: {pr_auc:.4f} \n"
                f"Test F1: {f1:.4f} \n"
            )

            # also try saving ws pickle (weight issue with custom model?)
            with open(experimental_directory / "current_model.pkl", "wb") as f:
                pickle.dump(self, f)

            # Write Metadata
            logger.info(f"saving to experimental directory: {experimental_directory}")
            with open(experimental_directory / "experiment.pkl", "wb") as f:
                pickle.dump(experiment_summary, f)


class EdgeSAGELayer(nn.Module):
    """Produces new node embeddings based off Aggregated edge features"""

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
    """Produces new node embeddings based off aggregated node neighbourhood"""

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
    """Edge aggregation for node embdeddings, followed by node neighbourhood aggregation"""

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
