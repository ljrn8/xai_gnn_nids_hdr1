from ML_utils import yield_subgraphs, fidelities
import pandas as pd
from torch import nn, threshold
import torch
from tqdm import tqdm
import sys
import pickle
from loguru import logger
from pathlib import Path
from EGraphSAGE import EGraphSAGE
import numpy as np
from pprint import pprint
from datetime import datetime

device = "cpu"


class GNNExplainer(nn.Module):
    """GNNexplainer adapted for link prediction on network flows"""

    def __init__(
        self,
        feature_bank,
        feature_reg_weight=0.1,
        edge_reg_weight=0.1,
        l1_norm_weight=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_bank = feature_bank
        self.feature_reg_weight = feature_reg_weight
        self.edge_reg_weight = edge_reg_weight
        self.l1_norm_weight = l1_norm_weight

    def elementwise_entropy(x):
        return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

    def regularization(self, edge_mask, feature_mask):
        reg = 0
        reg += self.edge_reg_weight * torch.mean(
            torch.tensor([GNNExplainer.elementwise_entropy(e) for e in edge_mask])
        )
        reg += self.feature_reg_weight * torch.mean(
            torch.tensor([GNNExplainer.elementwise_entropy(e) for e in feature_mask])
        )
        reg += self.l1_norm_weight * (
            torch.mean(edge_mask) + torch.mean(feature_mask) / 2
        )
        return reg

    def sample_from_empirical(self, edge_attr):
        n_edges, n_features = edge_attr.shape
        N = self.feature_bank.shape[0]
        idx = torch.randint(0, N, (n_edges, n_features))
        Z = self.feature_bank[idx, torch.arange(n_features).unsqueeze(0)]
        return Z  # (n_edges, n_features)

    def fit(
        self, model, test_flows, epochs, window, lr=0.01, loss_f=torch.nn.BCELoss()
    ):
        """Iterator that trains feature and edge masks on each window graph in test_flows,
        evaluating sparsity fidelities sequentially and those yielding results for each window.
        """

        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        for i, G in enumerate(yield_subgraphs(test_flows, window, linegraph=False)):
            num_features = len(test_flows.columns) - 3  # src dst Attack
            num_nodes = G.edge_index.max().item() + 1

            # initialize edge mas k with requires_grad
            feature_W = torch.randn((num_features), requires_grad=True, device=device)
            edge_W = torch.randn(
                (G.edge_attr.shape[0]), requires_grad=True, device=device
            )
            optimizer = torch.optim.Adam([edge_W, feature_W], lr=lr)
            losses, regularization_penalties = [], []

            for epc in range(1, 1 + epochs):
                edge_mask = torch.sigmoid(edge_W)
                feature_mask = torch.sigmoid(feature_W)

                y_pred, masked_y_pred = self.masked_forward(
                    model, G, edge_mask, feature_mask
                )
                loss = loss_f(y_pred, masked_y_pred)

                # regularization that controls sparsity and confidence
                reg = self.regularization(edge_mask, feature_mask)
                total_loss = loss + reg

                total_loss.backward()
                optimizer.step()

                losses.append(loss.detach())
                regularization_penalties.append(reg.detach())

                logger.info(
                    f"epoch: {epc} | "
                    f"av loss for mask: {np.mean(losses):.5f} | "
                    f"edge mask average value: {torch.mean(edge_mask):.5f} | "
                    f"feature mask average value: {torch.mean(feature_mask):.5f} | "
                    f"regularization penalty: {reg:.5f} | "
                )

            losses_out = torch.stack(losses).cpu().numpy()
            regularization_penalties_out = (
                torch.stack(regularization_penalties).cpu().numpy()
            )
            yield (G, edge_W, feature_W, losses_out, regularization_penalties_out)

    def masked_forward(
        self, model, G, edge_mask, feature_mask, thresh=None, sample_Z=30
    ):
        """Returns the original and masked predictions, applying"""

        # use hard masks if thresh is provided, otherwise use soft masks
        if thresh:
            edge_mask = torch.BoolTensor(edge_mask > thresh)
            feature_mask = torch.BoolTensor(feature_mask > thresh)

        # normal prediction
        with torch.no_grad():
            n_nodes = G.edge_index.max().item() + 1
            y_pred, _ = model.forward(
                G.edge_attr,
                G.edge_index,
                node_attr=torch.ones(size=(n_nodes, feature_mask.shape[0])).to(device),
            )

        # sample multiple Z's for monte carlo estimation
        mask_y_candidates = []
        for _ in range(sample_Z):
            # impirical marginal distribution sampling for noise features (Z)
            Z = torch.FloatTensor(
                np.array(self.sample_from_empirical(G.edge_attr), dtype=float)
            )

            # reparameterization trick
            feature_masked_edge_attr = Z + ((G.edge_attr - Z) * feature_mask)
            fully_masked_edge_attr = Z + (
                (feature_masked_edge_attr - Z) * edge_mask.unsqueeze(1)
            )

            # keep grad on masked prediction for loss backprop
            masked_y_pred, _ = model.forward(
                fully_masked_edge_attr,
                G.edge_index,
                node_attr=torch.ones(size=(n_nodes, feature_mask.shape[0])).to(device),
            )

            mask_y_candidates.append(masked_y_pred)

        if sample_Z > 1:
            all_preds = torch.vstack(mask_y_candidates)
            masked_y_pred_mean = torch.mean(all_preds, axis=0)
            assert len(masked_y_pred_mean) == len(mask_y_candidates[0])

        else:
            masked_y_pred_mean = mask_y_candidates[0]

        return torch.sigmoid(y_pred), torch.sigmoid(masked_y_pred_mean)

    def evaluate_window(
        self,
        model,
        G,
        edge_mask,
        feature_mask,
        prediction_threshhold=0.5,
        sparsities=np.arange(0, 0.4, 0.02),
    ):
        metrics = {}
        for s in tqdm(sparsities, desc=f"Evaluating masks at spasities"):
            threshold = np.percentile(edge_mask.cpu().detach().numpy(), s * 100)

            y_pred, masked_y_pred = self.masked_forward(
                model, G, edge_mask, feature_mask, thresh=threshold
            )

            binary_masked_y_pred = (masked_y_pred > prediction_threshhold).float()
            binary_y_pred = (y_pred > prediction_threshhold).float()
            fp, fm = fidelities(
                y_pred=binary_y_pred,
                y_mask=binary_masked_y_pred,
                y_imask=1 - binary_masked_y_pred,
                y=G.y,
            )

            metrics[s] = {"fp": fp, "fn": fm, "threshold": threshold}

        return metrics


def main(model, test_flows, epochs, window, lr):
    gnne = GNNExplainer(
        feature_bank=test_flows[
            test_flows.columns.difference(["src", "dst", "Attack"])
        ].values,  # for sampling noise
        edge_reg_weight=0.1,
        feature_reg_weight=0.1,
        l1_norm_weight=0.1,
    )

    # convert test_flows Attack to binary
    test_flows["Attack"] = torch.Tensor(
        (test_flows["Attack"] != "Benign").astype(float).values
    ).float()

    metrics_output_dir = Path(f"interm/xAI/GNNE_{timestamp}")

    explainability_report = {
        "epochs": epochs,
        "window": window,
        "learning_rate": lr,
        "description": "GNNExplainer with featurewise and edge wise reparmaterization, sampled once each on different Z",
        "metrics_per_window": [],
        "window_number": [],
        "average_mask_loss_curves": [],
        "average_reg_loss_curves": [],
        "mal_flows_in_window": [],
    }
    for i, (window_G, edge_W, feature_W, losses, regularization_penalties) in enumerate(
        gnne.fit(
            model=model,
            test_flows=test_flows,
            epochs=epochs,
            window=window,
            lr=lr,
            loss_f=torch.nn.BCELoss(),
        )
    ):
        logger.info(
            f"LEARNED WINDOW: {i+1} | " f"av loss for mask: {np.mean(losses):.5f} | "
        )

        # only evaluate when windows contain atleast 1 anomaly, otherwise metrics are not meaningful
        if sum(window_G.y == 1) > 100:
            edge_mask = torch.sigmoid(edge_W)
            feature_mask = torch.sigmoid(feature_W)
            metrics = gnne.evaluate_window(model, window_G, edge_mask, feature_mask)

            explainability_report["metrics_per_window"].append(metrics)
            explainability_report["window_number"].append(i)
            explainability_report["average_mask_loss_curves"].append(losses)
            explainability_report["average_reg_loss_curves"].append(
                regularization_penalties
            )
            explainability_report["mal_flows_in_window"].append(sum(window_G.y == 1))

            logger.debug(f"writing metrics for window {i+1}: {metrics}")
            metrics_output_dir.mkdir(parents=True, exist_ok=True)

            with open(metrics_output_dir / f"experiment.pkl", "wb") as f:
                pickle.dump(explainability_report, f)
            logger.debug("finished writing metrics")


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_f = Path("interm/unsw_nb15_processed_test.csv")
    model_dir = Path(
        "interm/runs/EGraphSAGE_anomdetection_UNSW_graphsage_20260303_033347/best_model.pkl"
    )
    logger.info("loading data")
    test_flows = pd.read_csv(test_f)

    logger.info("loading model ..")
    with open(model_dir, "rb") as f:
        model = pickle.load(f)

    model.to(device)

    main(model, test_flows, epochs=200, window=10_000, lr=0.05)
