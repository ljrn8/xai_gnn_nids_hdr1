from ML_utils import graph_encode, yield_subgraphs, fidelities
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
        edge_mask_entropy_reg=0.05,
        edge_mask_mean_reg=0.05,
        empirical_samples=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_bank = feature_bank
        self.edge_mask_entropy_reg = edge_mask_entropy_reg
        self.edge_mask_mean_reg = edge_mask_mean_reg
        self.empirical_samples = empirical_samples

    def elementwise_entropy(x):
        return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

    def regularization(self, edge_mask):
        edge_entr_reg = self.edge_mask_entropy_reg * torch.mean(
            torch.tensor([GNNExplainer.elementwise_entropy(e) for e in edge_mask])
        )
        edge_mean_reg = self.edge_mask_mean_reg * torch.mean(edge_mask)
        return edge_entr_reg, edge_mean_reg

    def sample_from_empirical(self, edge_attr):
        n_edges, n_features = edge_attr.shape
        N = self.feature_bank.shape[0]
        idx = torch.randint(0, N, (n_edges, n_features))
        Z = self.feature_bank[idx, torch.arange(n_features).unsqueeze(0)]
        return Z  # (n_edges, n_features)

    def fit(
        self,
        model,
        G,
        epochs,
        output_directory,
        lr=0.01,
        loss_f=None,
        experiment_info={},
    ):

        logger.info(f"n# of nodes in graph: {G.edge_index.max().item() + 1}")
        logger.info(f"n# of edges in graph: {G.edge_index.shape[1]}")

        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # normal prediction
        with torch.no_grad():
            y_pred_logits, edge_embeddings, node_embeddings = self.model.forward(
                edge_attr=G.edge_attr,
                edge_index=G.edge_index,
            )
            y_pred = torch.sigmoid(y_pred_logits)
            positive_predictions = y_pred > self.MODEL_PRED_THRESHOLD

        # predicted pos weighted BCEloss for MI
        if loss_f is None:
            mal_weight = (positive_predictions == 0).sum() / (
                positive_predictions == 1
            ).sum()
            logger.info(f"using weight: {mal_weight}")
            loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([mal_weight]))

        # initialize edge mask with requires_grad
        W = torch.randn((G.edge_attr.shape[0]), requires_grad=True, device=device)

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        losses, edge_entr_regs, edge_mean_regs = [], [], []
        best_loss = float("inf")

        logger.info("training..")
        for epc in range(1, 1 + epochs):

            mask = torch.sigmoid(W)
            masked_y_pred_mean_logits = self.masked_forward(model, G, mask)
            loss = loss_f(masked_y_pred_mean_logits, y_pred)

            # regularization that controls sparsity and confidence
            edge_entr_reg, edge_mean_reg = self.regularization(mask)
            total_loss = loss + edge_entr_reg + edge_mean_reg

            logger.info("backprop..")
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # logging
            y_pred_masked = torch.sigmoid(masked_y_pred_mean_logits)
            logger.info(
                f"epoch: {epc} | "
                f"mal predictions of normal graph: {(y_pred > 0.5).float().sum():.5f} \n"
                + f"average predictions of normal graph: {(y_pred).mean():.5f} \n"
                + f"mal predictions of masked graph: {(y_pred_masked > 0.5).float().sum():.5f} \n"
                + f"average prediction for masked graph: {y_pred_masked.mean():.5f} \n"
                f"av loss for mask: {np.mean(losses):.5f} | "
                f"edge mask average value: {torch.mean(mask):.5f} | "
                f"regularization penalty for entropy: {(edge_entr_reg):.5f} | "
                f"regularization penalty for mean: {(edge_mean_reg):.5f} | "
            )

            logger.info(f"writing current and best masks to {output_directory}")
            losses.append(loss.detach())
            edge_entr_regs.append(edge_entr_reg.detach())
            edge_mean_regs.append(edge_mean_reg.detach())
            run = {
                "edge_mask": mask,
                "losses": losses,
                "y_pred_masked": y_pred_masked,
                "y_pred": y_pred,
                "info": experiment_info,
                "regularization": {
                    "edge_entropy_reg": edge_entr_regs,
                    "edge_mean_reg": edge_mean_regs,
                },
            }

            # write the current and best masks
            with open(output_directory / "current_mask.pkl", "wb") as f:
                pickle.dump(run, f)

            if total_loss < best_loss:
                best_loss = total_loss
                with open(output_directory / "best_mask.pkl", "wb") as f:
                    pickle.dump(run, f)

        return (
            mask,
            losses,
            {"edge_entropy_reg": edge_entr_regs, "edge_mean_reg": edge_mean_regs},
        )

    def masked_forward(self, model, G, edge_mask):
        """Logits for masked graph through empirical marginalization of noise features (Z)"""

        # sample multiple Z's for monte carlo estimation
        mask_y_candidates = []
        for _ in range(self.empirical_samples):
            # impirical marginal distribution sampling for noise features (Z)
            Z = torch.FloatTensor(
                np.array(self.sample_from_empirical(G.edge_attr), dtype=float)
            )

            # reparameterization trick
            masked_edge_attr = Z + ((G.edge_attr - Z) * edge_mask.unsqueeze(1))

            # keep grad on masked prediction for loss backprop
            masked_y_pred, _, _ = model.forward(
                masked_edge_attr,
                G.edge_index,
            )

            mask_y_candidates.append(masked_y_pred)

        if self.empirical_samples > 1:
            all_preds = torch.vstack(mask_y_candidates)
            masked_y_pred_mean = torch.mean(all_preds, axis=0)
            assert len(masked_y_pred_mean) == len(mask_y_candidates[0])

        else:
            masked_y_pred_mean = mask_y_candidates[0]

        return masked_y_pred_mean


def main(args):

    # open EgraphSAGE run
    run_dir = Path(args.run_dir)
    with open(run_dir / "experiment.pkl", "rb") as f:
        run_info = pickle.load(f)

    # load test data from metadata
    logger.info("loading data")
    test_f = Path(run_info["test_df_location"])
    test_flows = pd.read_csv(test_f)

    # downsample of prototyping
    downsample = args.prototype_downsample_rate
    if downsample is not None:
        logger.warning("downsampling flows will ruin temporal continguency")
        l = len(test_flows)
        test_flows = test_flows.sample(frac=downsample, random_state=0)

    # convert test_flows Attack to binary
    test_flows["Attack"] = torch.Tensor(
        (test_flows["Attack"] != "Benign").astype(float).values
    ).float()

    # load best model from run dir
    logger.info("loading model ..")
    model_dir = run_dir / "current_model.pkl"
    with open(model_dir, "rb") as f:
        model = pickle.load(f)
    model.to(device)

    # setup up explaination directory
    explaination_dir = Path(args.explaination_dir)
    if args.add_timestamp_subfolder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        explaination_dir = explaination_dir / timestamp
    explaination_dir.mkdir(parents=True, exist_ok=True)

    logger.info("encoding graph")
    G, _ = graph_encode(
        test_flows, edge_cols=["src", "dst"], target_col="Attack", linegraph=False
    )
    explainer = GNNExplainer(
        model=model,
        feature_bank=G.edge_attr.cpu().detach().numpy(),
        edge_mask_entropy_reg=args.edge_mask_entropy_reg,
        edge_mask_mean_reg=args.edge_mask_mean_reg,
        empirical_samples=args.empirical_samples,
    )

    experimental_output = {
        "model": model,
        "mask_type": "edge",
        "explainer": explainer,
        "args": args,
        "model_dir": model_dir,
        "test_f": test_f,
        "explaination_dir": explaination_dir,
        "description": "PGExplainer for link prediction NIDS, using an edge mask",
    }

    logger.info("learning explaination")
    mask, losses, mask_regularization = explainer.fit(
        G,
        output_directory=explaination_dir,
        epochs=args.epochs,
        lr=args.learning_rate,
        experiment_info=experimental_output,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--edge-entropy-reg", default=0.05, type=float)
    parser.add_argument("--edge-sum-reg", default=0.05, type=float)
    parser.add_argument("--empirical-samples", default=10, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prototype-downsample-rate", default=None, type=float)
    parser.add_argument(
        "-p", "--parameters", help="number of parameters in MLP", default=64, type=int
    )
    parser.add_argument(
        "--run-dir",
    )
    parser.add_argument("--explaination-dir")
    parser.add_argument(
        "--add-timestamp-subfolder",
        action="store_true",
        help="Add a timestamp subfolder to the explanation directory",
    )
    main(parser.parse_args())
