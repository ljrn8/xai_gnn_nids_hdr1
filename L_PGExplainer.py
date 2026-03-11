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
from ML_utils import graph_encode
import argparse
import matplotlib.pyplot as plt
from varname import nameof


class L_PGExplainer(nn.Module):
    """
    PGExplainer adapted for link prediction on network flows (edge mask)
    """

    MODEL_PRED_THRESHOLD = 0.5
    IMPIRICAL_SAMPLING = False
    TAU = 0.2

    def __init__(
        self,
        model: EGraphSAGE,
        model_embedding_features: int,
        edge_attr=None,
        edge_entr_reg=0.1,
        edge_sum_reg=0.1,
        mlp_lasso_reg=0.005,
        hidden_parameters=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.mlp = nn.Sequential(
            # model embed features x2 for node emb concatenations to produce edge embs
            nn.Linear(
                in_features=model_embedding_features * 2, out_features=hidden_parameters
            ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_parameters, out_features=1),
        )
        self.edge_attr = edge_attr
        self.edge_sum_reg = edge_sum_reg
        self.edge_entr_reg = edge_entr_reg
        self.mlp_lasso_reg = mlp_lasso_reg

    def elementwise_entropy(x):
        x = x.clamp(1e-6, 1 - 1e-6)
        return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

    def regularization(self, edge_mask):
        edge_mask = edge_mask.clamp(1e-6, 1 - 1e-6)
        entropy = -edge_mask * torch.log2(edge_mask) - (1 - edge_mask) * torch.log2(
            1 - edge_mask
        )
        edge_entr_reg = self.edge_entr_reg * entropy.mean()
        edge_mean_reg = self.edge_sum_reg * edge_mask.mean()
        return edge_entr_reg, edge_mean_reg

    def tau_sigmoid_subgraph_prediction(self, G, mask, tau):
        """applies tau sigmoid binarization instead of noise sampling"""
        binarized_mask = torch.sigmoid(mask / tau)
        masked_y_pred, _ = self.model.forward(
            edge_attr=G.edge_attr, edge_index=G.edge_index, edge_weight=binarized_mask
        )
        return masked_y_pred

    def _sample_BCD_mask(self, mask_logits, tau):
        mask_sigmoid = torch.sigmoid(mask_logits).clamp(1e-6, 1 - 1e-6)
        epsilon = torch.rand(1)[0].clamp(1e-6, 1 - 1e-6)
        logit = (
            torch.log2(epsilon)
            - torch.log2(1 - epsilon)
            + torch.log2(mask_sigmoid)
            - torch.log2(1 - mask_sigmoid)
        ) / tau
        # logit = logit.clamp(-10, 10)  # prevent sigmoid overflow to nan
        return torch.sigmoid(logit)

    def approximate_subgraph_BCD(self, G, mask_logits, samples=100):
        """sample BCD subgraph predictions from the fractional mask with MC estimation"""
        predictions = []
        for _ in range(samples):
            near_binary_mask = self._sample_BCD_mask(mask_logits, self.TAU)
            masked_y_pred, _ = self.model.forward(
                edge_attr=G.edge_attr,
                edge_index=G.edge_index,
                edge_weight=near_binary_mask,
            )
            predictions.append(torch.sigmoid(masked_y_pred))

        if samples > 1:
            all_preds = torch.vstack(predictions)
            masked_y_pred_mean = torch.mean(all_preds, axis=0)
            assert len(masked_y_pred_mean) == len(predictions[0])
        else:
            masked_y_pred_mean = predictions[0]

        return masked_y_pred_mean

    def approximate_subgraph_imipirically(self, G, edge_mask, samples=30):
        """Sample using impirical marginal distribution"""
        predictions = []
        for _ in range(samples):

            # sample imipirical Z
            assert self.edge_attr is not None
            n_edges, n_features = self.edge_attr.shape
            N = self.edge_attr.shape[0]
            idx = torch.randint(0, N, (n_edges, n_features))
            Z = self.edge_attr[idx, torch.arange(n_features).unsqueeze(0)]
            X = G.edge_attr
            y_pred, _ = self.model(
                edge_attr=Z + ((X - Z) * edge_mask.unsqueeze(1)),
                edge_index=G.edge_index,
            )
            predictions.append(torch.sigmoid(y_pred))

        # average prediction
        return torch.mean(torch.vstack(predictions), axis=0)

    def lasso_reg(self):
        params = torch.cat([p.view(-1) for p in self.model.parameters()])
        return params.abs().sum() / params.numel()

    def fit(self, G, epochs, lr=0.01, loss_f=torch.nn.BCELoss(), verbose=False):
        """train the MLP"""

        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # normal prediction
        with torch.no_grad():
            y_pred, embeddings = self.model.forward(
                edge_attr=G.edge_attr,
                edge_index=G.edge_index,
            )
            y_pred = torch.sigmoid(y_pred)
            malicious_edge_mask = y_pred > self.MODEL_PRED_THRESHOLD
            # clamp to prevent exploding BCE, as output is very confident
            embeddings = embeddings.detach()

        # train MLP on window
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        losses, mask_regularization = [], []

        logger.info("training..")
        for epc in tqdm(range(1, 1 + epochs)):
            mask_logits = self.mlp(embeddings).squeeze()
            mask = torch.sigmoid(mask_logits)

            y_pred_masked = self.approximate_subgraph_BCD(G, mask_logits)

            # only compute loss on malicious edges
            loss = loss_f(
                y_pred.clamp(1e-6, 1 - 1e-6)[malicious_edge_mask],
                y_pred_masked.clamp(1e-6, 1 - 1e-6)[malicious_edge_mask],
            )

            # regularization and update
            edge_entr_reg, edge_mean_reg = self.regularization(mask)
            mlp_l1_reg = self.mlp_lasso_reg * self.lasso_reg()

            total_loss = loss + edge_entr_reg + edge_mean_reg + mlp_l1_reg
            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.detach())
            mask_regularization.append([edge_entr_reg, edge_mean_reg, mlp_l1_reg])

            # epoch report
            logger.info(
                f"epoch: {epc} \n"
                + f"loss for mask only: \t {loss.detach():.5f} \n "
                + f"mask average value: \t {torch.mean(mask):.5f} \n "
                + f"regularization penalties: \n\t mlp reg: {mlp_l1_reg:.5f}, \n\t edge entropy reg: {edge_entr_reg:.5f}, \n\t edge mean reg: {edge_mean_reg:.5f} | "
            )

            if verbose:
                bins, bin_edges = np.histogram(y_pred.detach().numpy(), bins=10)
                hist = {edge: val for edge, val in zip(bin_edges[1:], bins)}
                print(
                    f"---- MASKED prediction stats\n mean:{torch.mean(y_pred):.5f} \nhistogram: {hist}"
                )
                plt.hist(y_pred_masked.detach().numpy(), bins=500)
                plt.show()

                bins, bin_edges = np.histogram(y_pred_masked.detach().numpy(), bins=10)
                hist = {edge: val for edge, val in zip(bin_edges[1:], bins)}
                print(
                    f"---- NORMAL prediction stats\n mean:{torch.mean(y_pred_masked):.5f} \nhistogram: {hist}"
                )
                plt.hist(y_pred.detach().numpy(), bins=500)
                plt.show()

        return (mask, losses, mask_regularization)


def main(args):
    device = "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_f = Path(args.test_flows_csv)
    model_dir = Path(args.model)
    logger.info("loading data")
    test_flows = pd.read_csv(test_f)

    logger.info("loading model ..")
    with open(model_dir, "rb") as f:
        model = pickle.load(f)

    model.to(device)

    # convert test_flows Attack to binary
    test_flows["Attack"] = torch.Tensor(
        (test_flows["Attack"] != "Benign").astype(float).values
    ).float()

    metrics_output_dir = Path(f"interm/xAI/GNNE_{timestamp}")
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("encoding graph")
    G, _ = graph_encode(
        test_flows, edge_cols=["src", "dst"], target_col="Attack", linegraph=False
    )

    explainer = L_PGExplainer(
        model=model,
        model_embedding_features=256,
        hidden_parameters=args.parameters,
        edge_entr_reg=args.edge_entropy_reg,
        edge_sum_reg=args.edge_sum_reg,
        mlp_lasso_reg=args.lasso_reg,
        edge_attr=G.edge_attr,
    )

    experimental_output = {
        "model": model,
        "explainer": explainer,
        "meta": {
            "args": args,
            "model_dir": model_dir,
            "test_f": test_f,
            "metrics_output_dir": metrics_output_dir,
        },
        "description": "PGExplainer for line prediction NIDS",
    }

    logger.info("learning explanation")
    mask, losses, mask_regularization = explainer.fit(
        G, epochs=args.epochs, lr=args.learning_rate
    )
    experimental_output["results"] = {
        "mask": mask,
        "losses": losses,
        "mask_regularization": mask_regularization,
    }

    logger.info(f"writing experiment output to {metrics_output_dir}")
    with open(metrics_output_dir / f"experiment.pkl", "wb") as f:
        pickle.dump(experimental_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=10, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.005, type=float)
    parser.add_argument("-eer", "--edge-entropy-reg", default=0.1, type=float)
    parser.add_argument("-esr", "--edge-sum-reg", default=0.2, type=float)
    parser.add_argument("-l", "--lasso-reg", default=0.05, type=float)
    parser.add_argument(
        "-p", "--parameters", help="number of parameters in MLP", default=64, type=int
    )
    parser.add_argument(
        "-tf",
        "--test-flows-csv",
        default="interm/unsw_nb15_processed_test.csv",
        help="location of processsed test flow dataset ( numeric only), requiring src, dst and Attack columns",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="interm/runs/EGraphSAGE_anomdetection_UNSW_AD_no_windows20260311_000357/best_model.pkl",
        help="location of pickled binary EGraphSAGE model",
    )

    main(parser.parse_args())
