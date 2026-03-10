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

class PGExplainer(nn.Module):
    """PGExplainer adapted for link prediction on network flows
    Current tenatic aspects of the impl include
    1. 1 MLP explainer for each graph window.
    """

    MODEL_PRED_THRESHOLD = 0.5
    IMPIRICAL_SAMPLING = False
    TAU = 0.5

    def __init__(
        self,
        model_embedding_features,
        full_edge_attr=None,
        edge_entr_reg=0.1,
        edge_sum_reg=0.1,
        mlp_lasso_reg=0.005,
        hidden_parameters=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            # model embed features x2 for node emb concatenations to produce edge embs
            nn.Linear(in_features=model_embedding_features*2, out_features=hidden_parameters),
            nn.ReLU(),
            nn.Linear(in_features=hidden_parameters, out_features=1),
        )
        self.l1_norm_weight = edge_sum_reg
        self.edge_reg_weight = edge_entr_reg
        self.mlp_lasso_reg = mlp_lasso_reg
        self.full_edge_attr = full_edge_attr

    def elementwise_entropy(x):
        x = x.clamp(1e-6, 1 - 1e-6)
        return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

    def regularization(self, edge_mask):
        edge_mask = edge_mask.clamp(1e-6, 1 - 1e-6)
        entropy = -edge_mask * torch.log2(edge_mask) - (1 - edge_mask) * torch.log2(1 - edge_mask)
        reg = self.edge_reg_weight * entropy.mean()
        reg += self.l1_norm_weight * edge_mask.mean()
        return reg

    def sample_from_empirical(self, edge_attr):
        """Randomly (impirically) sample from X_E for downstream MC estimation"""
        feature_bank = self.full_edge_attr
        n_edges, n_features = edge_attr.shape
        N = feature_bank.shape[0]
        idx = torch.randint(0, N, (n_edges, n_features))
        Z = feature_bank[idx, torch.arange(n_features).unsqueeze(0)]
        return Z  # (n_edges, n_features)

    def sample_masked_graph(self, G, mask, tau, impirical_reparamaterization=False):
        """
        Converts the soft edge mask to a binary concrete distribution,
        sampled with epsilon, with strength tau. This directly multiplied into the mask,
        though with impirical_reparaterization=True, its replaced with an impirical sample from X_E
        """
        epsilon = torch.rand(1)[0]
        logger.debug(f"epsilon: {epsilon}")
        near_binary_mask = torch.sigmoid(
            (torch.log2(epsilon) - torch.log2(1 - epsilon) + mask) / tau
        )

        if impirical_reparamaterization:
            Z = self.sample_from_empirical(G.edge_attr)
            masked_edge_attr = Z + ((G.edge_attr - Z) * near_binary_mask.unsqueeze(1))
        else:
            masked_edge_attr = G.edge_attr * near_binary_mask.unsqueeze(1)

        return masked_edge_attr

    def get_masked_prediction(self, model, G, mask):
        """
        Return a prediction with the mask applied through a BCD
        and potentially impirical sampling for the 'subgraph' of the same size.
        """
        masked_edge_attr = self.sample_masked_graph(
            G,
            mask,
            tau=self.TAU,
            impirical_reparamaterization=self.IMPIRICAL_SAMPLING,
        )
        masked_y_pred, _ = model.forward(
            masked_edge_attr,
            G.edge_index,
        )
        return masked_y_pred

    def appromiximate_subgraph_prediction(self, model, G, mask, samples=10):
        """sample subgraph predictions from the fractional mask with MC estimation"""
        predictions = []
        for _ in range(samples):
            predictions.append(
                torch.sigmoid(self.get_masked_prediction(model, G, mask))
            )

        if samples > 1:
            all_preds = torch.vstack(predictions)
            masked_y_pred_mean = torch.mean(all_preds, axis=0)
            assert len(masked_y_pred_mean) == len(predictions[0])
        else:
            masked_y_pred_mean = predictions[0]

        return masked_y_pred_mean

    def fit(self, model, G, epochs, lr=0.01, loss_f=torch.nn.BCELoss(), verbose=False):
        """train the MLP on the pygeomtric graph G"""

        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # normal prediction
        with torch.no_grad():
            y_pred, embeddings = model.forward(
                G.edge_attr,
                G.edge_index,
            )
            y_pred = torch.sigmoid(y_pred)
            embeddings = embeddings.detach()

        mal_prediction_mask = y_pred > self.MODEL_PRED_THRESHOLD

        # train MLP on window
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        losses, mask_regularization = [], []

        logger.info("training..")
        for epc in tqdm(range(1, 1 + epochs)):
            mask_logits = self.mlp(embeddings)
            l1_norm = sum(p.abs().sum() for p in self.mlp.parameters())
            mask = torch.sigmoid(mask_logits).squeeze()

            # masked prediction
            y_pred_masked = self.appromiximate_subgraph_prediction(model, G, mask)
            
            # only compute loss on malicious edges
            loss = loss_f(y_pred[mal_prediction_mask], y_pred_masked[mal_prediction_mask])

            # regularization and update
            reg = self.regularization(mask) + (self.mlp_lasso_reg * l1_norm)
            loss += reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())
            mask_regularization.append(reg.detach())

            logger.info(
                f"epoch: {epc} | "
                + f"total loss for mask: {loss:.5f} | "
                + f"mask average value: {torch.mean(mask):.5f} | "
                + f"regularization penalty: {reg:.5f} | "
            )
            if verbose:
                bins, bin_edges = np.histogram(y_pred.detach().numpy(), bins=10)
                hist = {edge: val for edge,val in zip(bin_edges[1:], bins)}
                print(f"---- MASKED prediction stats\n mean:{torch.mean(y_pred):.5f} \nhistogram: {hist}")
                plt.hist(y_pred_masked.detach().numpy(), bins=500)
                plt.show()
                
                bins, bin_edges  = np.histogram(y_pred_masked.detach().numpy(), bins=10)
                hist = {edge: val for edge,val in zip(bin_edges[1:], bins)}
                print(f"---- NORMAL prediction stats\n mean:{torch.mean(y_pred_masked):.5f} \nhistogram: {hist}")
                plt.hist(y_pred.detach().numpy(), bins=500)
                plt.show()
                
            
        return (mask, losses, mask_regularization)

def main(args, device = "cpu"):
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

    pge = PGExplainer(
        model_embedding_features=256,
        hidden_parameters=args.parameters,
        edge_entr_reg=args.edge_entropy_reg,
        edge_sum_reg=args.edge_sum_reg,
        full_edge_attr=G.edge_attr,
        mlp_lasso_reg=args.lasso_reg
    )

    experimental_output = {
        "meta": {
            "args": args,
            "model_dir": model_dir,
            "test_f": test_f,
            "metrics_output_dir": metrics_output_dir,
        },
        "description": "PGExplainer for line prediction NIDS",
    }

    logger.info('learning explanation')
    mask, losses, mask_regularization = pge.fit(
        model, G, epochs=args.epochs, lr=args.learning_rate
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
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-eer", "--edge-entropy-reg", default=0.01, type=float)
    parser.add_argument("-esr", "--edge-sum-reg", default=0.01, type=float)
    parser.add_argument("-l", "--lasso-reg", default=0.0005, type=float)
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
        default="interm/runs/EGraphSAGE_anomdetection_UNSW_AD_no_windows20260309_124202/best_model.pkl",
        help="location of pickled binary EGraphSAGE model",
    )

    main(parser.parse_args())
