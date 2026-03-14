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
from ML_utils import most_recent_object


class N_PGExplainer(nn.Module):
    """
    PGExplainer adapted for link prediction on network flows (node mask)
    """

    MODEL_PRED_THRESHOLD = 0.5
    IMPIRICAL_SAMPLING = False
    TAU = 1.0
    BCD_BIAS = 1e-6

    def __init__(
        self,
        model: EGraphSAGE,
        model_embedding_features: int,
        node_mask_entropy_reg=0.1,
        node_mask_sum_reg=0.1,
        mlp_lasso_reg=0.005,
        hidden_parameters=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model: EGraphSAGE = model
        self.mlp = nn.Sequential(
            # model embed features x2 for node emb concatenations to produce edge embs
            nn.Linear(
                in_features=model_embedding_features, out_features=hidden_parameters
            ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_parameters, out_features=1),
        )
        self.node_mask_sum_reg = node_mask_sum_reg
        self.node_mask_entropy_reg = node_mask_entropy_reg
        self.mlp_lasso_reg = mlp_lasso_reg

    def elementwise_entropy(x):
        x = x.clamp(1e-6, 1 - 1e-6)
        return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

    def regularization(self, node_mask):
        node_mask = node_mask.clamp(1e-6, 1 - 1e-6)
        entropy = -node_mask * torch.log2(node_mask) - (1 - node_mask) * torch.log2(
            1 - node_mask
        )
        entropy_reg = self.node_mask_entropy_reg * entropy.mean()
        mean_reg = self.node_mask_sum_reg * node_mask.mean()
        return entropy_reg, mean_reg

    def tau_sigmoid_subgraph_prediction(self, G, mask, tau):
        """applies tau sigmoid binarization instead of noise sampling"""
        binarized_mask = torch.sigmoid(mask / tau)
        masked_y_pred_logits, _, _ = self.model.forward(
            edge_attr=G.edge_attr, edge_index=G.edge_index, node_weight=binarized_mask
        )
        return masked_y_pred_logits

    def _sample_BCD_mask(self, node_mask_logits, tau):
        mask_sigmoid = torch.sigmoid(node_mask_logits).clamp(1e-6, 1 - 1e-6)
        epsilon = torch.rand(1)[0].clamp(1e-6, 1 - 1e-6)
        logit = (
            torch.log2(epsilon)
            - torch.log2(1 - epsilon)
            + torch.log2(mask_sigmoid)
            - torch.log2(1 - mask_sigmoid)
        ) / tau
        # logit = logit.clamp(-10, 10)  # prevent sigmoid overflow to nan
        return torch.sigmoid(logit)

    def _sample_BCD_direct_weight(self, node_mask_logits, tau):
        """Apply BCD as from https://github.com/flyingdoog/PGExplainer/blob/master/codes/Explainer.py"""
        epsilon = torch.rand(1)[0].clamp(self.BCD_BIAS, 1 - self.BCD_BIAS)
        logit = (torch.log2(epsilon) - torch.log2(1 - epsilon) + node_mask_logits) / tau
        # logit = logit.clamp(-10, 10)  # prevent sigmoid overflow to nan
        return torch.sigmoid(logit)

    def approximate_subgraph_BCD(self, G, node_mask_logits, samples=30):
        """sample BCD subgraph predictions from the fractional mask with MC estimation"""
        predictions = []
        for _ in range(samples):
            near_binary_mask = self._sample_BCD_mask(node_mask_logits, self.TAU)
            masked_y_pred, edge_embs, node_embs = self.model.forward(
                edge_attr=G.edge_attr,
                edge_index=G.edge_index,
                node_weight=near_binary_mask,
            )
            predictions.append(torch.sigmoid(masked_y_pred))

        if samples > 1:
            all_preds = torch.vstack(predictions)
            masked_y_pred_mean = torch.mean(all_preds, axis=0)
            assert len(masked_y_pred_mean) == len(predictions[0])
        else:
            masked_y_pred_mean = predictions[0]

        return masked_y_pred_mean

    def approximate_subgraph_BCD_with_prior_masking(
        self, G, node_mask_logits, samples=30
    ):
        """returns average logits
        uses G.x * mask once, using max(node mask) for the edge mask, not masking the message passing layers at all
        """
        predictions = []
        near_binary_masks = []
        for _ in range(samples):
            near_binary_mask = self._sample_BCD_direct_weight(
                node_mask_logits, self.TAU
            )
            near_binary_masks.append(near_binary_mask)
            src, dst = G.edge_index
            # only disable edges not connected to any 'on' nodes
            corresponding_edge_mask = torch.max(
                near_binary_mask[src], near_binary_mask[dst]
            )
            masked_y_pred, edge_embs, node_embs = self.model.forward(
                edge_attr=G.edge_attr * corresponding_edge_mask.unsqueeze(1),
                edge_index=G.edge_index,
            )
            predictions.append(masked_y_pred)

        if samples > 1:
            all_preds = torch.vstack(predictions)
            masked_y_pred_mean_logits = torch.mean(all_preds, axis=0)
            assert len(masked_y_pred_mean_logits) == len(predictions[0])
        else:
            masked_y_pred_mean_logits = predictions[0]

        return masked_y_pred_mean_logits, torch.mean(torch.vstack(near_binary_masks))

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
        params = torch.cat([p.view(-1) for p in self.mlp.parameters()])
        return params.abs().sum() / params.numel()

    def fit(self, G, epochs, lr=0.01, loss_f=None, verbose=False, tb_logging_dir=None):
        """train the MLP"""

        logger.info(f"n# of nodes in graph: {G.edge_index.max().item() + 1}")

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
            malicious_edge_mask = y_pred > self.MODEL_PRED_THRESHOLD
            node_embeddings = node_embeddings.detach()

        # predicted pos weighted BCEloss for MI
        if loss_f is None:
            n_edges = G.edge_attr.shape[0]
            n_mal_edges = malicious_edge_mask.float().sum()
            mal_weight = n_mal_edges / n_edges
            ben_weight = 1 - mal_weight
            sample_weights = torch.full([G.edge_attr.shape[0]], ben_weight)
            sample_weights[malicious_edge_mask] = mal_weight
            # !! using weight
            loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(mal_weight))

        # train MLP on window
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        losses, mask_regularization = [], []

        # ------------------------------------------------------------------------
        # ------------------------- MAIN EXPLANATION LEARNING --------------------

        logger.info("training..")
        for epc in tqdm(range(1, 1 + epochs)):
            mask_logits = self.mlp(node_embeddings).squeeze()

            masked_y_pred_mean_logits, average_near_binary_mask = (
                self.approximate_subgraph_BCD_with_prior_masking(G, mask_logits)
            )
            y_pred_masked = torch.sigmoid(masked_y_pred_mean_logits)

            loss = loss_f(masked_y_pred_mean_logits, y_pred)

            # check if gradient flows back to mask
            loss.backward(retain_graph=True)
            logger.info(f"mask_logits grad: {mask_logits.grad}")
            logger.info(f"mlp 0.weight grad norm: {self.mlp[0].weight.grad.norm():.8f}")

            # TB logging for  gradients (need have this early)
            if tb_logging_dir:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(log_dir=tb_logging_dir)
                # writer.add_scalar('masked_y_pred_grad', masked_y_pred_mean_logits.grad, epc)
                # writer.add_scalar('mask_logit_grad', mask_logits.grad, epc)
                writer.add_scalar(
                    f"all/mlp_first_layer_weight_grad",
                    self.mlp[0].weight.grad.norm(),
                    epc,
                )
                writer.add_scalar(
                    f"all/mlp_second_layer_weight_grad",
                    self.mlp[2].weight.grad.norm(),
                    epc,
                )

            optimizer.zero_grad()  # clear before the real backward

            mask = torch.sigmoid(mask_logits)

            # regularization and update
            entr_reg, mean_reg = self.regularization(mask)
            mlp_l1_reg = self.mlp_lasso_reg * self.lasso_reg()

            total_loss = loss + entr_reg + mean_reg + mlp_l1_reg
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.append(loss.detach())
            mask_regularization.append([entr_reg, mean_reg, mlp_l1_reg])

            # epoch report
            logger.info(
                f"epoch: {epc} \n"
                + f"mal predictions of normal graph: {(y_pred > 0.5).float().sum():.5f} \n"
                + f"average predictions of normal graph: {(y_pred).mean():.5f} \n"
                + f"mal predictions of masked graph: {(y_pred_masked > 0.5).float().sum():.5f} \n"
                + f"average prediction for masked graph: {y_pred_masked.mean():.5f} \n"
                + f"loss for node mask only: \t {loss.detach():.5f} \n "
                + f"mask average value: \t {torch.mean(mask):.5f} \n "
                + f"regularization penalties: \n\t mlp reg: {mlp_l1_reg:.5f}, \n\t edge entropy reg: {entr_reg:.5f}, \n\t edge mean reg: {mean_reg:.5f} | "
            )

            # rest of tensorboard logging
            if tb_logging_dir:

                # scalars
                writer.add_scalar(
                    f"all/Mean_Masked_Mal_Preds",
                    (y_pred_masked > 0.5).float().sum(),
                    epc,
                )
                writer.add_scalar(
                    f"all/Mean_preds_mean_value", torch.mean(y_pred_masked), epc
                )
                writer.add_scalar(f"all/BCE_loss", loss.detach().item(), epc)
                writer.add_scalar(f"all/mask_average_value", torch.mean(mask), epc)
                writer.add_scalar(f"all/mlp_lasso_reg_after_coeff", mlp_l1_reg, epc)
                writer.add_scalar(f"all/edge_entropy_reg_after_coeff", entr_reg, epc)
                writer.add_scalar(f"all/edge_mean_reg_after_coeff", mean_reg, epc)
                writer.add_scalar(
                    f"all/Mean_value_for_BCD_near_binary_mask",
                    torch.mean(average_near_binary_mask),
                    epc,
                )

                # histograms
                writer.add_histogram(f"Mean_masked_prediction", y_pred_masked, epc)
                writer.add_histogram(f"mask", mask, epc)
                writer.add_histogram(f"mask_logits", mask_logits, epc)
                writer.add_histogram(
                    f"Average_BCD_near_binary_mask", average_near_binary_mask, epc
                )

                # embeddings
                writer.add_embedding(edge_embeddings)
                writer.add_embedding(node_embeddings)

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
    logger.info(f"running with args: {args}")
    device = args.device
    run_dir = Path(args.run_directory)

    with open(run_dir / "experiment.pkl", "rb") as f:
        run_info = pickle.load(f)

    # load test data from metadata
    logger.info("loading data")
    test_f = Path(run_info["test_df_location"])
    test_flows = pd.read_csv(test_f)

    # convert test_flows Attack to binary
    test_flows["Attack"] = torch.Tensor(
        (test_flows["Attack"] != "Benign").astype(float).values
    ).float()

    # load best model from run dir
    logger.info("loading model ..")
    model_dir = run_dir / "best_model.pkl"
    with open(model_dir, "rb") as f:
        model = pickle.load(f)

    model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_output_dir = Path(args.output_directory) / timestamp
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("encoding graph")
    G, _ = graph_encode(
        test_flows, edge_cols=["src", "dst"], target_col="Attack", linegraph=False
    )

    explainer = N_PGExplainer(
        model=model,
        model_embedding_features=256,
        hidden_parameters=args.parameters,
        node_mask_entropy_reg=args.node_entropy_reg,
        node_mask_sum_reg=args.node_sum_reg,
        mlp_lasso_reg=args.lasso_reg,
    )

    experimental_output = {
        "model": model,
        "explainer": explainer,
        "args": args,
        "model_dir": model_dir,
        "test_f": test_f,
        "metrics_output_dir": metrics_output_dir,
        "description": "PGExplainer for link prediction NIDS, using a node mask",
    }

    logger.info("learning explanation")
    mask, losses, mask_regularization = explainer.fit(
        G, epochs=args.epochs, lr=args.learning_rate, tb_logging_dir=metrics_output_dir
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
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-ner", "--node-entropy-reg", default=0.05, type=float)
    parser.add_argument("-nsr", "--node-sum-reg", default=0.005, type=float)
    parser.add_argument("-l", "--lasso-reg", default=0.005, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "-p", "--parameters", help="number of parameters in MLP", default=64, type=int
    )
    parser.add_argument(
        "--run-directory",
        default=most_recent_object("interm/runs"),
    )
    parser.add_argument(
        "--output-directory",
        default="interm/xAI",
    )
    main(parser.parse_args())


# run it again
