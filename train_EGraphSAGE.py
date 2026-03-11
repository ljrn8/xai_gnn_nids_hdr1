import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, auc
import torch
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, pickle
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from ML_utils import yield_subgraphs, graph_encode
from loguru import logger
from EGraphSAGE import EGraphSAGE


def get_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor):
    y_pred = y_pred_probs > 0.5
    P, R = (
        precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1),
    )
    precision, recall, _ = precision_recall_curve(
        y_true, y_pred_probs.detach().numpy(), pos_label=1
    )
    pr_auc = auc(recall, precision)
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return (pr_auc, roc_auc_score(y_true, y_pred_probs.detach().numpy()), F1, P, R)


def write_metrics(
    y_trues: torch.Tensor, y_probs: torch.Tensor, writer, epc, train_category: bool
):
    all_metrics = get_metrics(y_trues, y_probs)
    pr_auc, roc_auc, f1, prec, rec = all_metrics
    metrics = {"PR-AUC": pr_auc, "ROC-AUC": roc_auc, "F1": f1, "PREC": prec, "rec": rec}
    istrain = "TRAIN" if train_category else "TEST"
    for metric_name, metric in metrics.items():
        writer.add_scalar(f"{metric_name}/{istrain}/{metric_name.lower()}", metric, epc)
    return all_metrics


def validate_flow_dataframe(flows):
    for required_col in ["src", "dst", "Attack"]:
        assert required_col in flows.columns, "missing required col"


# ------------- script -------------

# parse arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--layer-size", default=256, type=int)
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--train-flows", default="interm/unsw_nb15_processed_train.csv")
parser.add_argument("--test-flows", default="interm/unsw_nb15_processed_test.csv")
parser.add_argument("--device", default="cpu")
parser.add_argument("--run-directory", default="interm/runs")
args = parser.parse_args()
logger.info(f"using args: {args}")
run_ID = f"EGraphSAGE_AD_{Path(args.train_flows).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
channels = [args.layer_size] * args.num_layers

# load processed flows
logger.info(f"Loading data from {args.train_flows}, {args.test_flows}...", c="blue")
train_flows = pd.read_csv(args.train_flows)
test_flows = pd.read_csv(args.test_flows)
classes = list(np.unique(train_flows.Attack))
flows = pd.concat([train_flows, test_flows], ignore_index=True)
logger.info("Loaded", c="blue")

# expected conditions for flows
validate_flow_dataframe(flows)

# fit label encoder for future use
le = LabelEncoder()
le.fit(flows.Attack)

# experimental directory
exp_dir = Path(args.run_directory) / run_ID
exp_dir.mkdir(parents=True, exist_ok=True)
log_dir = exp_dir / "run.log"
log_dir.touch()
logger.add(log_dir)
writer = SummaryWriter(log_dir=exp_dir)


# convert train_flows to 10:1 benign:attack ratio
def resample(flows, benign_class="Benign", ratio=10):
    ben_flows = flows[flows.Attack == benign_class]
    mal_flows = flows[flows.Attack != benign_class]
    ben_index = np.arange(len(ben_flows))
    sample_index = np.random.choice(
        ben_index, replace=False, size=len(mal_flows) * ratio
    )
    resampled_flows = pd.concat((mal_flows, ben_flows.iloc[sample_index]))
    return resampled_flows.sort_values(by="FLOW_START_MILLISECONDS")


train_flows = resample(train_flows, benign_class="Benign", ratio=10)
test_flows = resample(test_flows, benign_class="Benign", ratio=10)

logger.warning("train set resampled resulting in n flows: {}".format(len(train_flows)))
logger.warning(
    "train set new class distribution: {}".format(
        np.unique(train_flows.Attack, return_counts=True)
    )
)
logger.warning("test set resampled resulting in n flows: {}".format(len(test_flows)))
logger.warning(
    "test set new class distribution: {}".format(
        np.unique(test_flows.Attack, return_counts=True)
    )
)

logger.info(
    f'train mal:ben = {sum(train_flows.Attack == "Benign")}:{sum(train_flows.Attack != "Benign")}'
)
logger.info(
    f'test mal:ben = {sum(test_flows.Attack == "Benign")}:{sum(test_flows.Attack != "Benign")}'
)

logger.info(f"class distribution: {np.unique(train_flows.Attack, return_counts=True)}")
features = list(flows.columns)
print(features)
[features.remove(s) for s in ["src", "dst", "Attack"]]

model_kwargs = {
    "layer_sizes": channels,
    "flow_features": len(features),
    "output_dim": 1,
}
model = EGraphSAGE(**model_kwargs)
logger.info(f"MODEL SUMMARY: ", model)
for layer in model.layers:
    logger.info(layer)

# re encode attack for anomoly detection
train_attack_classes, test_attack_classes = train_flows.Attack, test_flows.Attack
train_flows.Attack = torch.Tensor(
    (train_flows["Attack"] != "Benign").astype(float).values
).float()
test_flows.Attack = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()
logger.info(
    f"TRAIN: ben: {len(train_flows.Attack ) - sum(train_flows.Attack)}, mal sum {sum(train_flows.Attack )}, mal perc: {np.mean(train_flows.Attack )}"
)
logger.info(
    f"TEST: ben: {len(test_flows.Attack) - sum(test_flows.Attack)}, mal sum {sum(test_flows.Attack)}, mal perc: {np.mean(test_flows.Attack)}"
)

# 10:1 class imbalance, so use pos_weight in BCE loss
w = sum(1 - train_flows.Attack) / sum(train_flows.Attack)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best_test_auc = 0.0

logger.info("encoding graphs")
test_G, _ = graph_encode(
    test_flows, edge_cols=["src", "dst"], linegraph=False, target_col="Attack"
)
train_G, _ = graph_encode(
    train_flows, edge_cols=["src", "dst"], linegraph=False, target_col="Attack"
)

# ----------------------------------------------------------------
# ----------------------- TRAINING LOOP --------------------------
for epc in range(1, args.epochs - 1):

    # ----- TRAIN -----
    logger.info("training...")
    model.train()
    loss, y, probs, emb = model.pass_flowgraph(
        train_G, criterion=criterion, optimizer=optimizer, train=True, debug=True
    )
    train_pr_auc, train_roc_auc, train_f1, prec, rec = write_metrics(
        y, probs, writer, epc, train_category=True
    )
    writer.add_scalar(f"PosRate/Train/MeanProb", torch.mean(probs), epc)
    writer.add_histogram(f"Probs/Train/probs_hist", probs, epc)

    # ---- TEST -----
    logger.info("testing...")
    model.eval()
    with torch.no_grad():
        test_loss, test_y, test_probs, test_emb = model.pass_flowgraph(
            test_G, criterion=criterion, optimizer=optimizer, train=False, debug=True
        )
        pr_auc, roc_auc, f1, prec, rec = write_metrics(
            test_y, test_probs, writer, epc, train_category=False
        )
        writer.add_scalar(f"PosRate/Test/MeanProb", torch.mean(test_probs), epc)
        writer.add_histogram(f"Probs/Test/probs_hist", test_probs, epc)

    logger.info(
        f"Epoch {epc:02d} | "
        f"Train Loss: {loss:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Test PR AUC: {pr_auc:.4f} | "
        f"Train PR AUC: {train_pr_auc:.4f} | "
        f"Test ROC AUC: {roc_auc:.4f} | "
        f"Train ROC AUC: {train_roc_auc:.4f} | "
        f"Test F1: {f1:.4f} | "
        f"Train F1: {train_f1:.4f} | "
    )

    # save current model (warning: may overfit)
    torch.save(model.state_dict(), exp_dir / f"best_model.pt")

    # also try saving ws pickle (weight issue with custom model?)
    with open(exp_dir / "best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ------------------ Metadata ------------------
    experiment_summary = {
        "description": f"EgraphSAGE binary anomoly detection with args: {args}",
        "epochs": args.epochs,
        "model_kwargs": model_kwargs,
        "le": le,
        "best_test_auc": best_test_auc,
        "label_encoder_classes": list(classes),
        "lr": args.lr,
        "test_df_location": args.test_flows,
        "train_df_location": args.train_flows,
    }

    logger.info(f"saving to experimental directory: {exp_dir}")
    with open(exp_dir / "experiment.pkl", "wb") as f:
        pickle.dump(experiment_summary, f)


writer.close()
