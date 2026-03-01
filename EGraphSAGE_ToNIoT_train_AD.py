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
from ML_utils import yield_subgraphs
from loguru import logger
from EGraphSAGE import EGraphSAGE
import sys

### potential issues
# too little layers? too many layers?
# binary anomoly encoding or df -> tensor will not be writable 
# perfect fitting training set, poor for test set

WINDOW = 1_000
LR = 0.01
EPOCHS = 10
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"EGraphSAGE_anomdetection_ToNIoT_graphsage_{timestamp}"

log_level = sys.argv[1] if len(sys.argv) > 1 else 'INFO'
logger.remove() 
logger.add(sys.stderr, level=log_level)
device = "cpu"

def get_metrics(y_true, y_pred_probs):
    y_pred = y_pred_probs > 0.5
    P, R = (
        precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1),
    )
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs, pos_label=1)
    pr_auc = auc(recall, precision)
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return (pr_auc, roc_auc_score(y_true, y_pred_probs), F1, P, R)


def write_metrics(y_trues, y_probs, writer, epc, train_category: bool):
    all_metrics = get_metrics(y_trues, y_probs)
    pr_auc, roc_auc, f1, prec, rec = all_metrics
    metrics = {"PR-AUC": pr_auc, "ROC-AUC": roc_auc, "F1": f1, "PREC": prec, "rec": rec}
    istrain = "TRAIN" if train_category else "TEST"
    for metric_name, metric in metrics.items():
        writer.add_scalar(f"{metric_name}/{istrain}/{metric_name.lower()}", metric, epc)
    return all_metrics


# load processed flows
logger.info("Loading data...", c="blue")
train_flows = pd.read_csv("interm/ToNIoT_processed_train.csv")
test_flows = pd.read_csv("interm/ToNIoT_processed_test.csv")
classes = list(np.unique(train_flows.Attack))
flows = pd.concat([train_flows, test_flows], ignore_index=True)
logger.info("Loaded", c="blue")

# FIT label encoder for future use
le = LabelEncoder()
le.fit(flows.Attack)

# experimental directory
exp_dir = Path(f"interm/runs/{RUN_ID}")
exp_dir.mkdir(parents=True, exist_ok=True)
log_dir = exp_dir / "run.log"
log_dir.touch()
logger.add(log_dir)
writer = SummaryWriter(log_dir=exp_dir)

logger.info(
    f'train mal:ben = {sum(train_flows.Attack == "Benign")}:{sum(train_flows.Attack != "Benign")}')
logger.info(
    f"TRAIN class distribution: {np.unique(train_flows.Attack, return_counts=True)}")
logger.info(
    f"TEST class distribution: {np.unique(test_flows.Attack, return_counts=True)}")
features = list(flows.columns)
print(features)
[features.remove(s) for s in ["src", "dst", "Attack"]]
logger.info(f"features: [{len(features)}] {features}")

model_kwargs = {
    "hidden_channels": [128, 128],
    "num_features": len(features),
}
model = EGraphSAGE(**model_kwargs)
logger.info(model)

# re encode attack for anomoly detection
train_attack_classes, test_attack_classes = train_flows.Attack, test_flows.Attack
train_flows.Attack = torch.Tensor(
    (train_flows["Attack"] != "Benign").astype(float).values
).float()
test_flows.Attack = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()
logger.info(e
    f"TRAIN: ben: {len(train_flows.Attack ) - sum(train_flows.Attack)}, mal sum {sum(train_flows.Attack )}, mal perc: {np.mean(train_flows.Attack )}"
)
logger.info(
    f"TEST: ben: {len(test_flows.Attack) - sum(test_flows.Attack)}, mal sum {sum(test_flows.Attack)}, mal perc: {np.mean(test_flows.Attack)}"
)
# w = float((train_flows.Attack == 0).sum() / train_flows.Attack.sum())
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_test_auc = 0.0

# ----------------------- TRAINING LOOP --------------------------

for epc in range(1, EPOCHS - 1):

    # ----- TRAIN -----
    model.train()
    avg_loss, y_trues, y_probs, embeddings = model.train_flows(
        train_flows, criterion=criterion, optimizer=optimizer, window=WINDOW, train=True
    )
    pr_auc, roc_auc, f1, prec, rec = write_metrics(
        y_trues, y_probs, writer, epc, train_category=True
    )
    writer.add_scalar(f"PosRate/Train/MeanProb", np.mean(y_probs), epc)
    writer.add_histogram(f"Probs/Train/probs_hist", y_probs, epc)

    # ---- TEST -----
    model.eval()
    with torch.no_grad():
        test_avg_loss, y_trues, y_probs, embeddings = model.train_flows(
            test_flows, criterion=criterion, optimizer=None, window=WINDOW, train=False
        )
        pr_auc, roc_auc, f1, prec, rec = write_metrics(
            y_trues, y_probs, writer, epc, train_category=False
        )
        writer.add_scalar(f"PosRate/Test/MeanProb", np.mean(y_probs), epc)
        writer.add_histogram(f"Probs/Test/probs_hist", y_probs, epc)

    logger.info(
        f"Epoch {epc:02d} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Test loss: {test_avg_loss:.4f} | "
        f"Test PR AUC: {pr_auc:.4f} | "
        f"Test ROC AUC: {roc_auc:.4f} | "
        f"mean y_prob: {y_probs.mean()} | \n"
        f"y_trues 1={(y_trues).sum()} 0={(1-y_trues).sum()} | "
        f"y_probs >0.5={(y_probs>0.5).sum()} <0.5={(y_probs<0.5).sum()}"
    )

    del embeddings, y_trues, y_probs

    # Save best model only (using TEST PR AUC)
    if pr_auc > best_test_auc:
        best_test_auc = pr_auc
        with open(exp_dir / "best_model.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(exp_dir / "current_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # ------------------ Metadata ------------------
    experiment_summary = {
        "description": f"EgraphSAGE binary anomoyl detection (no ensembling) on Bot IoT v1",
        "epochs": EPOCHS,
        "model_kwargs": model_kwargs,
        "window_size": WINDOW,
        "le": le,
        "best_test_auc": best_test_auc,
        "label_encoder_classes": list(classes),
    }

    with open(exp_dir / "experiment.pkl", "wb") as f:
        pickle.dump(experiment_summary, f)

writer.close()
