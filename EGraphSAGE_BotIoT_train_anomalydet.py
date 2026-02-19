import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
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

WINDOW = 1_000
LR = 0.005
EPOCHS = 15

os.environ["LOGURU_LEVEL"] = "INFO" 
device = "cpu"

def get_metrcs(y_true, y_pred):
    P, R = (precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1))
    return (
         2 * P * R / (P + R), # f1
         P,
         R
    )

# ------------- script -------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# load processed flows
logger.info("Loading data...", c='blue')
train_flows = pd.read_csv("interm/BotIoT_v1_processed_test.csv")
test_flows = pd.read_csv("interm/BotIoT_v1_processed_test.csv")
classes = list(np.unique(train_flows.Attack))
flows = pd.concat([train_flows, test_flows], ignore_index=True)
logger.info("Loaded", c='blue')

# FIT label encoder for future use
le = LabelEncoder()
le.fit(flows.Attack)

# experimental directory
exp_dir = Path(f"interm/runs/EGraphSAGE_anomdetection_botiot_graphsage_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)
log_dir = (exp_dir / 'run.log')
log_dir.touch()
logger.add(log_dir)
writer = SummaryWriter(log_dir=exp_dir)

logger.info(f'class distribution: {np.unique(train_flows.Attack, return_counts=True)}')
features = list(flows.columns)
features.remove('src', 'dst', 'Attack')

model_kwargs = {
    'hidden_channels':[256, 256],
    'num_features':len(features)
}
model = EGraphSAGE(**model_kwargs)

# re encode attack for anomoly detection
train_attack_classes, test_attack_classes = train_flows.Attack, test_flows.Attack
train_flows.Attack = torch.Tensor(train_flows['Attack'] != 'Benign').float()
test_flows.Attack = torch.Tensor(test_flows['Attack'] != 'Benign').float()

logger.info(f'TRAIN: ben: {len(train_flows.Attack ) - sum(train_flows.Attack )}, mal sum {sum(train_flows.Attack )}, mal perc: {np.mean(train_flows.Attack )}')
logger.info(f'TEST: ben: {len(test_flows.Attack) - sum(test_flows.Attack)}, mal sum {sum(test_flows.Attack)}, mal perc: {np.mean(test_flows.Attack)}')
test_flows.Attack
criterion = torch.nn.BCELoss(weight=torch.Tensor(np.mean(train_flows.Attack)).float())
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_test_loss = float('inf')

for epc in range(1, EPOCHS-1):

    # ----- train
    avg_loss, y_trues, y_probs, embeddings = model.train_flows(
        train_flows,
        criterion=criterion,
        optimizer=optimizer,
        window=WINDOW,
        train=True
    )    
    auc, f1, prec, rec = get_metrcs(y_trues, y_probs)
    writer.add_scalar(f"F1/Train/f1", f1, epc)
    writer.add_scalar(f"AUC/Train/auc", auc, epc)
    writer.add_scalar(f"REC/Train/rec", rec, epc)
    writer.add_scalar(f"PREC/Train/prec", prec, epc)
    writer.add_scalar(f"Loss/Train/loss", avg_loss, epc)
    writer.add_scalar(
        f"PosRate/Train/PosRate", np.sum(np.mean(y_probs > 0.5)), epc
    )
    writer.add_histogram(f"Probs/Train/probs_hist", y_probs, epc)

    test_avg_loss, y_trues, y_probs, embeddings = model.train_flows(
        test_flows,
        criterion=criterion,
        optimizer=optimizer,
        window=WINDOW,
        train=False
    )
    auc, f1, prec, rec = get_metrcs(y_trues, y_probs)
    writer.add_scalar(f"F1/Test/f1", f1, epc)
    writer.add_scalar(f"AUC/Test/auc", auc, epc)
    writer.add_scalar(f"REC/Test/rec", rec, epc)
    writer.add_scalar(f"PREC/Test/prec", prec, epc)
    writer.add_scalar(f"Loss/Test/loss", test_avg_loss, epc)
    writer.add_scalar(
        f"PosRate/Test/PosRate", np.sum(np.mean(y_probs > 0.5)), epc
    )
    writer.add_histogram(f"Probs/Test/probs_hist", y_probs, epc)
    logger.info(
        f"Epoch {epc:02d} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Test loss: {test_avg_loss:.4f} | "
    )

    # Save best model only
    if test_avg_loss < best_test_loss:
        best_test_loss = test_avg_loss
        torch.save(model.state_dict(), exp_dir / f"best_model.pt")

# ------------------ Metadata ------------------

experiment_summary = {
    "description": f"EgraphSAGE binary anomoyl detection (no ensembling) on Bot IoT v1",
    "epochs": EPOCHS,
    "model_kwargs": model_kwargs,
    "window_size": WINDOW,
    "le": le,
    "best_test_loss": best_test_loss,
    "label_encoder_classes": list(classes),
}

with open(exp_dir / "experiment.pkl", "wb") as f:
    pickle.dump(experiment_summary, f)

writer.close()
