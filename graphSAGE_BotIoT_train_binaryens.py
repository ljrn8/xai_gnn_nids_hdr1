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

WINDOW = 2_500
LR = 1e-3
EPOCHS = 30

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

def forward(flows, model, criterion, train=True):
    """ train/eval model on windowed flows 
    """
    losses, y_trues, y_preds = [], [], []
    for i, G in enumerate(yield_subgraphs(flows, window=WINDOW)):
        logger.debug(f'train windows completed procssing: {i}')
        logger.debug(f'Window G{i} edge index shape {G.edge_index.shape} x shape {G.x.shape}')
        optimizer.zero_grad()
        y = G.x[:, -1].long() 

        # fucking 2 showed up at here once, for no reason, make it a fucking 1.
        y = (y != 0).long()
        x = G.x[:, :-1]

        logger.debug(f'y train shape = {y.shape}')
        logger.debug(f'y train unique = {np.unique(y, return_counts=True)}')

        out = model(x, G.edge_index)
        loss = criterion(out, y)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        y_trues.append(y)
        y_preds.append( torch.argmax(out, dim=1))
        del G

    y_trues = torch.cat(y_trues).numpy()
    y_preds = torch.cat(y_preds).numpy()
    avg_loss = np.mean(losses)

    logger.info(f'train y_trues.shape, sum {y_trues.shape}, {y_trues.sum()} | y_preds.shape, sum {y_preds.shape},{y_preds.sum()}', c='green')
    f1, prec, rec = get_metrcs(y_trues, y_preds)
    return f1, prec, rec, avg_loss


# ------------- script -------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logger.info("Loading data...", c='blue')
train_flows = pd.read_csv("interm/BotIoT_v1_processed_test.csv")
test_flows = pd.read_csv("interm/BotIoT_v1_processed_test.csv")
classes = list(np.unique(train_flows.Attack))
train_flows = train_flows[train_flows.Attack.isin(classes)]
test_flows = test_flows[test_flows.Attack.isin(classes)]

flows = pd.concat([train_flows, test_flows], ignore_index=True)
logger.info("Loaded", c='blue')

le = LabelEncoder()
le.fit(flows.Attack)

exp_dir = Path(f"interm/runs/binaryens_botiot_graphsage_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)
logger.add(exp_dir / 'run.log')
writer = SummaryWriter(log_dir=exp_dir)

logger.info(f'class distribution: {np.unique(train_flows.Attack, return_counts=True)}')
for a in classes:
    if a == "Benign":
        continue

    logger.info(f"Training binary classifier for {a}...", c='blue')

    # binary classification for each class
    train_cp = train_flows.copy()
    test_cp = test_flows.copy()
    train_cp.Attack = (train_cp.Attack == a).astype(int)
    test_cp.Attack = (test_cp.Attack == a).astype(int)

    logger.debug(f"number of {a} flows: {sum(train_cp.Attack != 0)}")
    logger.debug(f"number of Benign flows: {sum(train_cp.Attack == 0)}")

    model_kwargs = {
        "in_channels": len(train_flows.columns) - 3,
        "hidden_channels": 256,
        "out_channels": 2,
        "num_layers": 2, 
    }

    model = GraphSAGE(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # weight classes
    logger.debug(np.unique(train_cp.Attack, return_counts=True))
    class_counts = np.bincount(train_cp.Attack)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    logger.debug(f"Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_test_loss = float("inf")

    # ------------------ Training Loop ------------------

    for epc in range(1, EPOCHS + 1):
        logger.info(f' --- epoch: {epc}/{EPOCHS+1}', c='blue')

        # Train
        model.train()
        f1, prec, rec, avg_loss = forward(train_cp, model, criterion, train=True)
        writer.add_scalar(f"F1/Train/{a}_f1", f1, epc)
        writer.add_scalar(f"REC/Train/{a}_rec", rec, epc)
        writer.add_scalar(f"PREC/Train/{a}_prec", prec, epc)
        writer.add_scalar(f"Loss/Train/{a}_loss", avg_loss, epc)

        # Test
        model.eval()
        with torch.no_grad():
            f1, prec, rec, test_avg_loss = forward(
                train_cp, model, criterion, train=False)
        
        writer.add_scalar(f"F1/Test/{a}_f1", f1, epc)
        writer.add_scalar(f"REC/Test/{a}_rec", rec, epc)
        writer.add_scalar(f"PREC/Test/{a}_prec", prec, epc)
        writer.add_scalar(f"Loss/Test/{a}_loss", test_avg_loss, epc)

        logger.info(
            f"{a} Epoch {epc:02d} | "
            f"{a} Train Loss: {avg_loss:.4f} | "
            f"{a} Test Acc: {test_avg_loss:.4f} | "
        , c='green')

        # Save best model only
        if test_avg_loss < best_test_loss:
            best_test_loss = test_avg_loss
            torch.save(model.state_dict(), exp_dir / f"best_model_{a}.pt")

    # ------------------ Metadata ------------------

    experiment_summary = {
        "description": f"{a} Binary Ensemble GraphSAGE with linegraph on Bot IoT v1",
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
