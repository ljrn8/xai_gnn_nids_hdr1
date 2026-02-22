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

WINDOW = 5000
LR = 0.01
EPOCHS = 50

os.environ["LOGURU_LEVEL"] = "INFO"
device = "cpu"


def get_metrcs(y_true, y_pred_probs):
    y_pred = y_pred_probs > 0.5
    P, R = (
        precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1),
    )
    return (roc_auc_score(y_true, y_pred_probs), 2 * P * R / (P + R), P, R)  # f1


def forward(flows, model, criterion, train=True):
    """train/eval model on windowed flows"""
    if train:
        optimizer.zero_grad()

    losses, y_trues, y_probs = [], [], []
    for i, G in enumerate(yield_subgraphs(flows, window=WINDOW)):
        y = G.y.to(device).reshape(-1)  # explicit label tensor
        x = G.x.to(device)
        out = model((x, G.edge_index.to(device)))
        loss = criterion(out, y)

        # fucking 2 showed up at here once, for no reason, make it a fucking 1.
        y = (y != 0).long()

        if i % 20 == 0:
            logger.debug(
                f"window G{i} edge index shape {G.edge_index.shape} x shape {G.x.shape}"
            )
            logger.debug(f"y train shape = {y.shape}")
            logger.debug(f"y train unique = {np.unique(y, return_counts=True)}")

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        y_trues.append(y)
        y_probs.append(torch.softmax(out, dim=1)[:, 1].detach().cpu())
        del G

    y_trues = torch.cat(y_trues).cpu().numpy()
    y_probs = torch.cat(y_probs).cpu().numpy()
    avg_loss = np.mean(losses)

    logger.info(
        f"train y_trues.shape, sum {y_trues.shape}, {y_trues.sum()} | y_preds.shape, sum {y_probs.shape},{y_probs.sum()}",
        c="green",
    )
    return avg_loss, y_trues, y_probs


# ------------- script -------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logger.info("Loading data...", c="blue")
train_flows = pd.read_csv("interm/BotIoT_v1_processed_train.csv")
test_flows = pd.read_csv("interm/BotIoT_v1_processed_test.csv")
classes = list(np.unique(train_flows.Attack))
train_flows = train_flows[train_flows.Attack.isin(classes)]
test_flows = test_flows[test_flows.Attack.isin(classes)]
flows = pd.concat([train_flows, test_flows], ignore_index=True)
logger.info("Loaded", c="blue")

logger.info("class distr " + str(np.unique(flows.Attack, return_counts=True)))
logger.info("test class distr " + str(np.unique(test_flows.Attack, return_counts=True)))
logger.info(
    "train class distr " + str(np.unique(train_flows.Attack, return_counts=True))
)

le = LabelEncoder()
le.fit(flows.Attack)

exp_dir = Path(f"interm/runs/binaryens_botiot_graphsage_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)
log_dir = exp_dir / "run.log"
log_dir.touch()
logger.add(log_dir)
writer = SummaryWriter(log_dir=exp_dir)

logger.info(f"class distribution: {np.unique(train_flows.Attack, return_counts=True)}")
for a in classes:
    if a == "Benign":
        continue

    logger.info(f"Training binary classifier for {a}...", c="blue")

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
    from sklearn.utils.class_weight import compute_class_weight

    classes_arr = np.array([0, 1])
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes_arr, y=train_cp.Attack
    )
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.debug(f"Class weights computed: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_test_loss = float("inf")

    # ------------------ Training Loop ------------------

    for epc in range(1, EPOCHS + 1):
        logger.info(f" --- epoch: {epc}/{EPOCHS+1}", c="blue")

        # --------- Train
        model.train()
        avg_loss, trues, probs = forward(train_cp, model, criterion, train=True)
        auc, f1, prec, rec = get_metrcs(trues, probs)
        writer.add_scalar(f"F1/Train/{a}_f1", f1, epc)
        writer.add_scalar(f"AUC/Train/{a}_auc", auc, epc)
        writer.add_scalar(f"REC/Train/{a}_rec", rec, epc)
        writer.add_scalar(f"PREC/Train/{a}_prec", prec, epc)
        writer.add_scalar(f"Loss/Train/{a}_loss", avg_loss, epc)
        writer.add_scalar(
            f"PosRate/Train/{a}_PosRate", np.sum(np.mean(probs > 0.5)), epc
        )
        writer.add_histogram(f"Probs/Train/{a}_probs_hist", probs, epc)

        # --------- Test
        model.eval()
        with torch.no_grad():
            test_avg_loss, trues, probs = forward(
                test_cp, model, criterion, train=False
            )
            auc, f1, prec, rec = get_metrcs(trues, probs)

        writer.add_scalar(f"F1/Test/{a}_f1", f1, epc)
        writer.add_scalar(f"AUC/Test/{a}_auc", auc, epc)
        writer.add_scalar(f"REC/Test/{a}_rec", rec, epc)
        writer.add_scalar(f"PREC/Test/{a}_prec", prec, epc)
        writer.add_scalar(f"Loss/Test/{a}_loss", test_avg_loss, epc)
        writer.add_scalar(
            f"PosRate/Test/{a}_PosRate", np.sum(np.mean(probs > 0.5)), epc
        )
        writer.add_histogram(f"Probs/Test/{a}_probs_hist", probs, epc)

        logger.info(
            f"{a} Epoch {epc:02d} | "
            f"{a} Train Loss: {avg_loss:.4f} | "
            f"{a} Test loss: {test_avg_loss:.4f} | "
        )

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
