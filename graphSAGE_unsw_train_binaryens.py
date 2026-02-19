import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, pickle
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from ML_utils import yield_subgraphs, log, debug

WINDOW = 5000
LR = 1e-3
EPOCHS = 30

# ------------------ Setup ------------------

device = "cpu"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log("Loading data...", c="blue")
train_flows = pd.read_csv("interm/unsw_nb15_processed_train.csv")
test_flows = pd.read_csv("interm/unsw_nb15_processed_test.csv")
classes = list(np.unique(train_flows.Attack))
train_flows = train_flows[train_flows.Attack.isin(classes)]
test_flows = test_flows[test_flows.Attack.isin(classes)]

flows = pd.concat([train_flows, test_flows], ignore_index=True)
log("Loaded", c="blue")

le = LabelEncoder()
le.fit(flows.Attack)

exp_dir = Path(f"interm/runs/binaryens_unsw_graphsage_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=exp_dir)

# ------------------- Ensemble -------------------

for a in classes:
    if a == "Benign" or a == "Analysis":  # !! skipping analysis, too few samples
        continue

    log(f"Training binary classifier for {a}...", c="blue")

    # binary classification for each class
    train_cp = train_flows.copy()
    test_cp = test_flows.copy()
    train_cp.Attack = (train_cp.Attack == a).astype(int)
    test_cp.Attack = (test_cp.Attack == a).astype(int)

    # only sample reasonable attack:benign ration from flows dataframe
    ben_flows = train_cp[train_cp.Attack == 0]
    mal_flows = train_cp[train_cp.Attack == 1]
    sample_i = np.random.choice(
        range(len(ben_flows)), size=200_000, replace=False  # !! 10% sampling
    )
    train_cp = pd.concat([ben_flows.iloc[sample_i], mal_flows])

    debug(f"number of {a} flows: {sum(train_cp.Attack != 0)}")
    debug(f"number of Benign flows: {sum(train_cp.Attack == 0)}")

    model_kwargs = {
        "in_channels": 72,
        "hidden_channels": 256,
        "out_channels": 2,
        "num_layers": 2,
    }

    model = GraphSAGE(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # weight classes
    debug(np.unique(train_cp.Attack, return_counts=True))
    class_counts = np.bincount(train_cp.Attack)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    debug(f"Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_test_loss = float("inf")

    # ------------------ Training Loop ------------------

    for epc in range(1, EPOCHS + 1):
        log(f" --- epoch: {epc}/{EPOCHS+1}", c="blue")

        # ---- Train ----
        model.train()
        train_losses, train_accs = [], []
        y_preds, y_trues = [], []

        for i, G in enumerate(yield_subgraphs(train_cp, window=WINDOW)):
            debug(f"train windows completed procssing: {i}")
            debug(
                f"Window G{i} edge index shape {G.edge_index.shape} x shape {G.x.shape}"
            )

            optimizer.zero_grad()
            y_train = G.x[:, -1].long()
            x_train = G.x[:, :-1]
            out = model(x_train, G.edge_index)

            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(out, dim=1)
            acc = (pred == y_train).float().mean().item()

            train_accs.append(acc)
            train_losses.append(loss.item())
            y_trues.append(y_train.cpu())
            y_preds.append(pred.cpu())

        # log class wise f1
        y_trues = torch.cat(y_trues).numpy()
        y_preds = torch.cat(y_preds).numpy()
        log(
            f"train y_trues.shape, sum {y_trues.shape}, {y_trues.sum()} | y_preds.shape, sum {y_preds.shape},{y_preds.sum()}",
            c="green",
        )
        class_f1 = f1_score(y_trues, y_preds, average="binary")
        writer.add_scalar(f"F1/Train/{a}_f1", class_f1, epc)

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)

        # -------- Test ---------
        model.eval()
        test_losses, test_accs = [], []
        y_trues, y_preds = [], []
        with torch.no_grad():
            for G in yield_subgraphs(test_cp, window=WINDOW):
                y_test = G.x[:, -1].long()
                x_test = G.x[:, :-1]
                out = model(x_test, G.edge_index)
                loss = criterion(out, y_test)

                pred = torch.argmax(out, dim=1)
                acc = (pred == y_test).float().mean().item()
                test_accs.append(acc)
                test_losses.append(loss.item())
                y_trues.append(y_test.cpu())
                y_preds.append(pred.cpu())

        # log class wise f1
        y_trues = torch.cat(y_trues).numpy()
        y_preds = torch.cat(y_preds).numpy()
        class_f1 = f1_score(y_trues, y_preds, average="binary")
        writer.add_scalar(f"F1/Test/{a}_f1", class_f1, epc)

        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accs)

        # TensorBoard Logging
        writer.add_scalar(f"Loss/{a}_Train", avg_train_loss, epc)
        writer.add_scalar(f"Loss/{a}_Test", avg_test_loss, epc)
        writer.add_scalar(f"Accuracy/{a}_Train", avg_train_acc, epc)
        writer.add_scalar(f"Accuracy/{a}_Test", avg_test_acc, epc)

        log(
            f"{a} Epoch {epc:02d} | "
            f"{a} Train Loss: {avg_train_loss:.4f} | "
            f"{a} Test Loss: {avg_test_loss:.4f} | "
            f"{a} Train Acc: {avg_train_acc:.4f} | "
            f"{a} Test Acc: {avg_test_acc:.4f}",
            c="green",
        )

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), exp_dir / f"best_model_{a}.pt")

    # ------------------ Metadata ------------------

    experiment_summary = {
        "description": f"{a} Binary ensemble GraphSAGE with linegraph on UNSW-NB15",
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
