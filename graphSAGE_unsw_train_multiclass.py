import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from preprocessing import graph_encode
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, pickle
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from preprocessing import yield_subgraphs

WINDOW = 5000
LR = 0.0003
EPOCHS = 50

# ------------------ Setup ------------------

device = 'cpu'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

exp_dir = Path(f"interm/runs/unsw_graphsage_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=exp_dir)

print('Loading data...')
train_flows = pd.read_csv("interm/unsw_nb15_processed_train.csv")
test_flows = pd.read_csv("interm/unsw_nb15_processed_test.csv")
classes= ['Benign', 'Exploits', 'DoS', 'Fuzzers']
train_flows = train_flows[train_flows.Attack.isin(classes)]
test_flows = test_flows[test_flows.Attack.isin(classes)]

flows = pd.concat([train_flows, test_flows], ignore_index=True)
print('Loaded')

le = LabelEncoder()
le.fit(flows.Attack)

model_kwargs = {
    'in_channels': 72,
    'hidden_channels': 256,
    'out_channels': len(classes),
    'num_layers': 2,
}

model = GraphSAGE(**model_kwargs).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# criterion = torch.nn.CrossEntropyLoss()

# weight classes
class_counts = np.bincount(le.transform(train_flows.Attack))
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
weights = torch.tensor(
    class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {dict(zip(le.classes_, class_weights))}")
criterion = torch.nn.CrossEntropyLoss(weight=weights)

best_test_loss = float("inf")

# ------------------ Training Loop ------------------

for epc in range(1, EPOCHS + 1):

    # ---- TRAIN ----
    model.train()
    train_losses = []
    train_accs = []
    y_preds, y_trues = [], []

    for G, y_train in yield_subgraphs(train_flows, le, window=WINDOW):
        optimizer.zero_grad()
        out = model(G.x, G.edge_index)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        pred = torch.argmax(out, dim=1)
        acc = (pred == y_train).float().mean().item()
        train_accs.append(acc)

        y_trues.append(y_train.cpu())
        y_preds.append(pred.cpu())

    # log class wise AUC
    y_trues = torch.cat(y_trues).numpy()
    y_preds = torch.cat(y_preds).numpy()
    for a in classes:
        y_class = (y_trues == le.transform([a])[0]).astype(int)
        y_pred_class = y_preds == le.transform([a])[0]
        class_auc = roc_auc_score(y_class, y_pred_class)
        writer.add_scalar(f"AUC/Train/{a}_auc", class_auc, epc)
                     

    avg_train_loss = np.mean(train_losses)
    avg_train_acc = np.mean(train_accs)

    # ---- TEST ----
    model.eval()
    test_losses = []
    test_accs = []
    y_trues, y_preds = [], []

    with torch.no_grad():
        for G, y_test in yield_subgraphs(test_flows, le, window=WINDOW):
            out = model(G.x, G.edge_index)
            loss = criterion(out, y_test)

            test_losses.append(loss.item())

            pred = torch.argmax(out, dim=1)
            acc = (pred == y_test).float().mean().item()
            test_accs.append(acc)

            y_trues.append(y_test.cpu())
            y_preds.append(pred.cpu())

    # log class wise AUC
    y_trues = torch.cat(y_trues).numpy()
    y_preds = torch.cat(y_preds).numpy()
    for a in classes:
        y_class = (y_trues == le.transform([a])[0]).astype(int)
        y_pred_class = y_preds == le.transform([a])[0]
        class_auc = roc_auc_score(y_class, y_pred_class)
        writer.add_scalar(f"AUC/Test/{a}_auc", class_auc, epc)

    avg_test_loss = np.mean(test_losses)
    avg_test_acc = np.mean(test_accs)

    # ---- TensorBoard Logging ----
    writer.add_scalar("Loss/Train", avg_train_loss, epc)
    writer.add_scalar("Loss/Test", avg_test_loss, epc)
    writer.add_scalar("Accuracy/Train", avg_train_acc, epc)
    writer.add_scalar("Accuracy/Test", avg_test_acc, epc)

    print(
        f"Epoch {epc:02d} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Train Acc: {avg_train_acc:.4f} | "
        f"Test Acc: {avg_test_acc:.4f}"
    )

    # Save best model
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), exp_dir / "best_model.pt")

# ------------------ Metadata ------------------

experiment_summary = {
    "description": "Multiclass GraphSAGE with linegraph on UNSW-NB15",
    "epochs": EPOCHS,
    'model_kwargs': model_kwargs,
    "window_size": WINDOW,
    "le": le,
    "best_test_loss": best_test_loss,
    "label_encoder_classes": list(classes),
}

with open(exp_dir / "experiment.pkl", "wb") as f:
    pickle.dump(experiment_summary, f)

writer.close()
