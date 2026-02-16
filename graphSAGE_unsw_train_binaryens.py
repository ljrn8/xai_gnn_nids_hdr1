import pandas as pd
from sklearn.metrics import roc_auc_score
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

WINDOW = 20_000
LR = 0.005
EPOCHS = 30

# ------------------ Setup ------------------

device = 'cpu'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

print('Loading data...')
train_flows = pd.read_csv("interm/unsw_nb15_processed_train.csv")
test_flows = pd.read_csv("interm/unsw_nb15_processed_test.csv")
classes= list(np.unique(train_flows.Attack))
train_flows = train_flows[train_flows.Attack.isin(classes)]
test_flows = test_flows[test_flows.Attack.isin(classes)]

flows = pd.concat([train_flows, test_flows], ignore_index=True)
print('Loaded')

le = LabelEncoder()
le.fit(flows.Attack)

exp_dir = Path(f"interm/runs/binaryens_unsw_graphsage_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=exp_dir)

# ------------------- Ensemble -------------------

for a in classes:
    if a == 'Benign' or a == 'Analysis': # !! skip analysis for now
        continue

    print(f'Training binary classifier for {a}...')

    # binary classification for each class
    train_cp = train_flows.copy()
    test_cp = test_flows.copy()
    train_cp.Attack = (train_cp.Attack == a).astype(int)
    test_cp.Attack = (test_cp.Attack == a).astype(int)

    # only sample reasonable attack:benign ration from flows dataframe
    ben_flows = train_cp[train_cp.Attack == 0]
    mal_flows = train_cp[train_cp.Attack == 1]
    sample_i = np.random.choice(range(len(ben_flows)), size=50_000,  # !! 
                                replace=False)
    train_cp = pd.concat([ben_flows.iloc[sample_i], mal_flows])
    
        
    model_kwargs = {
        'in_channels': 72,
        'hidden_channels': 256,
        'out_channels': 2,
        'num_layers': 3,
    }

    model = GraphSAGE(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # weight classes
    print(np.unique(train_cp.Attack, return_counts=True))
    class_counts = np.bincount(train_cp.Attack)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device) 
    
    print(f"Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss() # !! not weighted

    best_test_loss = float("inf")

    # ------------------ Training Loop ------------------

    for epc in range(1, EPOCHS + 1):

        # ---- Train ----
        model.train()
        train_losses, train_accs = [], []
        y_preds, y_trues = [], []
        for G, y_train in yield_subgraphs(train_cp, window=WINDOW):
            optimizer.zero_grad()
            out = model(G.x, G.edge_index)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(out, dim=1)
            acc = (pred == y_train).float().mean().item()

            train_accs.append(acc)
            train_losses.append(loss.item())
            y_trues.append(y_train.cpu())
            y_preds.append(pred.cpu())

        # log class wise AUC
        y_trues = torch.cat(y_trues).numpy()
        y_preds = torch.cat(y_preds).numpy()
        y_class = (y_trues == le.transform([a])[0]).astype(int)
        y_pred_class = y_preds == le.transform([a])[0]
        class_auc = roc_auc_score(y_class, y_pred_class)
        writer.add_scalar(f"AUC/Train/{a}_auc", class_auc, epc)
                        
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)

        # -------- Test ---------
        model.eval()
        test_losses, test_accs = [], []
        y_trues, y_preds = [], []
        with torch.no_grad():
            for G, y_test in yield_subgraphs(test_cp, window=WINDOW):
                out = model(G.x, G.edge_index)
                loss = criterion(out, y_test)

                pred = torch.argmax(out, dim=1)
                acc = (pred == y_test).float().mean().item()
                test_accs.append(acc)
                y_trues.append(y_test.cpu())
                y_preds.append(pred.cpu())

        # log class wise AUC
        y_trues = torch.cat(y_trues).numpy()
        y_preds = torch.cat(y_preds).numpy()
        y_class = (y_trues == le.transform([a])[0]).astype(int)
        y_pred_class = y_preds == le.transform([a])[0]
        class_auc = roc_auc_score(y_class, y_pred_class)
        writer.add_scalar(f"AUC/Test/{a}_auc", class_auc, epc)

        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accs)

        # TensorBoard Logging 
        writer.add_scalar(f"Loss/{a}_Train", avg_train_loss, epc)
        writer.add_scalar(f"Loss/{a}_Test", avg_test_loss, epc)
        writer.add_scalar(f"Accuracy/{a}_Train", avg_train_acc, epc)
        writer.add_scalar(f"Accuracy/{a}_Test", avg_test_acc, epc)

        print(
            f"{a} Epoch {epc:02d} | "
            f"{a} Train Loss: {avg_train_loss:.4f} | "
            f"{a} Test Loss: {avg_test_loss:.4f} | "
            f"{a} Train Acc: {avg_train_acc:.4f} | "
            f"{a} Test Acc: {avg_test_acc:.4f}"
        )

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), exp_dir / f"best_model_{a}.pt")

    # ------------------ Metadata ------------------

    experiment_summary = {
        "description": f"{a} Multiclass GraphSAGE with linegraph on UNSW-NB15",
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
