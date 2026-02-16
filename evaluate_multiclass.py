import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from preprocessing import yield_subgraphs
import pickle
import argparse

# ------------------ Args ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, required=True)
args = parser.parse_args()

exp_dir = Path(args.exp_dir)

# ------------------ Load metadata ------------------
with open(exp_dir / "experiment.pkl", "rb") as f:
    meta = pickle.load(f)
    
# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load Model ------------------
model = GraphSAGE(
    **meta['model_kwargs']
).to(device)

model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
model.eval()

# ------------------ Load Test Data ------------------
test_flows = pd.read_csv("interm/unsw_nb15_processed_test.csv")

le = meta['le']

all_preds = []
all_probs = []
all_targets = []
all_raw_preds = []

# ------------------ Evaluation Loop ------------------
with torch.no_grad():
    for G, y_test in yield_subgraphs(test_flows, le, window=meta['window']):
        G = G.to(device)
        y_test = y_test.to(device)


        logits = model(G.x, G.edge_index)

        probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1)

        all_raw_preds.append(logits)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_targets.append(y_test.cpu())

# Concatenate
all_raw_preds = torch.cat(all_raw_preds).numpy()
all_preds = torch.cat(all_preds).numpy()
all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# ------------------ Classification Report ------------------
print("\n=== Classification Report ===\n")
print(classification_report(
    all_targets,
    all_preds,
    target_names=meta['le'].classes_,
    digits=4
))

# ------------------ ROC-AUC (Multiclass OvR) ------------------
roc_auc = roc_auc_score(
    all_targets,
    all_probs,
    multi_class="ovr",
    average="macro"
)

print(f"\nROC-AUC (macro, OvR): {roc_auc:.4f}")
print(f'\nSample first 10 predictions: {all_raw_preds[:10]}')

