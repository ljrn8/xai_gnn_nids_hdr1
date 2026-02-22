import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from EGraphSAGE import EGraphSAGE
import sys

# ---------------- CONFIG ----------------
device = "cpu"
exp_dir: Path = Path(sys.argv[1])
test_csv: Path = Path(sys.argv[2])
print(exp_dir)
figure = Path('figures') / exp_dir.name
figure.mkdir(exist_ok=True)
# ----------------------------------------

# ---------- Load experiment metadata ----------
with open(exp_dir / "experiment.pkl", "rb") as f:
    cfg = pickle.load(f)

model_kwargs = cfg["model_kwargs"]
WINDOW = cfg["window_size"]

# ---------- Rebuild model ----------
model = EGraphSAGE(**model_kwargs)
model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
model.to(device)
model.eval()

# ---------- Load test data ----------
test_flows = pd.read_csv(test_csv)
attack_labels = test_flows["Attack"].values  # original string labels

# Binary encoding (same logic as training)
test_flows["Attack"] = (test_flows["Attack"] != "Benign").astype(float)

# Dummy criterion (not used in eval)
criterion = torch.nn.BCEWithLogitsLoss()

# ---------- Run model ----------
with torch.no_grad():
    _, y_true_bin, y_probs, _ = model.train_flows(
        test_flows,
        criterion=criterion,
        optimizer=None,
        window=WINDOW,
        train=False,
    )

y_true_bin = np.array(y_true_bin)
y_probs = np.array(y_probs)
y_pred_bin = (y_probs > 0.5).astype(int)

# ============================================================
# -------------------- 1. Binary Evaluation ------------------
# ============================================================

print("\n===== BINARY (Benign vs Malicious) =====")
print("ROC AUC:", roc_auc_score(y_true_bin, y_probs))
print(classification_report(y_true_bin, y_pred_bin, digits=4))

# ROC
fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve (Binary)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.savefig(figure / 'ROC Curve (Binary).png')
plt.clf()

# PR
prec, rec, _ = precision_recall_curve(y_true_bin, y_probs)
plt.figure()
plt.plot(rec, prec)
plt.title("PR Curve (Binary)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.savefig(figure / 'PR Curve (Binary).png')
plt.clf()

# Histogram
plt.figure()
plt.hist(y_probs, bins=50)
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.savefig(figure / 'Prediction Probability Distribution.png')
plt.clf()


# ============================================================
# ------------- 2. Per-Attack One-vs-Benign ------------------
# ============================================================

print("\n===== PER ATTACK TYPE (Benign vs Attack X) =====")

unique_attacks = np.unique(attack_labels)
unique_attacks = unique_attacks[unique_attacks != "Benign"]

for attack in unique_attacks:
    print(f"\n--- Attack: {attack} ---")

    mask = (attack_labels == "Benign") | (attack_labels == attack)

    y_true_attack = (attack_labels[mask] == attack).astype(int)
    y_probs_attack = y_probs[mask]
    y_pred_attack = (y_probs_attack > 0.5).astype(int)

    if len(np.unique(y_true_attack)) < 2:
        print("Skipping (only one class present)")
        continue

    print("ROC AUC:", roc_auc_score(y_true_attack, y_probs_attack))
    print(classification_report(y_true_attack, y_pred_attack, digits=4))

    # ROC
    fpr, tpr, _ = roc_curve(y_true_attack, y_probs_attack)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC - {attack}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.savefig(figure / f"ROC - {attack}.png")
    plt.clf()

    # PR
    prec, rec, _ = precision_recall_curve(y_true_attack, y_probs_attack)
    plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR - {attack}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.savefig(figure / f"PR - {attack}.png")
    plt.clf()

    # Histogram
    plt.figure()
    plt.hist(y_probs_attack, bins=50)
    plt.title(f"Probability Distribution - {attack}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.savefig(figure / f"Probability Distribution - {attack}.png")
    plt.clf()