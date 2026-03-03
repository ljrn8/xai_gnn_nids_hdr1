import pickle
from xml.parsers.expat import model
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from tqdm import tqdm
from EGraphSAGE import EGraphSAGE
import sys
from loguru import logger


def get_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = y_pred_probs > threshold
    P, R = (
        precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1),
    )
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs, pos_label=1)
    pr_auc = auc(recall, precision)
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return (pr_auc, roc_auc_score(y_true, y_pred_probs), F1, P, R)


# ---------------- SETUP ----------------

device = "cpu"

# tensorboard log directory (most recently created in interm/runs)
exp_dir: Path = Path(sys.argv[1])
if exp_dir == Path("newest"):
    exp_dirs = list(Path("interm/runs").glob("*"))
    exp_dir = max(exp_dirs, key=lambda d: d.stat().st_ctime)
    logger.info(f"Using newest experiment directory: {exp_dir}")
else:
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory {exp_dir} does not exist.")
    logger.info(f"Using specified experiment directory: {exp_dir}")

logger.info(f"Experiment directory: {exp_dir}")
figure = Path("figures") / exp_dir.name
figure.mkdir(exist_ok=True)

# Load experiment metadata
with open(exp_dir / "experiment.pkl", "rb") as f:
    cfg = pickle.load(f)

# test flows directory is given or in experimental config pickle
if len(sys.argv) > 2:
    test_csv = Path(sys.argv[2])
else:
    test_csv = Path(cfg["test_df_location"])

model_kwargs = cfg["model_kwargs"]
WINDOW = cfg["window_size"]

# Rebuild model
with open(exp_dir / "best_model.pkl", "rb") as f:
    model = pickle.load(f)
model.to(device)
model.eval()

# Load test data
logger.info(f"Loading test data from: {test_csv}")
test_flows = pd.read_csv(test_csv)
attack_labels = test_flows["Attack"].values  # original string labels

# Binary encoding (same logic as training)
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float)
).float()

# Dummy criterion (not used in eval)
criterion = torch.nn.BCEWithLogitsLoss()

# --------------- EVAL ---------------

with torch.no_grad():
    avg_loss, y_true_bin, y_probs, _ = model.train_flows(
        test_flows,
        criterion=criterion,
        optimizer=None,
        window=WINDOW,
        train=False,
    )

y_true_bin = np.array(y_true_bin)
y_probs = np.array(y_probs)
logger.debug(f"start of y_probs: {y_probs[:10]}")


# histogram of predicted probabilities
plt.figure()
plt.hist(y_probs, bins=100)
plt.title("Prediction Probability Distribution")
plt.show()
plt.clf()

# get the best threshhold
candidate_threshholds = np.linspace(0, 1, 500)
best_f1 = 0
best_thresh = 0.5
print("!!! not serching for thershold, using 0.5 !!!")
# for t in tqdm(candidate_threshholds, desc="Finding best threshold for f1"):
#     y_pred = (y_probs > t).astype(int)
#     P, R = (
#         precision_score(y_true_bin, y_pred, pos_label=1),
#         recall_score(y_true_bin, y_pred, pos_label=1),
#     )
#     if P == 0.0:
#         break # stopped predicting any positives, no need to continue
#     f1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
#     if f1 > best_f1:
#         best_f1 = f1
#         best_thresh = t

logger.info(f"Best threshold: {best_thresh:.4f} with F1: {best_f1:.4f}")
logger.info(
    "n of mal edges edges predicted with threshold: {}".format(
        (y_probs > best_thresh).sum()
    )
)
y_pred_bin = (y_probs > best_thresh).astype(int)

# 1. Binary Evaluation
print("\n===== BINARY (Benign vs Malicious) =====")
print(
    f"results from previous function in the order of (PR-AUC, ROC-AUC, F1, Precision, Recall) : {get_metrics(y_true_bin, y_probs)}"
)
print("ROC AUC:", roc_auc_score(y_true_bin, y_probs))
print("Average test loss:", avg_loss)
precision, recall, _ = precision_recall_curve(y_true_bin, y_probs)
print("PR AUC:", auc(recall, precision))
print(classification_report(y_true_bin, y_pred_bin, digits=4))

# ROC
fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve (Binary)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.savefig(figure / "ROC Curve (Binary).png")
plt.clf()

# PR
prec, rec, _ = precision_recall_curve(y_true_bin, y_probs)
plt.figure()
plt.plot(rec, prec)
plt.title("PR Curve (Binary)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.savefig(figure / "PR Curve (Binary).png")
plt.clf()

# Histogram
plt.figure()
plt.hist(y_probs, bins=50)
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.savefig(figure / "Prediction Probability Distribution.png")
plt.clf()


# 2. Per-Attack One-vs-Benign
print("\n===== PER ATTACK TYPE (Benign vs Attack X) =====")

unique_attacks = np.unique(attack_labels)
unique_attacks = unique_attacks[unique_attacks != "Benign"]

for attack in unique_attacks:
    print(f"\n--- Attack: {attack} ---")

    mask = (attack_labels == "Benign") | (attack_labels == attack)

    y_true_attack = (attack_labels[mask] == attack).astype(int)
    y_probs_attack = y_probs[mask]
    y_pred_attack = (y_probs_attack > best_thresh).astype(int)

    if len(np.unique(y_true_attack)) < 2:
        print("Skipping (only one class present)")
        continue

    print("ROC AUC:", roc_auc_score(y_true_attack, y_probs_attack))
    precision, recall, _ = precision_recall_curve(y_true_attack, y_probs_attack)
    print("PR AUC:", auc(recall, precision))
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
