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
from ML_utils import graph_encode


def get_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, threshold=0.5):
    y_pred = y_pred_probs > threshold
    P, R = (
        precision_score(y_true, y_pred, pos_label=1),
        recall_score(y_true, y_pred, pos_label=1),
    )
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs, pos_label=1)
    pr_auc = auc(recall, precision)
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return (pr_auc, roc_auc_score(y_true, y_pred_probs), F1, P, R)


def most_recent_object(exp_dir):
    """tensorboard log directory (most recently created in interm/runs)"""
    exp_dirs = list(Path(exp_dir).glob("*"))
    exp_dir = max(exp_dirs, key=lambda d: d.stat().st_ctime)
    logger.info(f"Using newest experiment directory: {exp_dir}")
    return exp_dir


# ----- Setup

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default=None)
parser.add_argument("--nrows-test", default=None, type=int)
parser.add_argument("--eval-dir", default=None, type=str)
args = parser.parse_args()

# input directory
if args.run_dir is None:
    logger.warning('no run dir provided, using most recent in "interm/runs"')
    exp_dir = most_recent_object("interm/runs")
else:
    exp_dir: Path = Path(args.run_dir)
    logger.info(f"Using specified experiment directory: {exp_dir}")

# output directory (defaults to experiment/eval)
if args.eval_dir is None:
    eval_dir = exp_dir / "eval"
else:
    eval_dir = Path(args.eval_dir)

eval_dir.mkdir(exist_ok=True)
logger.info(f"Evaluation results will be saved to: {eval_dir}")

device = "cpu"
logger.info(f"Experiment directory: {exp_dir}")

# Load experiment metadata
with open(exp_dir / "experiment.pkl", "rb") as f:
    cfg = pickle.load(f)

test_csv = Path(cfg["test_df_location"])

# Rebuild model
with open(exp_dir / "current_model.pkl", "rb") as f:
    model = pickle.load(f)
model.to(device)
model.eval()

# Load test data
logger.info(f"Loading test data from: {test_csv}")
test_flows = (
    pd.read_csv(test_csv)
    if args.nrows_test is None
    else pd.read_csv(test_csv, nrows=args.nrows_test)
)

# Binary encoding (same logic as training)
attack_labels = test_flows["Attack"].values  # original string labels
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float)
).float()

# Dummy criterion (not used in eval)
criterion = torch.nn.BCEWithLogitsLoss()


# -- Binary Evaluation

G, _ = graph_encode(
    test_flows, edge_cols=["src", "dst"], linegraph=False, target_col="Attack"
)
with torch.no_grad():
    loss, y, y_probs, emb = model.pass_flowgraph(
        G, criterion, optimizer=None, train_now=False
    )

# histogram of predicted probabilities
plt.figure()
plt.hist(y_probs, bins=500)
plt.title("Prediction Probability Distribution")
plt.show()
plt.clf()

# get the best threshhold
# candidate_threshholds = np.linspace(0, 1, 500)
best_f1 = 0
best_thresh = 0.5
# logger.info(f"Best threshold: {best_thresh:.4f} with F1: {best_f1:.4f}")
print("!!! not serching for thershold, using 0.5 !!!")

y_probs = np.array(y_probs)

logger.info(
    "n of mal edges edges predicted with threshold: {}".format(
        (y_probs > best_thresh).sum()
    )
)
y_pred_bin = (y_probs > best_thresh).astype(int)

# 1. Binary Evaluation
print("\n===== BINARY (Benign vs Malicious) =====")
y_true_bin = y.cpu().numpy()
print(
    f"results from previous function in the order of (PR-AUC, ROC-AUC, F1, Precision, Recall) : {get_metrics(y_true_bin, y_probs)}"
)
print("ROC AUC:", roc_auc_score(y_true_bin, y_probs))
precision, recall, _ = precision_recall_curve(y_true_bin, y_probs)
print("PR AUC:", auc(recall, precision))
print(classification_report(y_true_bin, y_pred_bin, digits=4))
with open(eval_dir / "classification_report.txt", "w") as f:
    f.write(classification_report(y_true_bin, y_pred_bin, digits=4))

# ROC
fpr, tpr, _ = roc_curve(y_true_bin, y_probs)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve (Binary)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.savefig(eval_dir / "ROC Curve (Binary).png")
plt.clf()

# PR
prec, rec, _ = precision_recall_curve(y_true_bin, y_probs)
plt.figure()
plt.plot(rec, prec)
plt.title("PR Curve (Binary)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.savefig(eval_dir / "PR Curve (Binary).png")
plt.clf()

# Histogram
plt.figure()
plt.hist(y_probs, bins=50)
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.savefig(eval_dir / "Prediction Probability Distribution.png")
plt.clf()


# --- Per class evaluation

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
    with open(eval_dir / f"classification_report_{attack}.txt", "w") as f:
        f.write(classification_report(y_true_attack, y_pred_attack, digits=4))

    # ROC
    fpr, tpr, _ = roc_curve(y_true_attack, y_probs_attack)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC - {attack}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.savefig(eval_dir / f"ROC - {attack}.png")
    plt.clf()

    # PR
    prec, rec, _ = precision_recall_curve(y_true_attack, y_probs_attack)
    plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR - {attack}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.savefig(eval_dir / f"PR - {attack}.png")
    plt.clf()

    # Histogram
    plt.figure()
    plt.hist(y_probs_attack, bins=50)
    plt.title(f"Probability Distribution - {attack}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.savefig(eval_dir / f"Probability Distribution - {attack}.png")
    plt.clf()

    logger.info(f"finished eval for experiment {exp_dir}")
