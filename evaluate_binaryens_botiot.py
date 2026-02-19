import pandas as pd
import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from torch_geometric.nn.models.basic_gnn import GraphSAGE
import matplotlib.pyplot as plt
from ML_utils import yield_subgraphs
from loguru import logger

# ------------------ CONFIG ------------------

EXP_DIR = Path("interm/runs/binaryens_botiot_graphsage_20260219_193042")
TEST_PATH = "interm/BotIoT_v1_processed_test.csv"
WINDOW = 5000
device = "cpu"

# --------------------------------------------


def evaluate_model(model, flows):
    model.eval()

    y_trues, y_probs = [], []

    with torch.no_grad():
        for G in yield_subgraphs(flows, window=WINDOW):
            y = G.y.to(device)
            x = G.x.to(device)

            out = model(x, G.edge_index.to(device))
            probs = torch.softmax(out, dim=1)[:, 1]

            y_trues.append(y.cpu())
            y_probs.append(probs.cpu())

    y_trues = torch.cat(y_trues).numpy()
    y_probs = torch.cat(y_probs).numpy()

    return y_trues, y_probs


def compute_metrics(y_true, y_prob):
    y_pred = y_prob > 0.5

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def plot_roc(y_true, y_prob, class_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {class_name}")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_pr(y_true, y_prob, class_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {class_name}")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ------------------ MAIN ------------------

logger.info("Loading experiment metadata...")
with open(EXP_DIR / "experiment.pkl", "rb") as f:
    experiment = pickle.load(f)

model_kwargs = experiment["model_kwargs"]
classes = experiment["label_encoder_classes"]

logger.info("Loading test data...")
test_flows = pd.read_csv(TEST_PATH)

results = []

for class_name in classes:
    if class_name == "Benign":
        continue

    logger.info(f"Evaluating model for {class_name}")

    # binary transform
    test_cp = test_flows.copy()
    test_cp.Attack = (test_cp.Attack == class_name).astype(int)

    # rebuild model
    model = GraphSAGE(**model_kwargs).to(device)
    model.load_state_dict(
        torch.load(EXP_DIR / f"best_model_{class_name}.pt", map_location=device)
    )

    # evaluate
    y_true, y_prob = evaluate_model(model, test_cp)

    metrics = compute_metrics(y_true, y_prob)
    metrics["Class"] = class_name
    results.append(metrics)

    # save curves
    plot_roc(
        y_true,
        y_prob,
        class_name,
        EXP_DIR / f"ROC_{class_name}.png",
    )

    plot_pr(
        y_true,
        y_prob,
        class_name,
        EXP_DIR / f"PR_{class_name}.png",
    )

# save metrics
results_df = pd.DataFrame(results)
results_df.to_csv(EXP_DIR / "evaluation_metrics.csv", index=False)

logger.info("Evaluation complete.")
print(results_df)
