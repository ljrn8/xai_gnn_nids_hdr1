import numpy as np
import pickle
import pandas as pd
from loguru import logger
from ML_utils import graph_encode, fidelities, most_recent_object
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import pickle
from pathlib import Path
from EGraphSAGE import EGraphSAGE
import argparse
from tqdm import tqdm
from N_PGExplainer import N_PGExplainer
from L_PGExplainer import L_PGExplainer

# -- setup

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument("--xAI-run-dir", default=None, type=str)
parser.add_argument("--eval-dir", default="figures/xAI_graphs")
parser.add_argument("--skip-show-graphs", action="store_true")
parser.add_argument("--best-mask", action="store_true")
args = parser.parse_args()
skip = args.skip_show_graphs

# load experiment
pkl = "best_mask.pkl" if args.best_mask else "current_mask.pkl"

run_dir = Path(args.xAI_run_dir)
logger.info(f"Loading xAI experiment from: {run_dir}")

with open(run_dir / pkl, "rb") as f:
    run = pickle.load(f)

# load test data from experiment metadata
test_f = run["info"]["test_f"]
test_flows = pd.read_csv(test_f)

# load model from experiment metadata
with open(Path(run["info"]["model_dir"]), "rb") as f:
    model = pickle.load(f)

eval_dir = Path(args.eval_dir)
eval_dir.mkdir(exist_ok=True, parents=True)

# setup test graph
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()
G, _ = graph_encode(
    test_flows, edge_cols=["src", "dst"], linegraph=False, target_col="Attack"
)

mask = run["node_mask"] if "node_mask" in run.keys() else run["edge_mask"]

if run["info"]["mask_type"] == "node":
    # if using a node mask, compute the edge mask is computed from the highest node value
    # NOTE: this produces a far denser edge mask
    src, dst = G.edge_index
    node_mask = mask
    edge_mask = torch.max(node_mask[src], node_mask[dst])
else:
    edge_mask = mask
    # infer node mask by max edge
    # NOTE: this produces a far denser node mask, and is not the same as using a node mask directly
    num_nodes = G.edge_index.max().item() + 1
    src, dst = G.edge_index
    node_mask = torch.zeros(num_nodes)

    # scatter max over both src and dst nodes
    node_mask.scatter_reduce_(
        0, src, torch.FloatTensor(edge_mask), reduce="amax", include_self=True
    )
    node_mask.scatter_reduce_(
        0, dst, torch.FloatTensor(edge_mask), reduce="amax", include_self=True
    )


# -- plot basic mask info

losses = run["losses"]
plt.figure(figsize=(4, 4))
plt.plot(losses)
plt.title("loss")
plt.savefig(eval_dir / "loss.png")
if not skip:
    plt.show()
else:
    plt.clf()

logger.info("plotting regularization")
if "mask_regularization" in run.keys():
    logger.info("using mask_regularization array")

    regs = run["mask_regularization"]
    entr_reg, mean_reg, mlp_l1_reg = [[r[i].detach() for r in regs] for i in range(3)]
    plt.figure(figsize=(4, 4))
    plt.plot(entr_reg)
    plt.title("entropy regularization")
    plt.savefig(eval_dir / "entropy regularization.png")
    if not skip:
        plt.show()
    else:
        plt.clf()

    plt.figure(figsize=(4, 4))
    plt.plot(mean_reg)
    plt.title("mean regularization")
    plt.savefig(eval_dir / "mean regularization.png")
    if not skip:
        plt.show()
    else:
        plt.clf()

    plt.figure(figsize=(4, 4))
    plt.plot(mlp_l1_reg)
    plt.title("MLP L1 regularization")
    plt.savefig(eval_dir / "MLP L1 regularization.png")
    if not skip:
        plt.show()
    else:
        plt.clf()

# newer way to store regularization values, as a dict in the run info
elif "regularization" in run.keys():
    logger.info("using regularization dict")
    regs = run["regularization"]
    for name, regs in regs.items():
        plt.figure(figsize=(4, 4))
        plt.plot(regs)
        plt.title(f"{name} regularization")
        plt.savefig(eval_dir / f"{name} regularization.png")
        if not skip:
            plt.show()
        else:
            plt.clf()


plt.figure(figsize=(4, 4))
plt.hist(mask.detach().numpy(), bins=500)
plt.savefig(eval_dir / "mask hist.png")
if not skip:
    plt.show()
else:
    plt.clf()

# normal predictions reference
y_pred, _, _ = model.forward(
    G.edge_attr,
    G.edge_index,
)


def fidelities(y_pred, y_mask, y_imask, y):
    """Phenominal fidelity+ and Fidelity- (expects thresholded values)"""
    fp = ((y_pred == y).float() - (y_imask == y).float()).abs().mean()
    fm = ((y_pred == y).float() - (y_mask == y).float()).abs().mean()
    return fp, fm


# --------------------------------
# --- Edge mask sparsity variation

sparsities = np.arange(0, 0.5, 0.001)
fps, fms, threshes = [], [], []
for s in tqdm(sparsities, desc=f"Evaluating masks at sparsities"):

    # threshhold sparsities over the edge mask
    edge_mask = (
        edge_mask.detach().numpy()
        if type(edge_mask) in [torch.FloatTensor, torch.Tensor]
        else edge_mask
    )
    threshold = np.percentile(edge_mask, (1 - s) * 100)
    binary_mask = torch.FloatTensor(edge_mask > threshold)

    masked_edge_attr = G.edge_attr * binary_mask.unsqueeze(1)
    masked_y_pred, _, _ = model.forward(
        masked_edge_attr,
        G.edge_index,
    )
    binary_masked_y_pred = (masked_y_pred > 0.5).float()
    binary_y_pred = (y_pred > 0.5).float()
    malicious_mask = binary_y_pred == 1.0

    # NOTE: only computing fidelities for malicious flows
    fp, fm = fidelities(
        y_pred=binary_y_pred[malicious_mask],
        y_mask=binary_masked_y_pred[malicious_mask],
        y_imask=1 - binary_masked_y_pred[malicious_mask],
        y=G.y[malicious_mask],
    )

    fps.append(fp)
    fms.append(fm)
    threshes.append(threshold)

    with open(eval_dir / "edge mask sparsity variation.csv", "wb") as f:
        pickle.dump((sparsities, fps, fms, threshes), f)


plt.figure(figsize=(4, 4))
plt.plot(sparsities, fps)
plt.title("fid+")
plt.savefig(eval_dir / "edge mask fid+.png")
plt.xlim((0, 1))
plt.ylim((0, 1))
if not skip:
    plt.show()
else:
    plt.clf()

plt.figure(figsize=(4, 4))
plt.title("fid-")
plt.plot(sparsities, fms)
plt.savefig(eval_dir / "edge mask fid-.png")
plt.xlim((0, 1))
plt.ylim((0, 1))
if not skip:
    plt.show()
else:
    plt.clf()

plt.figure(figsize=(4, 4))
plt.plot(sparsities, threshes)
plt.savefig(eval_dir / "edge mask threshholds.png")
if not skip:
    plt.show()
else:
    plt.clf()


# ---------------------------------
# --- Node mask sparsity variation

sparsities = np.arange(0, 0.5, 0.001)
fps, fms, threshes = [], [], []
for s in tqdm(sparsities, desc=f"Evaluating masks at sparsities"):

    # threshhold sparisity over the node mask
    node_mask = node_mask.detach().numpy()
    threshold = np.percentile(node_mask, (1 - s) * 100)
    if type(node_mask) not in [torch.FloatTensor, torch.Tensor]:
        node_mask = torch.FloatTensor(node_mask)

    binary_mask = node_mask > threshold

    # infer back edge mask by max node
    src, dst = G.edge_index
    edge_mask = torch.max(binary_mask[src], binary_mask[dst])

    masked_edge_attr = G.edge_attr * edge_mask.unsqueeze(1)
    masked_y_pred, _, _ = model.forward(
        masked_edge_attr,
        G.edge_index,
    )
    binary_masked_y_pred = (masked_y_pred > 0.5).float()
    binary_y_pred = (y_pred > 0.5).float()
    malicious_mask = binary_y_pred == 1.0

    # NOTE: only computing fidelities for malicious flows
    fp, fm = fidelities(
        y_pred=binary_y_pred[malicious_mask],
        y_mask=binary_masked_y_pred[malicious_mask],
        y_imask=1 - binary_masked_y_pred[malicious_mask],
        y=G.y[malicious_mask],
    )

    fps.append(fp)
    fms.append(fm)
    threshes.append(threshold)

    with open(eval_dir / "node mask sparsity variation.csv", "wb") as f:
        pickle.dump((sparsities, fps, fms, threshes), f)


plt.figure(figsize=(4, 4))
plt.plot(sparsities, fps)
plt.title("fid+")
plt.savefig(eval_dir / "node mask fid+.png")
plt.xlim((0, 1))
plt.ylim((0, 1))
if not skip:
    plt.show()
else:
    plt.clf()

plt.figure(figsize=(4, 4))
plt.title("fid-")
plt.plot(sparsities, fms)
plt.savefig(eval_dir / "node mask fid-.png")
plt.xlim((0, 1))
plt.ylim((0, 1))
if not skip:
    plt.show()
else:
    plt.clf()

plt.figure(figsize=(4, 4))
plt.plot(sparsities, threshes)
plt.savefig(eval_dir / "node mask threshholds.png")
if not skip:
    plt.show()
else:
    plt.clf()
