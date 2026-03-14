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


parser = argparse.ArgumentParser()
parser.add_argument("--experiment-pickle", default=most_recent_object("./interm/xAI"))
parser.add_argument("--output-figures-directory", default="figures/xAI_graphs")
parser.add_argument("--skip-show-graphs", action='store_true')
args = parser.parse_args()
skip = args.skip_show_graphs

# load experiment
with open(args.experiment_pickle, "rb") as f:
    explainability_report = pickle.load(f)

# load test data from experiment metadata
test_f = explainability_report["test_f"]
test_flows = pd.read_csv(test_f)

# load model from experiment metadata
with open(Path(explainability_report["model_dir"]), "rb") as f:
    model = pickle.load(f)

losses, reg_losses = (
    explainability_report["results"]["losses"],
    explainability_report["results"]["mask_regularization"],
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_output = args.output_digures_directory / timestamp
figures_output.mkdir(exist_ok=True, parents=True)

plt.figure(figsize=(4, 4))
plt.plot(losses)
plt.title("loss")
plt.savefig(figures_output / "loss.png")
if not skip: plt.show()
else: plt.clf()

plt.figure(figsize=(4, 4))
plt.plot(reg_losses)
plt.title("reg loss")
plt.savefig(figures_output / "reg losses.png")
if not skip: plt.show()
else: plt.clf()

mask = explainability_report["results"]["mask"].detach().numpy()
plt.hist(mask, bins=500)
plt.savefig(figures_output / "mask hist.png")
if not skip: plt.show()
else: plt.clf()

# convert test_flows Attack to binary
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()
G, _ = graph_encode(
    test_flows, edge_cols=["src", "dst"], linegraph=False, target_col="Attack"
)

# normal predictions reference
y_pred, _ = model.forward(
    G.edge_attr,
    G.edge_index,
)

# sparsity variation graphs
sparsities = np.arange(0, 1.0, 0.02)
fps, fms, threshes = [], [], []
for s in tqdm(sparsities, desc=f"Evaluating masks at spasities"):
    # threshold = np.percentile(mask, s * 100)
    threshold = np.percentile(mask, (1 - s) * 100)

    # can use non differnetiable threshholding here
    binary_edge_mask = torch.FloatTensor(mask > threshold)
    masked_edge_attr = G.edge_attr * binary_edge_mask.unsqueeze(1)
    masked_y_pred, _ = model.forward(
        masked_edge_attr,
        G.edge_index,
    )
    binary_masked_y_pred = (masked_y_pred > 0.5).float()
    binary_y_pred = (y_pred > 0.5).float()
    malicious_mask = binary_y_pred == 1.0

    def fidelities(y_pred, y_mask, y_imask, y):
        """Phenominal fidelity+ and Fidelity- (expects THRESHOLDED values)"""
        fp = ((y_pred == y).float() - (y_imask == y).float()).abs().mean()
        fm = ((y_pred == y).float() - (y_mask == y).float()).abs().mean()
        return fp, fm

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

print("threshholds: ", threshes)

plt.figure(figsize=(4, 4))
plt.plot(sparsities, fps)
plt.title("fid+")
plt.savefig(figures_output / "fid+")
plt.xlim((0, 1))
plt.ylim((0, 1))
if not skip: plt.show()
else: plt.clf()

plt.figure(figsize=(4, 4))
plt.title("fid-")
plt.plot(sparsities, fms)
plt.savefig(figures_output / "fid-")
plt.xlim((0, 1))
plt.ylim((0, 1))
if not skip: plt.show()
else: plt.clf()

plt.figure(figsize=(4, 4))
plt.plot(sparsities, threshes)
plt.savefig(figures_output / "Threshholds")
if not skip: plt.show()
else: plt.clf()
