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


parser = argparse.ArgumentParser()
parser.add_argument("--xAI-output-directory", default=most_recent_object("./interm/xAI"))
parser.add_argument("--output-directory", default="figures/xAI_graphs")
parser.add_argument("--skip-show-graphs", action='store_true')
parser.add_argument("--best-mask", action='store_true')
args = parser.parse_args()
skip = args.skip_show_graphs

# -- extract xAI experiment (model, data, mask and losses)

# load experiment
pkl = 'best_mask.pkl' if args.best_mask else 'current_mask.pkl'
with open(Path(args.xAI_output_directory) / pkl, "rb") as f:
    run = pickle.load(f)

# load test data from experiment metadata
test_f = run["info"]['test_f']
test_flows = pd.read_csv(test_f)

# load model from experiment metadata
with open(Path(run["info"]["model_dir"]), "rb") as f:
    model = pickle.load(f)

losses, regs = run['losses'], run['mask_regularization']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_output = Path(args.output_directory) / timestamp
figures_output.mkdir(exist_ok=True, parents=True)


# -- plot basic mask info

plt.figure(figsize=(4, 4))
plt.plot(losses)
plt.title("loss")
plt.savefig(figures_output / "loss.png")
if not skip: plt.show()
else: plt.clf()

entr_reg, mean_reg, mlp_l1_reg = [
    [r[i].detach() for r in regs] for i in range(3)
]

plt.figure(figsize=(4, 4))
plt.plot(entr_reg)
plt.title("entropy regularization")
plt.savefig(figures_output / "entropy regularization.png")
if not skip: plt.show()
else: plt.clf()

plt.figure(figsize=(4, 4))
plt.plot(mean_reg)
plt.title("mean regularization")
plt.savefig(figures_output / "mean regularization.png")
if not skip: plt.show()
else: plt.clf()

plt.figure(figsize=(4, 4))
plt.plot(mlp_l1_reg)
plt.title("MLP L1 regularization")
plt.savefig(figures_output / "MLP L1 regularization.png")
if not skip: plt.show()
else: plt.clf()

mask = run["node_mask"].detach().numpy()
plt.hist(mask, bins=500)
plt.savefig(figures_output / "mask hist.png")
if not skip: plt.show()
else: plt.clf()


# -- sparsity variation graphs

# convert test_flows Attack to binary
test_flows["Attack"] = torch.Tensor(
    (test_flows["Attack"] != "Benign").astype(float).values
).float()
G, _ = graph_encode(
    test_flows, edge_cols=["src", "dst"], linegraph=False, target_col="Attack"
)

# normal predictions reference
y_pred, _, _ = model.forward(
    G.edge_attr,
    G.edge_index,
)

sparsities = np.arange(0, 1.0, 0.02)
fps, fms, threshes = [], [], []
for s in tqdm(sparsities, desc=f"Evaluating masks at spasities"):

    # threshold = np.percentile(mask, s * 100)
    threshold = np.percentile(mask, (1 - s) * 100)
    binary_mask = torch.FloatTensor(mask > threshold)

    # if run['node_mask']:
    if True: # ! temparary
        # if using a node mask, compute the edge mask is computed from the highest node value
        # ! this produces a far denser edge mask
        src, dst = G.edge_index
        binary_mask = torch.max(
                binary_mask[src], binary_mask[dst]
            )

    masked_edge_attr = G.edge_attr * binary_mask.unsqueeze(1)
    masked_y_pred, _, _ = model.forward(
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
