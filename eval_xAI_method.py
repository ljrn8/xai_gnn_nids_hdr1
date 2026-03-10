import numpy as np
import pickle
import pandas as pd
from loguru import logger
from ML_utils import graph_encode, fidelities
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import pickle
from pathlib import Path
from EGraphSAGE import EGraphSAGE
 
def most_recent_object(exp_dir):
    exp_dirs = list(Path(exp_dir).glob("*"))
    exp_dir = max(exp_dirs, key=lambda d: d.stat().st_ctime)
    logger.info(f"Using newest experiment directory: {exp_dir}")
    return Path(exp_dir)

# xAI pickle and test data
test_f = sys.argv[1] if len(sys.argv) > 1 else './interm/unsw_nb15_processed_test.csv'
test_flows = pd.read_csv(test_f)
metrics_dir = most_recent_object('./interm/xAI/GNNE_20260309_132324')
with open(metrics_dir, 'rb') as f:
	explainability_report = pickle.load(f)
with open(Path(explainability_report['meta']['model_dir']), 'rb')as f:
	model = pickle.load(f)

losses, reg_losses = explainability_report['results']['losses'], explainability_report['results']['mask_regularization']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figures_output = Path('figures/xAI_graphs') / timestamp
figures_output.mkdir(exist_ok=True, parents=True)

plt.figure(figsize=(4,4))
plt.plot(losses)
plt.title('loss')
plt.savefig(figures_output / 'loss.png')
plt.show()

plt.figure(figsize=(4,4))
plt.plot(reg_losses)
plt.title('reg loss')
plt.savefig(figures_output / 'reg losses.png')
plt.show()

mask = explainability_report['results']['mask'].detach().numpy()
plt.hist(mask, bins=500)
plt.savefig(figures_output / 'mask hist.png')
plt.show()

# convert test_flows Attack to binary
test_flows["Attack"] = torch.Tensor(
	(test_flows["Attack"] != "Benign").astype(float).values
).float()
G, _ = graph_encode(test_flows, edge_cols=['src','dst'], linegraph=False, target_col='Attack')


sparsities=np.arange(0, 1.0, 0.02)

from tqdm import tqdm

y_pred, _ = model.forward(
	G.edge_attr,
	G.edge_index,
)

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

	# NOTE: only computing fidelities for malicious flows

	def fidelities(y_pred, y_mask, y_imask, y):
		"""Phenominal fidelity+ and Fidelity- (expects THRESHOLDED values)"""
		fp = ((y_pred == y).float() - (y_imask == y).float()).abs().mean()
		fm = ((y_pred == y).float() - (y_mask == y).float()).abs().mean()
		return fp, fm

	fp, fm = fidelities(
		y_pred=binary_y_pred[malicious_mask],
		y_mask=binary_masked_y_pred[malicious_mask],
		y_imask=1 - binary_masked_y_pred[malicious_mask],
		y=G.y[malicious_mask],
	)

	fps.append(fp)
	fms.append(fm)
	threshes.append(threshold)

print('threshholds: ', threshes)

plt.figure(figsize=(4,4)); 
plt.plot(sparsities, fps); 
plt.title('fid+')
plt.savefig(figures_output / 'fid+')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()

plt.figure(figsize=(4,4))
plt.title('fid-') 
plt.plot(sparsities, fms); 
plt.savefig(figures_output / 'fid-')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()

plt.figure(figsize=(4,4)); 
plt.plot(sparsities, threshes); 
plt.savefig(figures_output / 'Threshholds')
plt.show()
