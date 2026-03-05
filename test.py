import pytest
import numpy
from pathlib import Path
from loguru import logger
import pickle
import pandas as pd
import torch
import numpy as np
from GNNExplainer import GNNExplainer
from ML_utils import yield_subgraphs

WINDOW = 10_000

test_f = Path('interm/unsw_nb15_processed_test.csv')
model_dir = Path('interm/runs/EGraphSAGE_anomdetection_UNSW_graphsage_20260303_033347/best_model.pkl')
logger.info('loading data')
proto_test_flows = pd.read_csv(test_f, nrows=100_000)

logger.info('loading model ..')
with open(model_dir, 'rb') as f:
	model = pickle.load(f)

windows = [SG for SG in yield_subgraphs(proto_test_flows, window=WINDOW, linegraph=False)]
G = windows[2]

gnne = GNNExplainer(
	feature_bank= proto_test_flows[proto_test_flows.columns.difference(['src', 'dst', 'Attack'])].values, # for sampling noise
	edge_reg_weight=0.1,
	feature_reg_weight=0.1,
	l1_norm_weight=0.1
)

def test_gnne_eval_window():
	edge_mask = torch.zeros(G.edge_attr.shape[0])
	feature_mask = torch.zeros(G.edge_attr.shape[1])

	metrics = gnne.evaluate_window(
		model, 
		G, 
		edge_mask, 
		feature_mask, 
		prediction_threshhold=0.5, 
		sparsities=np.arange(0,1,0.05)
	)

	assert metrics[0.0]['fp'] > 0
	assert metrics[0.0]['fn'] == 1.0

	edge_mask = torch.ones(G.edge_attr.shape[0])
	feature_mask = torch.ones(G.edge_attr.shape[1])

	metrics = gnne.evaluate_window(
		model, 
		G, 
		edge_mask, 
		feature_mask, 
		prediction_threshhold=0.5, 
		sparsities=np.arange(0,1,0.05)
	)

	assert metrics[0.0]['fp'] == 0.0
	assert metrics[0.0]['fn'] > 0.0

