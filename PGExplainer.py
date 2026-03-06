from ML_utils import yield_subgraphs, fidelities
import pandas as pd
from torch import nn, threshold
import torch
from tqdm import tqdm
import sys
import pickle
from loguru import logger
from pathlib import Path
from EGraphSAGE import EGraphSAGE
import numpy as np
from pprint import pprint
from datetime import datetime


device = 'cpu'


class PGExplainer(nn.Module):
	"""PGExplainer adapted for link prediction on network flows
	Current tenatic aspects of the impl include
	1. 1 MLP explainer for each graph window. 
	"""

	MODEL_PRED_THRESHOLD = 0.5
	IMPIRICAL_REPARAMATERIZATION = True
	TAU = 0.01

	def __init__(self, full_edge_attr, 
			  feature_reg_weight=0.1, 
			  edge_entr_reg=0.1, 
			  edge_sum_reg=0.1, 
			  mlp_lasso_reg=0.1,
			  hidden_mlp_layers=128, **kwargs):
		
		super().__init__(**kwargs)
		n_features = full_edge_attr.shape[1]
		self.mlp = nn.Sequential(
			nn.Linear(in_features=n_features, out_features=hidden_mlp_layers),
			nn.ReLU(),
			nn.Linear(in_features=hidden_mlp_layers, out_features=1)
		)
		self.feature_reg_weight = feature_reg_weight
		self.l1_norm_weight = edge_sum_reg
		self.edge_reg_weight = edge_entr_reg
		self.mlp_lasso_reg = mlp_lasso_reg
		self.full_edge_attr = full_edge_attr

	def elementwise_entropy(x):
		return -x * torch.log2(x) - (1-x)*torch.log2(1-x)

	def regularization(self, edge_mask):
		reg = 0
		reg += self.edge_reg_weight * torch.mean(torch.tensor(
			[PGExplainer.elementwise_entropy(e) for e in edge_mask]))
		reg += self.l1_norm_weight * torch.mean(edge_mask)
		return reg

	def sample_from_empirical(self, edge_attr):
		feature_bank = self.full_edge_attr
		n_edges, n_features = edge_attr.shape
		N = feature_bank.shape[0]
		idx = torch.randint(0, N, (n_edges, n_features))
		Z = feature_bank[idx, torch.arange(n_features).unsqueeze(0)]
		return Z  # (n_edges, n_features)

	def sample_masked_graph(self, G, mask, tau, impirical_reparamaterization=False):
		""" 
		Converts the soft edge mask to a binary concrete distribution, 
		sampled with epsilon, with strength tau. This directly multiplied into the mask, 
		though with impirical_reparaterization=True, its replaced with an impirical sample from X_E
		"""
		epsilon = torch.rand(1)[0]
		logger.debug(f'epsilon: {epsilon}')
		near_binary_mask = torch.sigmoid(
			(torch.log2(epsilon) + torch.log2(1 - epsilon) + mask) 
			/ tau) 
		
		if impirical_reparamaterization:
			Z = self.sample_from_empirical(G.edge_attr)
			masked_edge_attr = Z + ((G.edge_attr - Z) * near_binary_mask.unsqueeze(1))
		else:
			masked_edge_attr = G.edge_attr * near_binary_mask.unsqueeze(1)
		
		return masked_edge_attr

	def get_masked_prediction(self, model, G, mask):
		""" 
		Return a prediction with the mask applied through a BCD 
		and potentially impirical sampling for the 'subgraph' of the same size.
		"""
		masked_edge_attr = self.sample_masked_graph(
			G, mask, 
			tau=self.TAU, 
			impirical_reparamaterization=self.IMPIRICAL_REPARAMATERIZATION
		)
		n_nodes = G.edge_index.max().item() + 1
		masked_y_pred, _ = model.forward(
			masked_edge_attr, 
			G.edge_index,
			node_attr=torch.ones(size=(n_nodes, masked_edge_attr.shape[0])).to(device)
		)
		return masked_y_pred
		

	def appromiximate_subgraph_prediction(self, G, model, mask, samples=10):
		""" sample subgraph predictions from the fractional mask with MC estimation """
		predictions = []
		for _ in samples:
			predictions.append(self.get_masked_prediction(self, model, G, mask))

		if samples > 1:
			all_preds = torch.vstack(predictions)
			masked_y_pred_mean = torch.mean(all_preds, axis=0)
			assert len(masked_y_pred_mean) == len(predictions[0])
		else:
			masked_y_pred_mean = predictions[0]

		return masked_y_pred_mean


	def fit(self, model, test_flows, epochs,
                 window, lr=0.01, loss_f=torch.nn.BCELoss()):
		
        # need to freeze model so optimizer only touches the mask at BCE(Y, Y^)
		model.eval()
		for param in model.parameters():
			param.requires_grad = False

		# normal prediction
		with torch.no_grad():
			n_nodes = G.edge_index.max().item() + 1
			n_features = self.full_edge_attr.shape[1]
			y_pred, _ = model.forward(
				G.edge_attr, 
				G.edge_index,
				node_attr=torch.ones(size=(n_nodes, n_features)).to(device)) 

		# train MLP on window
		for i, G in enumerate(yield_subgraphs(test_flows, window, linegraph=False)):
			optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
			losses, mask_regularization = [], []

			for epc in range(1, 1 + epochs):
				mask_logits = self.mlp(G.edge_attr)
				l1_norm = sum(p.abs().sum() for p in model.parameters())
				mask = torch.sigmoid(mask_logits)

				# enforce that all predicted malicious flows are in the explanation mask
				y_pred_binary = y_pred > self.MODEL_PRED_THRESHOLD
				mask = mask - (mask + torch.ones_like(mask)) * y_pred_binary
				
				# masked prediction
				y_pred_masked = self.appromiximate_subgraph_prediction(model, G, mask)
				loss = loss_f(y_pred, y_pred_masked)

				# regularization 
				reg = self.regularization(mask)
				total_loss = loss + reg + (self.mlp_lasso_reg * l1_norm)

				total_loss.backward()
				optimizer.step()

				losses.append(loss.detach())
				mask_regularization.append(reg.detach())

				logger.info(
					f'epoch: {epc} | '
					f"av loss for mask: {np.mean(losses):.5f} | "
					f"edge mask average value: {torch.mean(mask):.5f} | "
					f"regularization penalty: {reg:.5f} | "
				)

			losses_out = torch.stack(losses).cpu().numpy()
			regularization_penalties_out = torch.stack(mask_regularization).cpu().numpy()
			yield (G, mask, losses_out, regularization_penalties_out)
