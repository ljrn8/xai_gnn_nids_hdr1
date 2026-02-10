import pandas as pd
import torch
from preprocessing import graph_encode
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch_geometric.data import Data
import torch.nn.functional as F
import matplotlib.pyplot as plt, pickle
from training import train_graph, eval_graph, epoch
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import torch.nn as nn


# -- Script ---

device = 'cpu'

print('Loading data...')
train_flows = pd.read_csv("interm/unsw_nb15_processed_train.csv")
test_flows = pd.read_csv("interm/unsw_nb15_processed_test.csv")
flows = pd.concat([train_flows, test_flows], ignore_index=True)

attacks = np.unique(flows.Attack)
attacks = [a for a in attacks if a != 'Benign']
output_dir = Path(f"interm/metrics/graphSAGE_unsw_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
for attack in attacks:

    # binary classification for each class
    train_cp = train_flows.copy()
    train_cp.Attack = (train_cp.Attack == attack).astype(int)
    test_cp = test_flows.copy()
    test_cp.Attack = (test_cp.Attack == attack).astype(int)

    # weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=[np.sum(train_cp.Attack == 0).astype(float), 
                                            np.sum(train_cp.Attack == 1).astype(float)])

    test_cp.Attack = test_cp.Attack.astype(float)
    train_cp.Attack = train_cp.Attack.astype(float)      

    model = GraphSAGE(
        49,
        hidden_channels=256,
        out_channels=2,
        num_layers=2,
    ).to(device)

    outputs = {
        'train_losses': [],
        'test_losses': [],
        'description': f'Binary classification on {attack} - GraphSAGE with linegraph on UNSW-NB15 dataset'
    }

    for epc in tqdm(range(1, 5+1)):
        train_losses, avg_train_loss, train_ys, train_ypreds = epoch(model, train_cp, 
                                                                     loss_fn=criterion)
        test_losses, avg_test_loss, test_ys, test_ypreds = epoch(model, test_cp, 
                                                                 evaluate_instead=True, 
                                                                 loss_fn=criterion)
        outputs['train_losses'].append(avg_train_loss)
        outputs['test_losses'].append(avg_test_loss)

    outputs['train predictions'] = (train_ys, train_ypreds)
    outputs['test predictions'] = (test_ys, test_ypreds)
    outputs['model'] = model.state_dict()

    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / f"{attack}_graphSAGE_unsw_train.pkl", "wb") as f:
        pickle.dump(outputs, f) 