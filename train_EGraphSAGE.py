import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, auc
import torch
from tqdm import tqdm
from torch_geometric.nn.models.basic_gnn import GraphSAGE
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, pickle
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from EGraphSAGE import EGraphSAGE
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr",                 default=0.01, type=float)
parser.add_argument("--n-windows",          default=None, type=float)
parser.add_argument("--window",             default=None, type=int)
parser.add_argument("--pos-weight",         default=None, type=float)
parser.add_argument("--epochs",             default=50, type=int)
parser.add_argument("--layer-size",         default=256, type=int)
parser.add_argument("--num_layers",         default=1, type=int)
parser.add_argument("--train-flows",        default="interm/unsw_nb15_processed_train.csv")
parser.add_argument("--test-flows",         default="interm/unsw_nb15_processed_test.csv")
parser.add_argument("--device",             default="cpu")
parser.add_argument("--run-directory",      default="interm/runs")
args = parser.parse_args()
logger.info(f"using args: {args}")
run_ID = f"EGraphSAGE_AD_{Path(args.train_flows).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
channels = [args.layer_size] * args.num_layers

logger.info(f"Using data from {args.train_flows}, {args.test_flows}...", c="blue")

# experimental directory
exp_dir = Path(args.run_directory) / run_ID
exp_dir.mkdir(parents=True, exist_ok=True)
log_dir = exp_dir / "run.log"
log_dir.touch()
logger.add(log_dir)
writer = SummaryWriter(log_dir=exp_dir)

one_row = pd.read_csv(args.train_flows, nrows=1)
features = len(one_row.columns) - 3 # src dst Attack
model_kwargs = {
    "layer_sizes": channels,
    "flow_features": features,
    "output_dim": 1,
}
model = EGraphSAGE(**model_kwargs)
logger.info(f"MODEL SUMMARY: ", model)
for layer in model.layers:
    logger.info(layer)



# meta data
experiment_summary = {
    "description": f"EgraphSAGE binary anomoly detection with args: {args}",
    "model_kwargs": model_kwargs,
    "args": args,
    "test_df_location": args.test_flows,
    "train_df_location": args.train_flows,
}

# training and testing
if args.window is not None:
    logger.warning(f'using windowed mode for training and testing = [{args.window} flows per window]')
        
    def flow_generator(flows_path, window):
        for i, chunk in enumerate(pd.read_csv(flows_path, chunksize=window)):
            if args.n_windows is not None and i == args.n_windows-1:
                logger.info(f'max windows reached: {args.n_windows}')
                break

            # re encode attack for anomoly detection
            chunk.Attack = torch.Tensor(
                (chunk["Attack"] != "Benign").astype(float).values
            ).float()
            yield chunk

    train_gen = flow_generator(args.train_flows, args.window)
    test_gen = flow_generator(args.test_flows, args.window)

    # pos weight must be provided else none
    if not args.pos_weight:
        mal_weight = None
        logger.info(f'pos weigh not given, and cannot be inferred')
    else:
        mal_weight = args.pos_weight

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([mal_weight]))

else:
    train_flows = pd.read_csv(args.train_flows)
    train_flows.Attack = torch.Tensor(
                (train_flows["Attack"] != "Benign").astype(float).values
            ).float()
    train_gen = [train_flows]

    test_flows = pd.read_csv(args.test_flows)
    test_flows.Attack = torch.Tensor(
                (test_flows["Attack"] != "Benign").astype(float).values
            ).float()
    test_gen = [test_flows]

    # if training on full datagrame, pos weigtht can be infered
    y = train_flows.Attack
    if not args.pos_weight:
        mal_weight = (y == 0).sum() / (y == 1).sum()
        logger.info(f'infering pos weigh: {mal_weight}')
    else:
        mal_weight = args.pos_weight

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([mal_weight]))


model.train_flows(
    criterion=criterion,
    epochs=args.epochs,
    train_flow_generator=train_gen,
    test_flow_generator=test_gen,
    experiment_summary=experiment_summary,
    experimental_directory=exp_dir,
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
)