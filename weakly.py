import os
import sys
import time
import torch
from torch import nn
from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from modules.ms_aagcn import ms_aagcn, extract_features
from modules.trainer import Trainer
from modules.evaluator import *
from modules.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split
from collections import OrderedDict
import random
import pickle
import argparse
import numpy as np
from modules.MIL.MILBagDataset import *
from modules.MIL.GraphMIL import *
from modules.MIL.utils import *
from torch_geometric.loader import DataLoader as PyGDataLoader

# --- Hyperparameters ---
DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = True
NUM_EPOCHS_PRE_MODEL = 75
NUM_EPOCHS_MIL_MODEL = 300
BATCH_SIZE = 64
BATCH_SIZE_MIL = 128
RANDOM_SEED = 42
MIL_MODEL_HIDDEN_DIM = 512
LEARNING_RATE_MIL = 5e-5
DROPOUT_RATE_MIL = 0.7
LOADING_PRETRAINED_PRE_MODEL = True
LOADING_FEATURES = False

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9999'
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def handle_train_ddp(rank, world_size, proportion):
    setup_ddp(rank, world_size)
    set_seed(RANDOM_SEED + rank)
    device = torch.device(f'cuda:{rank}')

    pre_transformer = NTU_Dataset.__nturgbd_pre_transformer__ if ENABLE_PRE_TRANSFORM else None
    dataset = NTU_Dataset(
        root=DATASET_PATH,
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
        pre_transform=pre_transformer,
        modality=MODALITY,
        benchmark=BENCHMARK,
        part="train",
        extended=USE_EXTENDED_DATASET
    )
    
    # Split dataset into labeled and unlabeled subsets
    dataset1, dataset2 = splitting_prop(dataset, proportion)
    train_set1, val_set1 = splitting_prop(dataset1, 0.7)
    train_set2, val_set2 = splitting_prop(dataset2, 0.7)

    num_classes = 60
    pre_model_path = os.path.join("models", f"supervised_{proportion*100:.0f}%.pt")
    pre_model = ms_aagcn(num_class=num_classes).to(device)

    if LOADING_PRETRAINED_PRE_MODEL:
        try:
            model_sd, _, _, _, _ = load_model(pre_model_path)
            # Remove 'module.' prefix if saved from DDP
            cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
            pre_model.load_state_dict(cleaned_sd, strict=False)
        except FileNotFoundError:
            if rank == 0:
                print(f"ERROR: Pre-trained model not found at {pre_model_path}. Exiting.", file=sys.stderr)
            sys.exit(1)

    pre_model = torch.nn.parallel.DistributedDataParallel(pre_model, device_ids=[rank])
    
    if rank == 0:
        print("Generating bags from dataset...")

    train_bag = data_generation(train_set2)
    val_bag = data_generation(val_set2)

    if rank == 0:
        train_total_bags = len(train_bag[1])
        if train_total_bags > 0:
            train_positive_bags = sum(train_bag[1].values())
            print(f"Train Bags: {train_positive_bags}/{train_total_bags} positives ({train_positive_bags/train_total_bags:.2%})")

        val_total_bags = len(val_bag[1])
        if val_total_bags > 0:
            val_positive_bags = sum(val_bag[1].values())
            print(f"Val Bags:   {val_positive_bags}/{val_total_bags} positives ({val_positive_bags/val_total_bags:.2%})")

    features_path = os.path.join("models/features", f"features_{proportion*100:.0f}.pkl")
    if LOADING_FEATURES and os.path.exists(features_path):
        with open(features_path, "rb") as f:
            feature_dict = pickle.load(f)
    else:
        feature_dict = extract_features(pre_model, dataset2, device)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        with open(features_path, "wb") as f:
            pickle.dump(feature_dict, f)

    train_dataset = BagGraphDataset(train_bag[0], train_bag[1], feature_dict)
    val_dataset = BagGraphDataset(val_bag[0], val_bag[1], feature_dict)
    train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE_MIL, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=BATCH_SIZE_MIL, shuffle=False)

    input_dim = next(iter(feature_dict.values())).shape[-1]
    mil_model = MIL_GCN_Attention(input_dim=input_dim, hidden_dim=MIL_MODEL_HIDDEN_DIM, dropout_rate=DROPOUT_RATE_MIL).to(device)
    
    mil_log_prefix = f"MIL_{proportion*100:.0f}%"
    train_mil_model(mil_model, train_loader, val_loader, device, n_epochs=NUM_EPOCHS_MIL_MODEL, lr=LEARNING_RATE_MIL, prefix=mil_log_prefix)

    if rank == 0:
        mil_model_path = os.path.join("models", f"weakly_mil_model_{proportion*100:.0f}.pt")
        torch.save(mil_model.state_dict(), mil_model_path)
        print(f"MIL model saved to {mil_model_path}")

    cleanup_ddp()

def handle_train_main(proportion):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        sys.exit("No CUDA devices found. This script requires GPUs for DistributedDataParallel.")
    mp.spawn(handle_train_ddp, args=(world_size, proportion), nprocs=world_size, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proportion", type=float, default=0.1, help="Proportion of the dataset to use for supervision.")
    args = parser.parse_args()
    handle_train_main(args.proportion)
