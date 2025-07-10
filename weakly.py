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
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Dict
from modules.utils import load_model, save_model


# --- Hyperparameters ---
DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = True
NUM_EPOCHS_PRE_MODEL = 75
NUM_EPOCHS_MIL_MODEL = 200
BATCH_SIZE = 32
RANDOM_SEED = 42
LOADING_PRETRAINED_PRE_MODEL = True # New hyperparameter for pre_model loading
LOADING_FEATURES = True  # Whether to load precomputed features if available


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

    pre_transformer = None
    if ENABLE_PRE_TRANSFORM:
        pre_transformer = NTU_Dataset.__nturgbd_pre_transformer__

    dataset = NTU_Dataset(
        root=DATASET_PATH,
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
        pre_transform=pre_transformer,
        modality=MODALITY,
        benchmark=BENCHMARK,
        part="train",
        extended=USE_EXTENDED_DATASET
    )
    print(f"Dataset size: {len(dataset)}")

    dataset1, dataset2 = splitting_prop(dataset, proportion)
    
    total_labeled_indices = dataset1.indices 
    split_point = int(len(total_labeled_indices) * 0.7)
    train_set_labeled_indices = total_labeled_indices[:split_point]
    val_set_indices = total_labeled_indices[split_point:]
    train_set1 = Subset(dataset, train_set_labeled_indices)
    val_set1 = Subset(dataset, val_set_indices)

    train_set2, val_set2 = splitting_prop(dataset2, 0.7)

    print(f"Train set size: {len(train_set1)}, Validation set size: {len(val_set1)}")

    num_classes = 120 if USE_EXTENDED_DATASET else 60

    # --- Pre-model loading/creation logic ---
    if LOADING_PRETRAINED_PRE_MODEL:
        print(f"Loading pretrained pre_model for semi-supervised training")
        pre_model_path = os.path.join("models", f"supervised_{proportion*100:.0f}%.pt")
        print(f"Loading pretrained pre_model from: {pre_model_path}")
        
        pre_model = ms_aagcn(num_class=num_classes).to(rank)
        
        # Load the state_dict, optimizer, loss_function, etc. from the saved file
        try:
            model_sd, optimizer_sd, loss_function_sd, history_sd, batch_size_sd = load_model(pre_model_path)
            # Clean state_dict keys if saved from a DDP wrapped model ("module." prefix)
            cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
            pre_model.load_state_dict(cleaned_sd, strict=False)
            print("Pre-model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Pre-trained model not found at {pre_model_path}. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading pre-trained model: {e}. Exiting.")
            sys.exit(1)

        pre_model = torch.nn.parallel.DistributedDataParallel(pre_model, device_ids=[rank])
        
        
        should_train_pre_model = False 
    elif not LOADING_FEATURES:
        print(f"Creating new pre_model for semi-supervised training")
        pre_model = ms_aagcn(num_class=num_classes).to(rank)
        pre_model = torch.nn.parallel.DistributedDataParallel(pre_model, device_ids=[rank])
        should_train_pre_model = True 

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pre_model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0001)

    trainer = Trainer(model=pre_model,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      batch_size=BATCH_SIZE,
                      rank=rank,
                      world_size=world_size,
                      device=device
                      )

    try:
        if should_train_pre_model:
            start_time = time.time()
            history = trainer.train(train_set1, val_set1, NUM_EPOCHS_PRE_MODEL)
            end_time = time.time()

            if rank == 0:
                print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)
                pre_model_path = os.path.join("models", f"weakly_premodel_{proportion*100:.0f}.pt")
                save_model(pre_model_path, pre_model, optimizer, loss_function, history, BATCH_SIZE)
        else:
            if rank == 0:
                print("Skipping pre-model training as it was loaded from pretrained.")


        print("Generating bags from dataset...")
        train_bag = data_generation(train_set2)
        val_bag = data_generation(val_set2)

        features_path = os.path.join("models/features", f"features_{proportion*100:.0f}.pkl")
        if LOADING_FEATURES:
            if os.path.exists(features_path):
                with open(features_path, "rb") as f:
                    feature_dict = pickle.load(f)
                print(f"Loaded features from {features_path}")
        else:
            feature_dict = extract_features(pre_model, dataset2, device)
            os.makedirs(os.path.dirname(features_path), exist_ok=True)
            with open(features_path, "wb") as f:
                pickle.dump(feature_dict, f)
            print(f"Features saved to {features_path}")

        train_dataset = BagGraphDataset(
            bags_dict=train_bag[0],
            bag_labels_dict=train_bag[1],
            feature_dict=feature_dict
        )
        val_dataset = BagGraphDataset(
            bags_dict=val_bag[0],
            bag_labels_dict=val_bag[1],
            feature_dict=feature_dict
        )

        train_loader = PyGDataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=1, shuffle=False)

        example_feat = next(iter(feature_dict.values()))
        input_dim = example_feat.shape[-1]
        # print("Proportion de bags positifs :", sum(train_bag.bag_labels.values()) / len(train_bag))
        mil_model = MIL_GCN_Attention(input_dim=input_dim, hidden_dim=256).to(device)

        train_mil_model(mil_model, train_loader, val_loader, device, n_epochs=NUM_EPOCHS_MIL_MODEL)

        torch.save(mil_model.state_dict(), "model")
        mil_model_path = os.path.join("models", f"weakly_mil_model.pt")
        print(f"MIL model saved to {mil_model_path}")

    finally:
        cleanup_ddp()


def handle_train_main(proportion):
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices found. This script is configured for DDP on GPU.", file=sys.stderr)
        sys.exit(1)
    print(f"Prepare training process on {world_size} GPU")
    mp.spawn(handle_train_ddp, args=(world_size, proportion), nprocs=world_size, join=True)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-streams Attention Adaptive model for Human's Action Recognition")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    
    parser.add_argument("--proportion", type=float, default=0.1, help="Proportion of the dataset to use for training (0.0 to 1.0)")


    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    """
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    """
    

    handle_train_main(args.proportion)