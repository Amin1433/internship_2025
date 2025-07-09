import argparse
import os
import sys
import time
import torch
from torch import nn
from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from modules.ms_aagcn import ms_aagcn
from modules.trainer import Trainer
from modules.evaluator import * 
from modules.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split
from collections import OrderedDict
import random
import numpy as np

# --- Hyperparameters ---

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False # Corresponds to --extended
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = False # Corresponds to --pre_transform
NUM_EPOCHS = 2
BATCH_SIZE = 80
RANDOM_SEED = 42

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Train Parser ---

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9991'

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

    dataset = NTU_Dataset(root=DATASET_PATH, 
                          pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                          pre_transform=pre_transformer,
                          modality=MODALITY, 
                          benchmark=BENCHMARK, 
                          part="train",
                          extended=USE_EXTENDED_DATASET
                          )

    # Using a proportion of the dataset
    dataset, _ = splitting_prop(dataset, proportion=proportion)
    print(f"Using {proportion*100:.2f}% of the dataset for training.")
    prefix = f"supervised_{proportion*100:.0f}%"



    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = total_len - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    if rank == 0:
        print(f"Training model on {os.path.basename(DATASET_PATH.rstrip('/'))} dataset...")
        print(f"Number of classes: {120 if USE_EXTENDED_DATASET else 60}") 
        print(f"Benchmark: {BENCHMARK}") 
        print(f"Modality: {MODALITY}")
        print(f"{len(train_set)}/{len(val_set)} as train/val split")

    num_classes = 120 if USE_EXTENDED_DATASET else 60
    model = ms_aagcn(num_class=num_classes).to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0001)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      batch_size=BATCH_SIZE, 
                      rank=rank,
                      world_size=world_size,
                      device=device
                      )


    try:
        start_time = time.time()

        history = trainer.train(train_set, val_set, NUM_EPOCHS, prefix=prefix)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)
            
            model_path = os.path.join("models", f"{prefix}.pt")
            save_model(model_path, model, optimizer, loss_function, history, BATCH_SIZE)

    finally:
        cleanup_ddp()

def handle_train_parser(proportion):
    world_size = torch.cuda.device_count()
    print(f"Prepare training process on {world_size} GPU")
    mp.spawn(handle_train_ddp, args=(world_size, proportion), nprocs=world_size, join=True)


# ==================== CLI Interface ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-streams Attention Adaptive model for Human's Action Recognition")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    
    parser.add_argument("--proportion", type=float, default=1.0, help="Proportion of the dataset to use for training (0.0 to 1.0)")


    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    """
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    """
    

    handle_train_parser(args.proportion)