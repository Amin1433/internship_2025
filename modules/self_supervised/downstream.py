import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split
import random
import numpy as np

from data_generator.ntu_data import EDGE_INDEX, NTU_Dataset
from modules.self_supervised.tower import MsAAGCN
from modules.self_supervised.siamese import (
    SiameseDataLoader, SiameseDataset, SiameseModel, 
    triplet_loss, DataAugmentations
)
from modules.self_supervised.trainer_ssl import SSLTrainer
from modules.utils import splitting_prop
from modules.self_supervised.utils import ssl_save_model

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False
MODALITY = "joint"
BENCHMARK = "xsub"
NUM_EPOCHS = 500
BATCH_SIZE = 24
RANDOM_SEED = 42
LATENT_SIZE = 256
PATIENCE = 25

NUM_JOINT = 25
NUM_SKELETON = 2
IN_CHANNELS = 3

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9991'
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def handle_train_ddp(rank, world_size, proportion, load_supervised_model):
    setup_ddp(rank, world_size)
    set_seed(RANDOM_SEED + rank)
    device = torch.device(f'cuda:{rank}')

    total_dataset = NTU_Dataset(root=DATASET_PATH, modality=MODALITY, benchmark=BENCHMARK, part="train")
    _, unlabeled_set = splitting_prop(total_dataset, proportion=proportion)
    
    train_size = int(0.9 * len(unlabeled_set))
    val_size = len(unlabeled_set) - train_size
    unlabeled_train_set, unlabeled_val_set = random_split(unlabeled_set, [train_size, val_size])

    if rank == 0:
        print("--- Upstream SSL Pre-training ---")
        print(f"Unlabeled subset size: {len(unlabeled_set)} ({100 - proportion * 100:.2f}% of dataset)")
        print(f"  -> Train set: {len(unlabeled_train_set)} samples")
        print(f"  -> Validation set: {len(unlabeled_val_set)} samples")

    augmentations = DataAugmentations()
    training_dataset = SiameseDataset(unlabeled_train_set, augmentations=augmentations)
    train_dataloader = SiameseDataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset = SiameseDataset(unlabeled_val_set, augmentations=augmentations)
    val_dataloader = SiameseDataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    tower_model = MsAAGCN(
        num_joint=NUM_JOINT, num_skeleton=NUM_SKELETON,
        edge_index=EDGE_INDEX, in_channels=IN_CHANNELS, latent_size=LATENT_SIZE
    ).to(rank)

    warm_start_successful = False
    if load_supervised_model:
        # Insert warm-start loading logic here if needed
        pass

    siamese_model = SiameseModel(tower_model).to(rank)
    siamese_model = torch.nn.parallel.DistributedDataParallel(siamese_model, device_ids=[rank])

    optimizer = torch.optim.Adam(siamese_model.parameters(), lr=0.001, weight_decay=0.0001)
    trainer = SSLTrainer(
        model=siamese_model, optimizer=optimizer,
        loss_function=triplet_loss, batch_size=BATCH_SIZE,
        rank=rank, world_size=world_size, device=device, patience=PATIENCE
    )

    try:
        start_time = time.time()
        ssl_prefix = f"ssl_{proportion * 100:.0f}%_warm_start" if warm_start_successful else f"ssl_{proportion * 100:.0f}%_from_scratch"
        history = trainer.train(train_dataloader, val_dataloader, NUM_EPOCHS, prefix=ssl_prefix)
        end_time = time.time()

        if rank == 0:
            print(f"Training completed in {end_time - start_time:.2f} seconds")
            print(f"Model saved to logs/{ssl_prefix}/best_siamese_tower.pt")
    finally:
        cleanup_ddp()

def handle_train_parser(args):
    world_size = torch.cuda.device_count()
    mp.spawn(handle_train_ddp, args=(world_size, args.proportion, args.load_supervised_model), nprocs=world_size, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upstream SSL Pre-training")
    parser.add_argument(
        "--proportion", type=float, default=0.1,
        help="Proportion of dataset considered 'labeled' and excluded from SSL training (e.g., 0.1)"
    )
    parser.add_argument(
        "--load-supervised-model", action="store_true",
        help="Warm start from a supervised model checkpoint"
    )
    args = parser.parse_args()
    handle_train_parser(args)
