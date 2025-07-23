
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
import random
import numpy as np

from data_generator.ntu_data import NTU_Dataset
from modules.ms_aagcn import ms_aagcn
from modules.trainer import Trainer
from modules.utils import splitting_prop, save_model

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False
MODALITY = "joint"
BENCHMARK = "xsub"
NUM_EPOCHS = 70
BATCH_SIZE = 32
RANDOM_SEED = 42

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
    os.environ['MASTER_PORT'] = '9992'
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def handle_train_ddp(rank, world_size, proportion, ssl_weights_path, freeze_encoder):
    setup_ddp(rank, world_size)
    set_seed(RANDOM_SEED + rank)
    device = torch.device(f'cuda:{rank}')

    total_dataset = NTU_Dataset(root=DATASET_PATH, modality=MODALITY, benchmark=BENCHMARK, part="train")
    labeled_set_for_downstream, _ = splitting_prop(total_dataset, proportion=proportion)
    
    train_size = int(0.7 * len(labeled_set_for_downstream))
    val_size = len(labeled_set_for_downstream) - train_size
    train_set, val_set = random_split(labeled_set_for_downstream, [train_size, val_size])

    if rank == 0:
        print("--- Downstream Supervised Fine-tuning ---")
        print(f"Labeled set size (proportion {proportion*100:.2f}%): {len(labeled_set_for_downstream)}")
        print(f"  -> Fine-tuning training set size: {len(train_set)}")
        print(f"  -> Fine-tuning validation set size: {len(val_set)}")

    num_classes = 120 if USE_EXTENDED_DATASET else 60
    model = ms_aagcn(num_class=num_classes, num_joint=NUM_JOINT, num_skeleton=NUM_SKELETON, in_channels=IN_CHANNELS)

    if rank == 0:
        print(f"Loading SSL pre-trained weights from: {ssl_weights_path}")
        try:
            pretrained_dict = torch.load(ssl_weights_path, map_location='cpu')
            model.load_state_dict(pretrained_dict, strict=False)
            print("SSL weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)

    if world_size > 1:
        dist.barrier()
    
    model.to(device)

    if freeze_encoder:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        if rank == 0: print("Encoder FROZEN.")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    else:
        if rank == 0: print("Full network fine-tuning.")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model = DDP(model, device_ids=[rank])
    
    loss_function = nn.CrossEntropyLoss()
    trainer = Trainer(model=model, optimizer=optimizer, loss_function=loss_function,
                      batch_size=BATCH_SIZE, rank=rank, world_size=world_size, device=device, patience=15)

    try:
        start_time = time.time()
        prefix = f"downstream_p{int(proportion*100)}_frozen" if freeze_encoder else f"downstream_p{int(proportion*100)}_full"
        history = trainer.train(train_set, val_set, NUM_EPOCHS, prefix=prefix)
        end_time = time.time()

        if rank == 0:
            print(f"Fine-tuning completed in {end_time - start_time:.2f} seconds")
            model_path = os.path.join("models", f"{prefix}_final_model.pt")
            save_model(model_path, model, optimizer, loss_function, history, BATCH_SIZE)
    finally:
        cleanup_ddp()

def handle_train_parser(args):
    prop_str = f"{args.proportion*100:.0f}"
    path_warm_start = os.path.join("logs", f"ssl_{prop_str}%_warm_start", "best_siamese_tower.pt")
    path_from_scratch = os.path.join("logs", f"ssl_{prop_str}%_from_scratch", "best_siamese_tower.pt")
    
    ssl_weights_path = None
    if os.path.exists(path_warm_start):
        ssl_weights_path = path_warm_start
    elif os.path.exists(path_from_scratch):
        ssl_weights_path = path_from_scratch

    if ssl_weights_path is None:
        print(f"FATAL ERROR: Could not find pre-trained weights for proportion {prop_str}%.")
        sys.exit(1)
    
    if args.rank == 0: print(f"Found SSL weights: {ssl_weights_path}")

    world_size = torch.cuda.device_count()
    mp.spawn(handle_train_ddp, args=(world_size, args.proportion, ssl_weights_path, args.freeze_encoder), nprocs=world_size, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream Fine-tuning from auto-detected SSL weights")
    parser.add_argument("--proportion", type=float, required=True, help="Proportion of labeled data. Ex: 0.1")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights.")

    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    handle_train_parser(args)