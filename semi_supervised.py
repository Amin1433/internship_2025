import argparse
import os
import sys
import time
import torch
from torch import nn
from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from modules.ms_aagcn import ms_aagcn
from modules.trainer_patience import Trainer
from modules.evaluator import * 
from modules.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split, Subset
from modules.semi.relabelling_batch import label_unk, recreate_datasets
# from modules.semi.relabelling import label_unk, recreate_datasets
# from modules.semi.relabelling import label_unk, recreate_datasets
from collections import OrderedDict
import random
import numpy as np

# --- Hyperparameters ---

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False # Corresponds to --extended
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = False # Corresponds to --pre_transform
NUM_EPOCHS = 150
NUM_PHASES = 10
BATCH_SIZE = 64
RANDOM_SEED = 42
LOADING_PRETRAINED = True # Whether to load a pretrained model for semi-supervised training


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

    total_dataset = NTU_Dataset(
        root=DATASET_PATH, 
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
        pre_transform=pre_transformer,
        modality=MODALITY, 
        benchmark=BENCHMARK, 
        part="train",
        extended=USE_EXTENDED_DATASET
    )

    true_labels = [int(total_dataset.y[i]) for i in range(len(total_dataset))]

    
    dataset, unlabeled_set = splitting_prop(total_dataset, proportion=proportion)

    
    for i in unlabeled_set.indices:
        total_dataset.y[i] = -1

    # Mélange aléatoire des indices
    total_labeled_indices = dataset.indices 

    split_point = int(len(total_labeled_indices) * 0.7)

    train_set_labeled_indices = total_labeled_indices[:split_point]
    val_set_indices = total_labeled_indices[split_point:]

    train_set = Subset(total_dataset, train_set_labeled_indices)
    val_set = Subset(total_dataset, val_set_indices)
    protected_indices = train_set_labeled_indices 

    train_set_indices = train_set_labeled_indices + unlabeled_set.indices
    train_dataset = Subset(total_dataset, train_set_indices)




    print(f"Using {proportion*100:.2f}% of the dataset labeled.")
    prefix = f"semi_supervised_{proportion*100:.0f}%"



    if rank == 0:
        print(f"Training model on {os.path.basename(DATASET_PATH.rstrip('/'))} dataset...")
        print(f"Number of classes: {120 if USE_EXTENDED_DATASET else 60}") 
        print(f"Benchmark: {BENCHMARK}") 
        print(f"Modality: {MODALITY}")
        print(f"{len(train_set)}/{len(val_set)} as train/val split")

    num_classes = 120 if USE_EXTENDED_DATASET else 60
    
    if LOADING_PRETRAINED:
        print(f"[INFO] Loading pretrained model for semi-supervised training with {proportion*100:.0f}% labeled data")
        # ----------- Load pretrained supervised model -----------
        supervised_model_path = os.path.join("models", f"supervised_{proportion*100:.0f}%.pt")
        print(f"[INFO] Loading pretrained model: {supervised_model_path}")
        model = ms_aagcn(num_class=num_classes).to(rank)
        model_sd, optimizer_sd, loss_function_sd, history_sd, batch_size_sd = load_model(supervised_model_path)
        cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
        model.load_state_dict(cleaned_sd, strict=False)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        starting_phase = 2
        if rank == 0:
            src_log_dir = os.path.join("logs", f"supervised_{proportion*100:.0f}%")
            dst_log_dir = os.path.join("logs", f"{prefix}", "phase1")
            os.makedirs(dst_log_dir, exist_ok=True)

            for metric in ["train_loss", "train_acc", "val_loss", "val_acc"]:
                src_file = os.path.join(src_log_dir, f"{metric}.npy")
                dst_file = os.path.join(dst_log_dir, f"{metric}.npy")
                if os.path.exists(src_file):
                    np.save(dst_file, np.load(src_file))  # ensure correct save even if path incompatible
                    print(f"[INFO] Copied {metric}.npy to {dst_file}")
                else:
                    print(f"[WARNING] {src_file} not found. Skipping.")

        # ----------- Create model -----------
    else:
        print(f"[INFO] Creating new model for semi-supervised trainning")
        model = ms_aagcn(num_class=num_classes).to(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        starting_phase = 1

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0001)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      batch_size=BATCH_SIZE, 
                      rank=rank,
                      world_size=world_size,
                      device=device,
                      patience=12
                      )

    try:

        start_time = time.time()
        
        if LOADING_PRETRAINED:
            label_unk(model=model, 
                      dataset=train_dataset, 
                      phase=1, 
                      device=device, 
                      protected_indices=protected_indices, 
                      prefix=prefix, 
                      true_labels=true_labels)

            train_set, unlabeled_set = recreate_datasets(train_dataset)

            print(f"Updated train dataset (phase {1}): {len(train_set)} labeled samples, {len(unlabeled_set)} unlabeled samples, {len(train_dataset)} total samples")

        for phase in range(starting_phase,NUM_PHASES+1):
            print(f"\nPhase {phase}/{NUM_PHASES}")

            history = trainer.train(train_set, val_set, NUM_EPOCHS, prefix=prefix + f"/phase{phase}")

            if rank == 0:
                model_path = os.path.join("models", f"{prefix}_phase_{phase}.pt")
                save_model(model_path, model, optimizer, loss_function, history, BATCH_SIZE)
                print(f"[INFO] Model saved: {model_path}")

            label_unk(model=model, 
                      dataset=train_dataset, 
                      phase=phase, 
                      device=device, 
                      protected_indices=protected_indices, 
                      prefix=prefix, 
                      true_labels=true_labels)

            train_set, unlabeled_set = recreate_datasets(train_dataset)

            print(f"Updated train dataset (phase {phase + 1}): {len(train_dataset)} samples")

        
        end_time = time.time()


        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)
            
            # model_path = os.path.join("models", f"{prefix}.pt")
            # save_model(model_path, model, optimizer, loss_function, history, BATCH_SIZE)

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
    
    parser.add_argument("--proportion", type=float, default=0.01, help="Proportion of the dataset to use for training (0.0 to 1.0)")


    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    """
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    """
    

    handle_train_parser(args.proportion)