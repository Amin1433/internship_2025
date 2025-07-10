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
from torch.utils.data import random_split, Subset
from modules.semi.relabelling import label_unk, recreate_datasets
from collections import OrderedDict
import random
import numpy as np

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False # Corresponds to --extended
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = False # Corresponds to --pre_transform
NUM_EPOCHS = 100
NUM_PHASES = 2
BATCH_SIZE = 64
RANDOM_SEED = 42
LOADING_PRETRAINED = True # Whether to load a pretrained model for semi-supervised training

# ==================== CLI Interface ====================

if __name__ == "__main__":
    device = torch.device(f'cuda:{0}')

    pre_transformer = None
    
    if ENABLE_PRE_TRANSFORM:
        pre_transformer = NTU_Dataset.__nturgbd_pre_transformer__

    # total_dataset = NTU_Dataset(
    #     root=DATASET_PATH, 
    #     pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
    #     pre_transform=pre_transformer,
    #     modality=MODALITY, 
    #     benchmark=BENCHMARK, 
    #     part="train",
    #     extended=USE_EXTENDED_DATASET
    # )
    # print(len(total_dataset), "total samples in the dataset\n")
    print(f"Initial labeled data: {(2850 / 40091)*100:.2f}%")
    print(f"{(2138 / 40091)*100:.2f}% of the dataset is labeled\n")
    print(f"total labeled data: {((2850+2138)/40091)*100:.2f} samples")