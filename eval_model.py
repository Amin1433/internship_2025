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
from collections import OrderedDict
import random
import re 
import numpy as np
from modules.evaluator import Evaluator


args = {
    'dataset': 'data/NTU-RGB-D', 
    'modality': 'joint',        
    'benchmark': 'xsub',
    'extended': False
}

def evaluate_split(split, model_name, model_path, log_file):
    print(f"\n==> Evaluating split: {split}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = NTU_Dataset(
        root=args['dataset'],
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
        pre_transform=NTU_Dataset.__nturgbd_pre_transformer__,
        modality=args['modality'],
        benchmark=args['benchmark'],
        part=split,
        extended=args['extended']
    )

    model = ms_aagcn().to(device)
    model_sd, _, _, _, batch_size = load_model(model_path)
    cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
    model.load_state_dict(cleaned_sd, strict=False)

    evaluator = Evaluator(model=model, batch_size=batch_size)
    evaluation = evaluator.evaluate(dataset=dataset, topk=(1, 5), display=False)

    top1 = evaluation['top1_accuracy'] * 100
    top5 = evaluation['top5_accuracy'] * 100

    print(f"Top-1 Accuracy ({split}): {top1:.2f}%")
    print(f"Top-5 Accuracy ({split}): {top5:.2f}%")

    with open(log_file, 'a') as f:
        f.write(f"[{split.upper()}] Top-1 Accuracy: {top1:.2f}%\n")
        f.write(f"[{split.upper()}] Top-5 Accuracy: {top5:.2f}%\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name.startswith("semi_supervised"):
        model_path = f"models/{model_name}.pt" 
        
        log_base_name = re.sub(r'_phase_\d+$', '', model_name) 
        log_dir = f"logs/{log_base_name}" 
    else:
        model_path = f"models/{model_name}%.pt" 
        log_dir = f"logs/{model_name}%"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evaluation.txt")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    with open(log_file, 'w') as f:
        f.write(f"Evaluation of model: {model_name}\n")

    for split in ['train', 'eval']:
        evaluate_split(split, model_name, model_path, log_file)

if __name__ == "__main__":
    main()
