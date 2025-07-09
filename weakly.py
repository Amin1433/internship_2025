import argparse
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
from modules.MIL.MILBagDataset import *
from modules.MIL.GraphMIL import *
from modules.MIL.utils import *
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Dict
from modules.utils import load_model, save_model


# --- Train Parser ---

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9999'

    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def handle_train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    pre_transformer = None
    if args.pre_transform:
        pre_transformer =  NTU_Dataset.__nturgbd_pre_transformer__

    pre_transformer = NTU_Dataset.__nturgbd_pre_transformer__ if args.pre_transform else None

    dataset = NTU_Dataset(
        root=args.dataset,
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
        pre_transform=pre_transformer,
        modality=args.modality,
        benchmark=args.benchmark,
        part="train",
        extended=args.extended
    )
    print(f"Dataset size: {len(dataset)}")
    # dataset, _ = splitting_prop(dataset, 0.1)
    # print(f"Dataset size after split: {len(dataset)}")
    dataset1, dataset2 = splitting_prop(dataset, 0.1)
    train_set1, val_set1 = splitting_prop(dataset1, 0.7)
    train_set2, val_set2 = splitting_prop(dataset2, 0.7)

    print(f"Train set size: {len(train_set1)}, Validation set size: {len(val_set1)}")

    #pre_train model
    num_classes = 120 if args.extended else 60
    pre_model = ms_aagcn(num_class=num_classes).to(rank)
    pre_model = torch.nn.parallel.DistributedDataParallel(pre_model, device_ids=[rank])

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pre_model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0001)


    trainer = Trainer(model=pre_model,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        batch_size=args.batch_size,
                        rank=rank,
                        world_size=world_size,
                        device=device
                        )

    try:
        # loading = True

        # model_path = os.path.join("models", f"weakly_premodel_ntu_rgbd{120 if args.extended else ''}_{args.benchmark}_{args.modality}.pt")

        # if loading:
        #     if os.path.exists(model_path):
        #         print(f"Loading pretrained model from {model_path}...", file=sys.stderr)
        #         pre_model, optimizer, loss_function, history, batch_size = load_model(model_path)
        #         trainer.model.load_state_dict(pre_model.state_dict())
        #         print("Model loaded successfully.", file=sys.stderr)
        #     else:
        #         print(f"Model path {model_path} does not exist!", file=sys.stderr)
        #         sys.exit(1)
        # else:
        #     start_time = time.time()
        #     history = trainer.train(train_set1, val_set1, num_epoch=40)
        #     end_time = time.time()
        start_time = time.time()
        history = trainer.train(train_set1, val_set1, 50)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)
            pre_model_path = os.path.join("models", f"weakly_premodel_ntu_rgbd{120 if args.extended else ''}_{args.benchmark}_{args.modality}.pt")
            save_model(pre_model_path, pre_model, optimizer, loss_function, history, args.batch_size)

            # Bag generation and MIL training
            print("Generating bags from dataset...")
            train_bag = data_generation(train_set2)  # bags, bag_labels
            val_bag = data_generation(val_set2)

            print("Extracting features for all instances...")
            feature_dict = extract_features(pre_model, dataset, device)

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

            mil_model = MIL_GCN_Attention(input_dim=input_dim, hidden_dim=256).to(device)

            # training on MIL model
            train_mil_model(mil_model, train_loader, val_loader, device, n_epochs=args.n_epochs)

            
            torch.save(mil_model.state_dict(), "modle")
            mil_model_path = os.path.join("models", f"weakly_mil_model.pt")
            print(f"MIL model saved to {mil_model_path}")

    finally:
        cleanup_ddp()



def handle_train_parser(args):
    world_size = torch.cuda.device_count()
    print(f"Prepare training process on {world_size} GPU")
    mp.spawn(handle_train_ddp, args=(world_size, args), nprocs=world_size, join=True)



# ==================== CLI Interface ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-streams Attention Adaptive model for Human's Action Recognition")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")

    mode_parser = parser.add_subparsers(dest="mode", required=True, help="select mode: process | train | eval | ensemble")

    # --- Train Mode ---
    train_parser = mode_parser.add_parser("train", help="Train Model")

    train_parser.add_argument("--dataset", type=str, default="data/nturgb+d_skeletons/", help="path towards dataset")
    train_parser.add_argument("--extended", action="store_true", help="use NTU RGB+D 120 dataset")
    train_parser.add_argument("--modality", default="joint", choices=["joint", "bone"], help="modality: joint | bone")
    train_parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview", "xsetup"], help="benchmark: xsub | xview | xsetup")
    train_parser.add_argument("--validation", action="store_true", help="enable validation") # TO ADD
    train_parser.add_argument("--pre_transform", action="store_true", help="authorize pre-transformation")
    train_parser.add_argument("--n_epochs", type=int, default=50, help="number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    train_parser.add_argument("--summary", action="store_true", help="summary of model") 

    train_parser.set_defaults(func=handle_train_parser)


    # --- CLI Interface ---

    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    """
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    """ 

    # print(f"Using {device}", file=sys.stderr)

    args.func(args)