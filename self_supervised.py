import argparse
import os
import sys
import time
import torch
from torch import nn
from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from modules.self_supervised.tower import MsAAGCN
from modules.self_supervised.siamese import SiameseDataLoader, SiameseDataset, SiameseModel, triplet_loss
from modules.self_supervised.trainer_ssl import SSLTrainer
from modules.utils import splitting_prop # Garde splitting_prop de l'ancien utils
from modules.self_supervised.utils import ssl_save_model # Importe la nouvelle fonction de sauvegarde SSL
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split, Subset
from collections import OrderedDict
import random
import numpy as np

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = False
NUM_EPOCHS = 10
BATCH_SIZE = 24
RANDOM_SEED = 42
LATENT_SIZE = 256

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
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def handle_train_ddp(rank, world_size, proportion, load_supervised_model):
    setup_ddp(rank, world_size)
    set_seed(RANDOM_SEED + rank)

    device = torch.device(f'cuda:{rank}')

    pre_transformer = None
    if ENABLE_PRE_TRANSFORM:
        pre_transformer = NTU_Dataset.__nturgbd_pre_transformer__

    total_dataset = NTU_Dataset(root=DATASET_PATH,
                                pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                                pre_transform=pre_transformer,
                                modality=MODALITY,
                                benchmark=BENCHMARK,
                                part="train",
                                extended=USE_EXTENDED_DATASET
                                )

    for i in range(len(total_dataset)):
        total_dataset.y[i] = total_dataset.y[i] - 1


    dataset, unlabeled_set = splitting_prop(total_dataset, proportion=proportion)

    total_labeled_indices = dataset.indices
    random.shuffle(total_labeled_indices)

    split_point = int(len(total_labeled_indices) * 0.7)
    train_set_labeled_indices = total_labeled_indices[:split_point]
    val_set_indices = total_labeled_indices[split_point:]
    
    train_set = Subset(total_dataset, train_set_labeled_indices)
    val_set = Subset(total_dataset, val_set_indices)

    if rank == 0:
        print(f"Training SSL model on {os.path.basename(DATASET_PATH.rstrip('/'))} dataset...")
        print(f"Benchmark: {BENCHMARK}")
        print(f"Modality: {MODALITY}")
        print(f"Total labeled samples (proportion {proportion*100:.2f}%): {len(dataset)}")
        print(f"Total unlabeled samples: {len(unlabeled_set)}")
        print(f"Train labeled subset size: {len(train_set)}")
        print(f"Validation labeled subset size: {len(val_set)}")

    num_classes = 120 if USE_EXTENDED_DATASET else 60
    train_samples_by_class = [[] for _ in range(num_classes)]
    for idx in train_set.indices:
        data_point = total_dataset[idx] # Assuming data_point.y is already 0-indexed now
        label = data_point.y.item()
        if 0 <= label < num_classes:
            # data_point.x is already a tensor (C, T, V, M)
            train_samples_by_class[label].append(data_point.x)

    # This should now be a list of lists of tensors (not TensorDataset objects)
    training_data_per_class = [samples for samples in train_samples_by_class if samples]
    class_count_for_siamese_train = len(training_data_per_class)

    # Pass this list of lists of tensors directly to SiameseDataset
    training_dataset_for_siamese = SiameseDataset(training_data_per_class)
    train_dataloader = SiameseDataLoader(
        training_dataset_for_siamese, batch_size=BATCH_SIZE, class_count=class_count_for_siamese_train, shuffle=True
    )

    val_samples_by_class = [[] for _ in range(num_classes)]
    for idx in val_set.indices:
        data_point = total_dataset[idx]
        label = data_point.y.item()
        if 0 <= label < num_classes:
            val_samples_by_class[label].append(data_point.x)

    # This should now be a list of lists of tensors (not TensorDataset objects)
    val_data_per_class = [samples for samples in val_samples_by_class if samples]
    class_count_for_siamese_val = len(val_data_per_class)

    # Pass this list of lists of tensors directly to SiameseDataset
    validation_dataset_for_siamese = SiameseDataset(val_data_per_class)
    test_dataloader = SiameseDataLoader( # Renamed from 'val_dataloader' in your error to 'test_dataloader' in code
        validation_dataset_for_siamese, batch_size=BATCH_SIZE, class_count=class_count_for_siamese_val, shuffle=False
    )

    tower_model = MsAAGCN(
        num_joint=NUM_JOINT,
        num_skeleton=NUM_SKELETON,
        edge_index=EDGE_INDEX,
        in_channels=IN_CHANNELS,
        latent_size=LATENT_SIZE
    ).to(rank)

    warm_start_successful = False
    if load_supervised_model:
        supervised_model_path = os.path.join("models", f"supervised_{proportion*100:.0f}%.pt")

        if os.path.exists(supervised_model_path) and rank == 0:
            print(f"Attempting to load supervised pre-trained model from {supervised_model_path}...")
            try:
                # Charger le dictionnaire complet sauvegardé par l'ancienne `save_model`
                checkpoint = torch.load(supervised_model_path, map_location=device)
                
                # Récupérer le state_dict du modèle depuis la clé 'model'
                loaded_full_state_dict = checkpoint['model'] 
                
                # Créer un state_dict pour votre tower_model actuelle pour la comparaison
                model_state_dict = tower_model.state_dict()
                
                # Filtrer les clés du state_dict chargé pour qu'elles correspondent à la tower_model
                pretrained_dict = {
                    k: v for k, v in loaded_full_state_dict.items() 
                    if k.replace('module.', '') in model_state_dict and 
                    model_state_dict[k.replace('module.', '')].shape == v.shape
                }
                
                # Mettre à jour les clés en retirant 'module.'
                new_pretrained_dict = OrderedDict()
                for k, v in pretrained_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_pretrained_dict[name] = v

                missing_keys, unexpected_keys = tower_model.load_state_dict(new_pretrained_dict, strict=False)

                if missing_keys:
                    print(f"Missing keys in tower_model after loading supervised: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in loaded supervised model not in tower_model: {unexpected_keys}")

                print("Supervised model weights (encoder part) loaded successfully into the tower_model.")
                warm_start_successful = True
            except Exception as e:
                print(f"Error loading supervised model: {e}. Starting SSL training with randomly initialized weights.")
                if world_size > 1:
                    dist.barrier()
            finally:
                if world_size > 1:
                    dist.barrier()
        elif rank == 0:
            print(f"Supervised model not found at {supervised_model_path}. Starting SSL training with randomly initialized weights.")
            if world_size > 1:
                dist.barrier()
        else:
            if world_size > 1:
                dist.barrier()

    siamese_model = SiameseModel(tower_model).to(rank)
    siamese_model = torch.nn.parallel.DistributedDataParallel(siamese_model, device_ids=[rank])

    loss_function = triplet_loss

    optimizer = torch.optim.SGD(siamese_model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0001)

    trainer = SSLTrainer(model=siamese_model,
                         optimizer=optimizer,
                         loss_function=loss_function,
                         batch_size=BATCH_SIZE,
                         rank=rank,
                         world_size=world_size,
                         device=device
                         )

    try:
        start_time = time.time()
        ssl_prefix = f"ssl_{proportion*100:.0f}%_warm_start" if warm_start_successful else f"ssl_{proportion*100:.0f}%_from_scratch"
        history = trainer.train(train_dataloader, test_dataloader, NUM_EPOCHS, prefix=ssl_prefix)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)
            
            model_path = os.path.join("models", f"{ssl_prefix}_siamese_tower.pt")
            ssl_save_model(model_path, siamese_model.module.tower, optimizer, loss_function, history, BATCH_SIZE)

    finally:
        cleanup_ddp()

def handle_train_parser(proportion, load_supervised_model):
    world_size = torch.cuda.device_count()
    print(f"Prepare training process on {world_size} GPU")
    mp.spawn(handle_train_ddp, args=(world_size, proportion, load_supervised_model), nprocs=world_size, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-streams Attention Adaptive model for Human's Action Recognition (SSL Training)")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    parser.add_argument("--proportion", type=float, default=1.0, help="Proportion of the dataset to use for training (0.0 to 1.0)")
    parser.add_argument("--load-supervised-model", action="store_true", 
                        help="Load a pre-trained supervised model for warm-starting the tower model.")

    args = parser.parse_args()

    handle_train_parser(args.proportion, args.load_supervised_model)