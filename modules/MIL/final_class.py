
import torch
import torch.nn as nn
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_generator.ntu_data import NTU_Dataset
from modules.utils import splitting_prop, save_model
from torch.amp import autocast, GradScaler

RANDOM_SEED = 42
PROPORTION = 0.1
REFINED_FEATURES_PATH = f"models/features/refined_features_{PROPORTION*100:.1f}%.pkl"
DATASET_PATH = "data/nturgb+d_skeletons/"
MODEL_PREFIX = "final_classifier_refined"
MODEL_OUTPUT_PATH = f"models/{MODEL_PREFIX}.pt"
NUM_CLASSES = 60
INPUT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RefinedFeatureDataset(Dataset):
    def __init__(self, original_dataset_subset, refined_features_dict):
        self.indices = original_dataset_subset.indices
        self.original_dataset = original_dataset_subset.dataset
        self.features = refined_features_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        feature = self.features[original_idx]
        label = self.original_dataset[original_idx].y.squeeze()
        return feature, label

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Trainer:
    def __init__(self, model, optimizer, criterion, batch_size, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.scaler = GradScaler('cuda')
        self.best_val_loss = float('inf')
        self.history = {}

    def _run_epoch(self, dataloader, is_training):
        self.model.train(is_training)
        total_loss = 0
        correct_preds = 0
        total_samples = 0

        data_iter = tqdm(dataloader, desc="Training" if is_training else "Validation")
        for features, labels in data_iter:
            features = features.to(self.device)
            truths = (labels - 1).to(self.device)

            self.optimizer.zero_grad()
            
            with autocast('cuda', enabled=is_training):
                outputs = self.model(features)
                loss = self.criterion(outputs, truths)

            if is_training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            total_loss += loss.item()
            correct_preds += (outputs.argmax(dim=1) == truths).sum().item()
            total_samples += truths.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct_preds / total_samples
        return avg_loss, avg_acc

    def train(self, train_data, val_data, num_epochs, prefix):
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=4)

        for epoch in range(num_epochs):
            train_loss, train_acc = self._run_epoch(train_loader, is_training=True)
            
            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(val_loader, is_training=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  -> New best validation loss. Saving model to models/{prefix}.pt")
                save_model(f"models/{prefix}.pt", self.model, self.optimizer, self.criterion, self.history, self.batch_size)
        
        log_dir = f"logs/{prefix}"
        os.makedirs(log_dir, exist_ok=True)
        for key, value in self.history.items():
            np.save(os.path.join(log_dir, f"{key}.npy"), np.array(value))
        print(f"Final logs saved to '{log_dir}'")

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Random Seed: {RANDOM_SEED}")

    print(f"Loading refined features from: {REFINED_FEATURES_PATH}")
    with open(REFINED_FEATURES_PATH, "rb") as f:
        refined_features_dict = pickle.load(f)

    print("Reproducing the original data splits...")
    full_dataset = NTU_Dataset(root=DATASET_PATH, part="train", extended=False)
    labeled_dataset, _ = splitting_prop(full_dataset, PROPORTION)
    train_subset, val_subset = splitting_prop(labeled_dataset, 0.7)
    
    print(f"Total labeled instances: {len(labeled_dataset)}")
    print(f"Final training set size: {len(train_subset)}")
    print(f"Final validation set size: {len(val_subset)}")

    train_dataset = RefinedFeatureDataset(train_subset, refined_features_dict)
    val_dataset = RefinedFeatureDataset(val_subset, refined_features_dict)
    
    model = SimpleClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nInitializing Trainer and starting training process...")
    trainer = Trainer(model, optimizer, criterion, BATCH_SIZE, device)
    trainer.train(train_dataset, val_dataset, EPOCHS, prefix=MODEL_PREFIX)
    
    print(f"\nTraining finished. Best model was saved during training to '{MODEL_OUTPUT_PATH}'")

if __name__ == "__main__":
    main()