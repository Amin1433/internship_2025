import gc
import sys
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from modules.utils import *

import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

import os 
import numpy as np 

class Trainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model: nn.Module, optimizer, loss_function, batch_size, rank=0, world_size=1, patience=5, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = device if device is not None else torch.device(f"cuda:{self.rank}")
        self.scaler = GradScaler('cuda')
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None  # Store the best model

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def train(self, train_data: Dataset, val_data: Dataset, num_epoch, prefix=None):
        
        # This ensures that each call to train() records metrics for that phase only.
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        self.best_val_loss = 10
        self.epochs_without_improvement = 0
        self.best_model_state = self.model.state_dict()
        

        if prefix is not None and self.rank == 0: 
            os.makedirs(f"logs/{prefix}", exist_ok=True) 

        train_sampler = DistributedSampler(train_data, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_data, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        train_data_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler,
                                       num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        val_data_loader = DataLoader(val_data, batch_size=self.batch_size, sampler=val_sampler,
                                     num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 40], gamma=0.1)
        self.scheduler = None
        for epoch in range(num_epoch):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            self.model.train()
            avg_train_loss = self._run_epoch(train_data_loader, train=True)
            
            self.model.eval()
            avg_val_loss = self._run_epoch(val_data_loader, train=False)

            if self.rank == 0:
                train_acc = self.history["train_acc"][-1]
                val_acc = self.history["val_acc"][-1]
                print(f"Epoch {epoch + 1}/{num_epoch}: "
                      f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f} | "
                      f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # Early Stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
            else:
                self.epochs_without_improvement += 1

            
            if self.epochs_without_improvement >= self.patience:
                break
            

            # if self.scheduler:
            #     self.scheduler.step()

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        
        if prefix is not None and self.rank == 0:  
            np.save(f"logs/{prefix}/train_loss.npy", np.array(self.history["train_loss"]))
            np.save(f"logs/{prefix}/train_acc.npy", np.array(self.history["train_acc"]))
            np.save(f"logs/{prefix}/val_loss.npy", np.array(self.history["val_loss"]))
            np.save(f"logs/{prefix}/val_acc.npy", np.array(self.history["val_acc"]))

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0
        correct_preds = 0
        total_samples = 0

        data_iter = tqdm(dataloader, unit="batch") if self.rank == 0 else dataloader
        for batch in data_iter:
            self.optimizer.zero_grad()

            inputs = batch.x.to(self.device)
            truths = (batch.y - 1).to(self.device)

            if train:
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, truths)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    with autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.compute_loss(outputs, truths)

            total_loss += loss.item()
            correct_preds += (outputs.argmax(dim=1) == truths).sum().item()
            total_samples += truths.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct_preds / total_samples

        phase = "train" if train else "val"
        self.history[f"{phase}_loss"].append(avg_loss)
        self.history[f"{phase}_acc"].append(avg_acc)

        return avg_loss

    def compute_loss(self, outputs, truths):
        return self.loss_function(outputs, truths)

    def compute_accuracy(self, outputs, truths):
        preds = outputs.argmax(dim=1)
        correct = (preds == truths).sum().item()
        total = truths.size(0)
        return correct / total
