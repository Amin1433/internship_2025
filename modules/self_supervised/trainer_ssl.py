import gc
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import os 
import numpy as np 
import torch.distributed as dist

class SSLTrainer:
    def __init__(self, model: nn.Module, optimizer, loss_function, batch_size, rank=0, world_size=1, patience=5, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        # Correction 1: Utilisation de torch.amp.GradScaler
        self.device = device if device is not None else torch.device(f"cuda:{self.rank}")
        self.scaler = torch.amp.GradScaler(device=self.device) # Spécifier le device

        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def train(self, train_dataloader, val_dataloader, num_epochs, prefix=None):
        if prefix is not None and self.rank == 0: 
            os.makedirs(f"logs/{prefix}", exist_ok=True) 

        self.scheduler = None # Conserver cette ligne pour la compatibilité, même si non utilisée ici

        for epoch in range(num_epochs):
            # Assurez-vous que le sampler est mis à jour pour DDP
            if isinstance(train_dataloader.data_loader.sampler, DistributedSampler):
                train_dataloader.data_loader.sampler.set_epoch(epoch)
            if isinstance(val_dataloader.data_loader.sampler, DistributedSampler):
                val_dataloader.data_loader.sampler.set_epoch(epoch)

            self.model.train()
            avg_train_loss = self._run_epoch(train_dataloader, train=True, epoch=epoch)
            
            self.model.eval()
            avg_val_loss = self._run_epoch(val_dataloader, train=False, epoch=epoch)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            if self.rank == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: "
                      f"Train Loss = {avg_train_loss:.4f} | "
                      f"Val Loss = {avg_val_loss:.4f}")

                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.epochs_without_improvement = 0
                    # Sauvegarde le state_dict du modèle sous-jacent (tower)
                    self.best_model_state = self.model.module.tower.state_dict()
                    if prefix is not None:
                        # Utilise un nom de fichier clair pour le meilleur modèle de la tour
                        torch.save(self.best_model_state, f"logs/{prefix}/best_siamese_tower.pt")
                        print(f"  --> Best model saved at epoch {epoch+1} with Val Loss: {self.best_val_loss:.4f}")

                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    if self.rank == 0: # Redondant car déjà dans rank == 0 mais pour clarté
                        print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break

            if self.world_size > 1:
                dist.barrier() # S'assure que tous les processus attendent le rank 0 avant de continuer

        if self.best_model_state and self.rank == 0:
            # Charge le meilleur état du modèle si early stopping s'est produit
            # Assurez-vous que `self.model.module.tower` est l'objet qui doit recevoir le `state_dict`
            self.model.module.tower.load_state_dict(self.best_model_state)
            print(f"Restored best model state with Val Loss: {self.best_val_loss:.4f}")

        if prefix is not None and self.rank == 0: 
            np.save(f"logs/{prefix}/train_loss.npy", np.array(self.history["train_loss"]))
            np.save(f"logs/{prefix}/val_loss.npy", np.array(self.history["val_loss"]))

        return self.history

    def _run_epoch(self, dataloader, train, epoch):
        total_loss = 0
        num_batches = 0

        data_iter = tqdm(dataloader.data_loader, unit="batch", 
                         desc=f"Epoch {epoch+1} {'Train' if train else 'Val'} (Rank {self.rank})", 
                         disable=(self.rank != 0))

        for anchor_data, positive_data, negative_data in data_iter:
            anchor_data = anchor_data.to(self.device)
            positive_data = positive_data.to(self.device)
            negative_data = negative_data.to(self.device)

            self.optimizer.zero_grad()

            if train:
                # Correction 2: Utilisation de torch.amp.autocast
                with torch.amp.autocast('cuda'):
                    anchor_embedding, positive_embedding, negative_embedding = self.model((anchor_data, positive_data, negative_data))
                    loss = self.loss_function(anchor_embedding, positive_embedding, negative_embedding)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    # Correction 2: Utilisation de torch.amp.autocast
                    with torch.amp.autocast('cuda'):
                        anchor_embedding, positive_embedding, negative_embedding = self.model((anchor_data, positive_data, negative_data))
                        loss = self.loss_function(anchor_embedding, positive_embedding, negative_embedding)

            total_loss += loss.item()
            num_batches += 1

            if self.rank == 0:
                data_iter.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            # Réduit la somme des pertes moyennes de tous les processus
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            # Divise par le nombre total de processus pour obtenir la moyenne globale
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss