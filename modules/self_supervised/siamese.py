import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Votre fonction triplet_loss reste inchangée
def triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin=1.0):
    distance_positive = (anchor_embedding - positive_embedding).pow(2).sum(1)
    distance_negative = (anchor_embedding - negative_embedding).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

# --- Fonctions d'augmentation de données pour les squelettes ---
# Les données de squelette sont de forme (C, T, V, M) : (Canaux, Temps, Articulations, Personnes)

class DataAugmentations:
    def __init__(self, spatial_noise_std=0.01, temporal_mask_ratio=0.1, spatial_mask_ratio=0.1):
        self.spatial_noise_std = spatial_noise_std
        self.temporal_mask_ratio = temporal_mask_ratio
        self.spatial_mask_ratio = spatial_mask_ratio

    def __call__(self, x):
        """Applique une composition aléatoire d'augmentations à un échantillon de squelette."""
        x_aug = x.clone()
        
        # Choisir aléatoirement une ou plusieurs augmentations
        if random.random() < 0.5: # 50% de chance d'appliquer le bruit spatial
            x_aug = self.spatial_jitter(x_aug)
        if random.random() < 0.3: # 30% de chance d'appliquer le masquage temporel
            x_aug = self.temporal_masking(x_aug)
        if random.random() < 0.3: # 30% de chance d'appliquer le masquage spatial
            x_aug = self.spatial_masking(x_aug)
        # Ajouter d'autres augmentations ici (rotation, scaling, etc.)

        return x_aug

    def spatial_jitter(self, x):
        """Ajoute un bruit aléatoire aux coordonnées des articulations."""
        noise = torch.randn_like(x) * self.spatial_noise_std
        return x + noise

    def temporal_masking(self, x):
        """Masque aléatoirement des segments temporels entiers (met à zéro)."""
        T_dim = x.shape[1]
        num_frames_to_mask = int(T_dim * self.temporal_mask_ratio)
        if num_frames_to_mask == 0 and T_dim > 0:
            num_frames_to_mask = 1

        if num_frames_to_mask > 0:
            start_idx = random.randint(0, T_dim - num_frames_to_mask)
            x[:, start_idx : start_idx + num_frames_to_mask, :, :] = 0
        return x

    def spatial_masking(self, x):
        """Masque aléatoirement des articulations entières (met à zéro)."""
        V_dim = x.shape[2]
        num_joints_to_mask = int(V_dim * self.spatial_mask_ratio)
        if num_joints_to_mask == 0 and V_dim > 0:
            num_joints_to_mask = 1

        if num_joints_to_mask > 0:
            joints_to_mask = random.sample(range(V_dim), num_joints_to_mask)
            x[:, :, joints_to_mask, :] = 0
        return x

# --- MODIFICATION : Implémentation correcte du Dataset pour SSL ---
class SiameseDataset(Dataset):
    def __init__(self, dataset, augmentations=None):
        """
        Args:
            dataset (torch.utils.data.Subset or torch.utils.data.Dataset): 
                L'ensemble de données à utiliser (typiquement non labellisé).
            augmentations (callable): Une fonction de transformation pour créer des paires positives.
        """
        self.dataset = dataset
        self.augmentations = augmentations if augmentations is not None else DataAugmentations()

    def __getitem__(self, index):
        # Anchor: L'échantillon original.
        # On suppose que le dataset renvoie un objet avec un attribut .x
        anchor_data = self.dataset[index].x 

        # Positive: Une vue augmentée de l'ancre.
        positive_data = self.augmentations(anchor_data)

        # Negative: Un échantillon complètement différent, aussi augmenté.
        negative_index = random.randint(0, len(self.dataset) - 1)
        while negative_index == index:
            negative_index = random.randint(0, len(self.dataset) - 1)
        
        negative_data_raw = self.dataset[negative_index].x
        negative_data = self.augmentations(negative_data_raw) 
        
        return anchor_data, positive_data, negative_data

    def __len__(self):
        return len(self.dataset)

# --- MODIFICATION : Simplification du DataLoader ---
class SiameseDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Le paramètre class_count est supprimé car non pertinent pour le SSL
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

# Votre SiameseModel reste inchangé
class SiameseModel(nn.Module):
    def __init__(self, tower):
        super(SiameseModel, self).__init__()
        self.tower = tower

    def forward(self, x_tuple):
        anchor_data, positive_data, negative_data = x_tuple
        
        # Le formatage des données reste le même
        if anchor_data.dim() == 6 and anchor_data.shape[1] == 1:
            anchor_data = anchor_data.squeeze(1)
        if positive_data.dim() == 6 and positive_data.shape[1] == 1:
            positive_data = positive_data.squeeze(1)
        if negative_data.dim() == 6 and negative_data.shape[1] == 1:
            negative_data = negative_data.squeeze(1)
        
        anchor_embedding = self.tower(anchor_data)
        positive_embedding = self.tower(positive_data)
        negative_embedding = self.tower(negative_data)

        return anchor_embedding, positive_embedding, negative_embedding