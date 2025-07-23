import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import math

def triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin=1.0):
    distance_positive = (anchor_embedding - positive_embedding).pow(2).sum(1)
    distance_negative = (anchor_embedding - negative_embedding).pow(3).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

class DataAugmentations:
    def __init__(self, spatial_noise_std=0.01, temporal_mask_ratio=0.1, spatial_mask_ratio=0.1):
        self.spatial_noise_std = spatial_noise_std
        self.temporal_mask_ratio = temporal_mask_ratio
        self.spatial_mask_ratio = spatial_mask_ratio

    def __call__(self, x):
        x_aug = x.clone()
        if random.random() < 0.5:
            x_aug = self.spatial_jitter(x_aug)
        if random.random() < 0.3:
            x_aug = self.temporal_masking(x_aug)
        if random.random() < 0.3:
            x_aug = self.spatial_masking(x_aug)
        if random.random() < 0.3:
            x_aug = self.rotate(x_aug)
        return x_aug

    def spatial_jitter(self, x):
        noise = torch.randn_like(x) * self.spatial_noise_std
        return x + noise

    def temporal_masking(self, x):
        T = x.shape[2]
        num_frames_to_mask = int(T * self.temporal_mask_ratio)
        if num_frames_to_mask > 0:
            start_idx = random.randint(0, T - num_frames_to_mask)
            x[:, :, start_idx : start_idx + num_frames_to_mask, :, :] = 0
        return x

    def spatial_masking(self, x):
        V = x.shape[3]
        num_joints_to_mask = int(V * self.spatial_mask_ratio)
        if num_joints_to_mask > 0:
            joints_to_mask = random.sample(range(V), num_joints_to_mask)
            x[:, :, :, joints_to_mask, :] = 0
        return x

    def rotate(self, x, angle=None):
        x_squeezed = x.squeeze(0)
        if angle is None:
            angle = random.uniform(-15.0, 15.0)
        angle = math.radians(angle)
        c = math.cos(angle)
        s = math.sin(angle)
        rotation_matrix = torch.tensor([[c, -s], [s, c]], dtype=x_squeezed.dtype, device=x_squeezed.device)
        xy = x_squeezed[:2, :, :, :]
        xy_perm = xy.permute(1, 2, 3, 0)
        xy_flat = xy_perm.reshape(-1, 2)
        rotated_xy_flat = torch.matmul(xy_flat, rotation_matrix.T)
        rotated_xy_perm = rotated_xy_flat.reshape(x_squeezed.shape[1], x_squeezed.shape[2], x_squeezed.shape[3], 2)
        rotated_xy = rotated_xy_perm.permute(3, 0, 1, 2)
        x_rot = x_squeezed.clone()
        x_rot[:2, :, :, :] = rotated_xy
        return x_rot.unsqueeze(0)

class SiameseDataset(Dataset):
    def __init__(self, dataset, augmentations=None):
        self.dataset = dataset
        self.augmentations = augmentations if augmentations is not None else DataAugmentations()

    def __getitem__(self, index):
        anchor = self.dataset[index].x 
        positive = self.augmentations(anchor)

        negative_index = random.randint(0, len(self.dataset) - 1)
        while negative_index == index:
            negative_index = random.randint(0, len(self.dataset) - 1)

        negative_raw = self.dataset[negative_index].x
        negative = self.augmentations(negative_raw)

        return anchor, positive, negative

    def __len__(self):
        return len(self.dataset)

class SiameseDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

class SiameseModel(nn.Module):
    def __init__(self, tower):
        super(SiameseModel, self).__init__()
        self.tower = tower

    def forward(self, x_tuple):
        anchor, positive, negative = x_tuple

        # Remove added dimension by DataLoader if present
        if anchor.dim() == 6 and anchor.shape[1] == 1:
            anchor = anchor.squeeze(1)
        if positive.dim() == 6 and positive.shape[1] == 1:
            positive = positive.squeeze(1)
        if negative.dim() == 6 and negative.shape[1] == 1:
            negative = negative.squeeze(1)

        anchor_embed = self.tower(anchor)
        positive_embed = self.tower(positive)
        negative_embed = self.tower(negative)

        return anchor_embed, positive_embed, negative_embed
