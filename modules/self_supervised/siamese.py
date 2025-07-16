import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

def triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin=1.0):
    distance_positive = (anchor_embedding - positive_embedding).pow(2).sum(1)
    distance_negative = (anchor_embedding - negative_embedding).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

class SiameseDataset(Dataset):
    def __init__(self, training_generators):
        self.training_generators = training_generators
        self.class_to_idx = {i: gen for i, gen in enumerate(training_generators)}
        self.classes = [i for i, gen in enumerate(training_generators) if len(gen) > 0]
        self.n_classes = len(self.classes)

        self.samples_by_class_flat = []
        for class_idx, gen in enumerate(self.training_generators):
            for sample_in_class_idx in range(len(gen)):
                self.samples_by_class_flat.append((class_idx, sample_in_class_idx))
        
        self.non_empty_classes = [c for c in self.classes if len(self.class_to_idx[c]) > 0]

    def __getitem__(self, index):
        anchor_class_idx = random.choice(self.non_empty_classes)
        anchor_generator = self.class_to_idx[anchor_class_idx]
        anchor_sample_idx = random.randint(0, len(anchor_generator) - 1)
        anchor_data = anchor_generator[anchor_sample_idx][0]

        positive_class_idx = anchor_class_idx
        positive_generator = self.class_to_idx[positive_class_idx]
        
        if len(positive_generator) > 1:
            positive_sample_idx = random.randint(0, len(positive_generator) - 1)
            while positive_sample_idx == anchor_sample_idx:
                positive_sample_idx = random.randint(0, len(positive_generator) - 1)
        else:
            positive_sample_idx = anchor_sample_idx
            
        positive_data = positive_generator[positive_sample_idx][0]

        negative_class_idx = random.choice(self.non_empty_classes)
        while negative_class_idx == anchor_class_idx:
            negative_class_idx = random.choice(self.non_empty_classes)
        
        negative_generator = self.class_to_idx[negative_class_idx]
        negative_sample_idx = random.randint(0, len(negative_generator) - 1)
        negative_data = negative_generator[negative_sample_idx][0]

        return anchor_data, positive_data, negative_data

    def __len__(self):
        total_samples = sum(len(gen) for gen in self.training_generators)
        return total_samples if total_samples > 0 else 1

class SiameseDataLoader:
    def __init__(self, dataset, batch_size, class_count, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

class SiameseModel(nn.Module):
    def __init__(self, tower):
        super(SiameseModel, self).__init__()
        self.tower = tower

    def forward(self, x_tuple):
        anchor_data, positive_data, negative_data = x_tuple
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