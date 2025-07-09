import numpy as np
from torch.utils.data import random_split  

def splitting_prop(dataset, proportion=0.5):
    dataset_size = len(dataset)
    dataset1_size = int(dataset_size * proportion)
    dataset2_size = dataset_size - dataset1_size # Ensure all elements are covered
    dataset1, dataset2 = random_split(dataset, [dataset1_size, dataset2_size])
    return dataset1, dataset2