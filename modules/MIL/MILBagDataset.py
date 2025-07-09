import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import copy
from typing import List, Dict, Tuple


def bag_label_from_instance_labels(instance_labels: List[int]) -> int:
    return int(any(50 <= x <= 60 for x in instance_labels))


def data_generation(dataset) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    instance_index_label = [(i, int(dataset[i].y)) for i in range(len(dataset))]
    bag_sizes = np.random.randint(3, 7, size=len(instance_index_label)//5)

    data_cp = copy.copy(instance_index_label)
    np.random.shuffle(data_cp)

    bags = {}
    bags_labels = {}

    for bag_id, size in enumerate(bag_sizes):
        try:
            indices = [data_cp.pop() for _ in range(size)]
            instance_ids = [x[0] for x in indices]
            instance_labels = [x[1] for x in indices]
            bags[bag_id] = instance_ids
            bags_labels[bag_id] = bag_label_from_instance_labels(instance_labels)
        except IndexError:
            break

    return bags, bags_labels


def create_edge_index(num_nodes: int) -> torch.Tensor:
    row = []
    col = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


class BagGraphDataset(Dataset):
    def __init__(self, bags_dict: Dict[int, List[int]],
                 bag_labels_dict: Dict[int, int],
                 feature_dict: Dict[int, torch.Tensor]):
        super().__init__()
        self.bag_ids = sorted(bags_dict.keys())
        self.bags_dict = bags_dict
        self.bag_labels_dict = bag_labels_dict
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.bag_ids)

    def __getitem__(self, idx):
        bag_id = self.bag_ids[idx]
        instance_ids = self.bags_dict[bag_id]
        features = [self.feature_dict[i] for i in instance_ids]
        x = torch.stack(features)  # [num_nodes, feat_dim]
        edge_index = create_edge_index(x.size(0))  # [2, num_edges]
        y = torch.tensor([self.bag_labels_dict[bag_id]], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)
