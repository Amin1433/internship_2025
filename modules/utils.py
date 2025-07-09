import sys
import torch
from torch.utils.data import random_split, Subset

def save_model(output_path : str, model, optimizer, loss_function, history, batch_size):
    parameters = {
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'loss_function' : loss_function,
            'history' : history,
            'batch_size' : batch_size 
        }
    
    with open(output_path, "wb") as f:
        torch.save(parameters, f)

    print(f"Model saved to {output_path}", file=sys.stderr)

def load_model(model_path : str):
    with open(model_path, 'rb') as f:
        parameters = torch.load(f, weights_only=False)
    model = parameters['model']
    optimizer = parameters['optimizer']
    loss_function = parameters['loss_function']
    history = parameters['history']
    batch_size = parameters['batch_size']

    print(f"Loaded model from {model_path}", file=sys.stderr)
    return model, optimizer, loss_function, history, batch_size

def splitting_prop(dataset, proportion=0.5):
    dataset_size = torch.randperm(len(dataset)).tolist()  
    splitpoint = int(len(dataset) * proportion)
    dataset1_indices = dataset_size[:splitpoint]
    dataset2_indices = dataset_size[splitpoint:]
    dataset1 = Subset(dataset, dataset1_indices)
    dataset2 = Subset(dataset, dataset2_indices)
    return dataset1, dataset2