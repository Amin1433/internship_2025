import sys
import torch
from collections import OrderedDict

def ssl_save_model(output_path: str, model_tower, optimizer, loss_function, history, batch_size):
    parameters = {
        'model_tower_state_dict': model_tower.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_function': loss_function,
        'history': history,
        'batch_size': batch_size
    }
    
    with open(output_path, "wb") as f:
        torch.save(parameters, f)

    print(f"SSL model saved to {output_path}", file=sys.stderr)

def ssl_load_model(model_path: str, model_tower_instance, optimizer_instance, loss_function_instance):
    with open(model_path, 'rb') as f:
        parameters = torch.load(f, weights_only=False)

    model_tower_state_dict = parameters['model_tower_state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in model_tower_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model_tower_instance.load_state_dict(new_state_dict)
    optimizer_instance.load_state_dict(parameters['optimizer_state_dict'])
    history = parameters.get('history', {})
    batch_size = parameters.get('batch_size', None)

    print(f"Loaded SSL model from {model_path}", file=sys.stderr)
    return model_tower_instance, optimizer_instance, loss_function_instance, history, batch_size