import torch
import numpy as np
import random
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

from modules.ms_aagcn import ms_aagcn
from modules.utils import load_model 
from data_generator.ntu_data import NTU_Dataset
from torch_geometric.loader import DataLoader

from captum.attr import Saliency, IntegratedGradients

# --- Configuration ---
MODEL_PATH = "models/supervised_10%.pt" 
DATASET_PATH = "data/nturgb+d_skeletons/"
BENCHMARK = "xsub"
MODALITY = "joint"
USE_EXTENDED_DATASET = False 
NUM_CLASSES = 120 if USE_EXTENDED_DATASET else 60


def model_forward_wrapper(net, inputs, device):
    return net(inputs.to(device))

def explain(method, net, data_input, target, device):
    input_tensor = data_input.clone().requires_grad_(True).to(device)
    forward_func = lambda x: model_forward_wrapper(net, x, device)

    if method == 'ig':
        explainer = IntegratedGradients(forward_func)
        attributions = explainer.attribute(input_tensor, target=target, internal_batch_size=1)
    elif method == 'saliency':
        explainer = Saliency(forward_func)
        attributions = explainer.attribute(input_tensor, target=target)
    else:
        raise NotImplementedError('Unsupported explanation method.')

    attributions = attributions.sum(dim=(0, 1, 2, 4)).cpu().detach().numpy()
    
    if attributions.max() > 0:
        attributions = np.abs(attributions) / np.abs(attributions).max()
    return attributions

def draw_and_save_skeleton(skeleton_data, attributions, title, save_path):
    skeleton_edges = [
        (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7),
        (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14),
        (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (7, 21), (7, 22),
        (11, 23), (11, 24)
    ]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    person1_activity = np.abs(skeleton_data[0, :, :, :, 0]).sum()
    person2_activity = np.abs(skeleton_data[0, :, :, :, 1]).sum()
    person_idx = 1 if person2_activity > person1_activity else 0
    
    valid_frame_idx = -1
    for i in range(skeleton_data.shape[2]):
        frame_activity = np.abs(skeleton_data[0, 0:2, i, :, person_idx]).sum()
        if frame_activity > 1e-5:
            valid_frame_idx = i
            break
            
    if valid_frame_idx == -1:
        print(f"Warning: Could not find a valid frame to draw for {save_path}. Skipping visualization.")
        return

    x = skeleton_data[0, 0, valid_frame_idx, :, person_idx].numpy()
    y = skeleton_data[0, 1, valid_frame_idx, :, person_idx].numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for p1_idx, p2_idx in skeleton_edges:
        ax.plot([x[p1_idx], x[p2_idx]], [y[p1_idx], y[p2_idx]], 'c-')
    scatter = ax.scatter(x, y, c=attributions, cmap='Reds', s=80 + attributions * 200, zorder=3, alpha=0.8)
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min if x_max > x_min else 0.2
    y_range = y_max - y_min if y_max > y_min else 0.2
    padding = max(x_range, y_range) * 0.1 + 0.1
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.colorbar(scatter, ax=ax, label='Attribution Score')
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")

def save_scores(ig_scores, saliency_scores, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("--- Attribution Scores per Joint (Normalized) ---\n")
        f.write(f"{'Joint Index':<12} | {'Integrated Gradients':<22} | {'Saliency':<10}\n")
        f.write("-" * 50 + "\n")
        for i in range(len(ig_scores)):
            f.write(f"{i:<12} | {ig_scores[i]:<22.4f} | {saliency_scores[i]:<10.4f}\n")
    print(f"Scores saved to: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = ms_aagcn(num_class=NUM_CLASSES)
    print(f"Loading model from: {MODEL_PATH}")
    model_state_dict, _, _, _, _ = load_model(MODEL_PATH)
    new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in model_state_dict.items())
    net.load_state_dict(new_state_dict)
    net.to(device)
    net.eval()
    print("Model loaded successfully.")

    print("Initializing dataset reference...")
    test_dataset = NTU_Dataset(
        root=DATASET_PATH, modality=MODALITY, benchmark=BENCHMARK, 
        part="eval", extended=USE_EXTENDED_DATASET,
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__
    )
    
    while True:
        random_index = random.choice(range(len(test_dataset)))
        data_sample = test_dataset[random_index]
        if data_sample.x.abs().sum() > 1.0:
            break
        else:
            print(f"Skipping completely empty sample at index: {random_index}")

    print(f"Found potentially valid sample at index: {random_index}")
    data_batch = next(iter(DataLoader([data_sample], batch_size=1)))

    input_data_tensor = data_batch.x
    true_label = (data_batch.y - 1).item()
    output = net(input_data_tensor.to(device))
    predicted_label = output.argmax(dim=1).item()
    
    print(f"\n--- Analyzing sample {random_index} ---")
    print(f"True class: {true_label}")
    print(f"Model prediction: {predicted_label}")

    target = predicted_label
    
    print("\nCalculating attributions...")
    ig_attributions = explain('ig', net, input_data_tensor, target, device)
    saliency_attributions = explain('saliency', net, input_data_tensor, target, device)

    action_name = f"sample{random_index}_true{true_label}_pred{predicted_label}"
    
    save_scores(
        ig_attributions, saliency_attributions,
        save_path=f'results/captum_visu/{action_name}/scores_{action_name}.txt'
    )
    
    draw_and_save_skeleton(
        input_data_tensor.cpu(), ig_attributions,
        f'Integrated Gradients for Action: {action_name}',
        f'results/captum_visu/{action_name}/ig_{action_name}.png'
    )
    
    draw_and_save_skeleton(
        input_data_tensor.cpu(), saliency_attributions,
        f'Saliency for Action: {action_name}',
        f'results/captum_visu/{action_name}/saliency_{action_name}.png'
    )
    
    print("\nAnalysis complete.")