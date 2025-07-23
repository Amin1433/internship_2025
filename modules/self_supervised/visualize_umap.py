import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from data_generator.ntu_data import NTU_Dataset, EDGE_INDEX
from modules.self_supervised.tower import MsAAGCN
from modules.utils import splitting_prop
from torch.utils.data import Subset

DATASET_PATH = "data/nturgb+d_skeletons/"
MODALITY = "joint"
BENCHMARK = "xsub"
RANDOM_SEED = 42
NUM_JOINT = 25
NUM_SKELETON = 2
IN_CHANNELS = 3
LATENT_SIZE = 256

def visualize(args):

    if not os.path.exists(args.weights_path):
        print(f"ERROR: Weights file '{args.weights_path}' not found.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    total_dataset = NTU_Dataset(root=DATASET_PATH, modality=MODALITY, benchmark=BENCHMARK, part="train")
    labeled_set, _ = splitting_prop(total_dataset, proportion=args.proportion)
    
    num_total_labeled = len(labeled_set)
    num_samples_to_use = min(args.num_samples, num_total_labeled)
    
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(range(num_total_labeled), num_samples_to_use, replace=False)
    visualization_subset = Subset(labeled_set, indices)
    
    print(f"{len(visualization_subset)} samples will be used for visualization.")

    print("Loading pre-trained encoder model...")
    model = MsAAGCN(
        num_joint=NUM_JOINT, 
        num_skeleton=NUM_SKELETON, 
        edge_index=EDGE_INDEX, 
        in_channels=IN_CHANNELS, 
        latent_size=LATENT_SIZE
    ).to(device)

    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Generating embeddings...")
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(visualization_subset, desc="Extracting embeddings"):
            inputs = data.x.to(device)
            label = data.y.item() - 1  # Adjust labels from 1-indexed to 0-indexed
            embedding = model(inputs)
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(label)

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.array(all_labels)
    
    print("Embeddings extracted.")

    print("Computing UMAP projection (this may take a while)...")
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1,    
        n_components=2,
        metric='cosine',
        random_state=42  
    )
    umap_results = reducer.fit_transform(embeddings_array)
    print("UMAP computation completed.")

    print("Generating plot...")
    n_classes = len(np.unique(labels_array))
    
    plt.figure(figsize=(20, 14))
    sns.scatterplot(
        x=umap_results[:, 0], y=umap_results[:, 1],
        hue=labels_array,
        palette=sns.color_palette("hsv", n_classes),
        legend='full',
        alpha=0.8
    )
    plt.title(f"UMAP Visualization of Embeddings\n(Model: {os.path.basename(args.weights_path)})", fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=12) 
    plt.ylabel("UMAP Component 2", fontsize=12) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)

    output_dir = os.path.dirname(args.weights_path)
    output_path = os.path.join(output_dir, "embeddings_umap_visualization.png")
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print("-" * 50)
    print(f"UMAP visualization successfully saved at: {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SSL encoder embeddings with UMAP.")
    parser.add_argument("--weights-path", type=str, required=True, help="Path to the encoder model weights (.pt).")
    parser.add_argument("--proportion", type=float, required=True, help="Proportion of labeled data, e.g. 0.1")
    parser.add_argument("--num-samples", type=int, default=1500, help="Number of samples to use.")
    args = parser.parse_args()
    visualize(args)
