import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import Subset

from data_generator.ntu_data import NTU_Dataset, EDGE_INDEX
from modules.self_supervised.tower import MsAAGCN
from modules.utils import splitting_prop

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
        print(f"ERROR: Model weights file '{args.weights_path}' not found.")
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

    print(f"{len(visualization_subset)} samples selected for visualization.")

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
            label = data.y.item() - 1  # Labels are 1-indexed in NTU, adjust to 0-indexed
            embedding = model(inputs)
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(label)

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    labels_array = np.array(all_labels)

    print("Computing t-SNE projection (this may take a few minutes)...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, learning_rate='auto', init='pca')
    tsne_results = tsne.fit_transform(embeddings_array)
    print("t-SNE computation done.")

    print("Generating plot...")
    n_classes = len(np.unique(labels_array))

    plt.figure(figsize=(20, 14))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels_array,
        palette=sns.color_palette("hsv", n_classes),
        legend='full',
        alpha=0.8
    )
    plt.title(f"t-SNE Visualization of Embeddings\n(Model: {os.path.basename(args.weights_path)})", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)

    output_dir = os.path.dirname(args.weights_path)
    output_path = os.path.join(output_dir, "embeddings_tsne_visualization.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print("-" * 50)
    print("t-SNE visualization saved successfully to:")
    print(output_path)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize encoder embeddings with t-SNE.")
    parser.add_argument("--weights-path", type=str, required=True,
                        help="Path to the pre-trained encoder model weights (.pt)")
    parser.add_argument("--proportion", type=float, required=True,
                        help="Proportion of the dataset considered labeled (should match training config)")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to use for visualization")
    args = parser.parse_args()
    visualize(args)
