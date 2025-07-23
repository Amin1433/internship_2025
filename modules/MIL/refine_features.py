import torch
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from data_generator.ntu_data import NTU_Dataset
from modules.ms_aagcn import ms_aagcn, extract_features
from modules.MIL.GraphMIL import MIL_GCN_Attention
from modules.MIL.MILBagDataset import data_generation, BagGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from modules.utils import load_model

PROPORTION = 0.1
PRE_MODEL_PATH = f"models/supervised_{int(PROPORTION*100)}%.pt"
MIL_MODEL_PATH = f"models/weakly_mil_model_10.pt"
FULL_FEATURES_PATH = f"models/features/features_full_dataset.pkl"
REFINED_FEATURES_OUT_PATH = f"models/features/refined_features_{PROPORTION*100:.1f}%.pkl"

DATASET_PATH = "data/nturgb+d_skeletons/"
MODALITY = "joint"
BENCHMARK = "xsub"
NUM_CLASSES = 60
MIL_MODEL_HIDDEN_DIM = 512
DROPOUT_RATE_MIL = 0.5

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NTU_Dataset(
        root=DATASET_PATH,
        pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
        pre_transform=NTU_Dataset.__nturgbd_pre_transformer__,
        modality=MODALITY,
        benchmark=BENCHMARK,
        part="train",
        extended=False
    )

    if os.path.exists(FULL_FEATURES_PATH):
        print(f"Loading full features from: {FULL_FEATURES_PATH}")
        with open(FULL_FEATURES_PATH, "rb") as f:
            feature_dict = pickle.load(f)
    else:
        print("Generating features for the full dataset...")
        pre_model = ms_aagcn(num_class=NUM_CLASSES).to(device)
        model_sd, _, _, _, _ = load_model(PRE_MODEL_PATH)
        cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
        pre_model.load_state_dict(cleaned_sd, strict=True)
        pre_model.eval()
        feature_dict = extract_features(pre_model, dataset, device)
        os.makedirs(os.path.dirname(FULL_FEATURES_PATH), exist_ok=True)
        with open(FULL_FEATURES_PATH, "wb") as f:
            pickle.dump(feature_dict, f)
        print(f"Saved features to: {FULL_FEATURES_PATH}")

    input_dim = next(iter(feature_dict.values())).shape[-1]
    mil_model = MIL_GCN_Attention(input_dim, MIL_MODEL_HIDDEN_DIM, DROPOUT_RATE_MIL).to(device)
    mil_model.load_state_dict(torch.load(MIL_MODEL_PATH, map_location=device))
    mil_model.eval()
    print(f"MIL model loaded from: {MIL_MODEL_PATH}")

    bags_dict, bags_labels_dict = data_generation(dataset)
    bag_dataset = BagGraphDataset(bags_dict, bags_labels_dict, feature_dict)
    bag_loader = PyGDataLoader(bag_dataset, batch_size=256, shuffle=False)

    instance_attentions = {i: [] for i in range(len(dataset))}

    print("Extracting attention weights...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(bag_loader)):
            data = data.to(device)
            instance_ids_in_batch = []
            for batch_ptr in range(len(data.ptr) - 1):
                bag_index = i * bag_loader.batch_size + batch_ptr
                bag_id = bag_loader.dataset.bag_ids[bag_index]
                instance_ids_in_batch.extend(bag_loader.dataset.bags_dict[bag_id])
            _, attention_weights, _ = mil_model(data)
            for instance_id, att_weight in zip(instance_ids_in_batch, attention_weights):
                instance_attentions[instance_id].append(att_weight.item())

    refined_feature_dict = {}
    print("Refining features using attention weights...")
    for i in tqdm(range(len(dataset))):
        original_feature = feature_dict[i]
        att_scores = instance_attentions[i]
        avg_attention = sum(att_scores) / len(att_scores) if att_scores else 1.0
        refined_feature = original_feature * avg_attention
        refined_feature_dict[i] = refined_feature

    os.makedirs(os.path.dirname(REFINED_FEATURES_OUT_PATH), exist_ok=True)
    with open(REFINED_FEATURES_OUT_PATH, "wb") as f:
        pickle.dump(refined_feature_dict, f)
    print(f"Refined features saved to: {REFINED_FEATURES_OUT_PATH}")

if __name__ == "__main__":
    main()
