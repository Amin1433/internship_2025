import torch
from torch.utils.data import random_split, Subset
import os
from tqdm import tqdm

def label_unk(model, dataset: Subset, phase: int, device, protected_indices, prefix=None, true_labels=[]):
    model.eval()

    threshold = 1 - 0.0005 * 0.30
    margin_min = 0.995

    unlabelling = True

    indices = dataset.indices
    dataset_base = dataset.dataset

    newly_labelled_count = 0
    newly_labelled_correct = 0

    unlabeled_count = 0
    unlabeled_correct = 0

    total_labeled_count = 0
    total_labeled_correct = 0

    for i in tqdm(indices, desc=f"[Phase {phase}] Relabelling"):
        data_sample = dataset_base[i]
        
        if hasattr(data_sample, 'x') and torch.is_tensor(data_sample.x):
            sample_tensor = data_sample.x.to(device)
        elif torch.is_tensor(data_sample):
            sample_tensor = data_sample.to(device)
        else:
            raise TypeError("Expected dataset item to be a torch.Tensor or have a .x attribute that is a tensor.")


        outputs = model(sample_tensor)
        probs = torch.softmax(outputs[0], dim=0)
        top2 = torch.topk(probs, 2)

        confidence = top2.values[0].item()
        margin = (top2.values[0] - top2.values[1]).item()
        pred_class = top2.indices[0].item()

        if data_sample.y == -1 and confidence > threshold and margin > margin_min and i not in protected_indices:
            data_sample.y = int(pred_class)
            newly_labelled_count += 1
            if data_sample.y == true_labels[i]:
                newly_labelled_correct += 1

        if unlabelling and confidence < threshold and margin < margin_min and i not in protected_indices and data_sample.y != -1:
            if data_sample.y == true_labels[i]:
                unlabeled_correct += 1
            data_sample.y = -1
            unlabeled_count += 1

    for i in indices:
        if i not in protected_indices:
            data_sample = dataset_base[i]
            if data_sample.y != -1:
                total_labeled_count += 1
                if data_sample.y == true_labels[i]:
                    total_labeled_correct += 1

    text = (
        f"       Phase {phase}\n"
        f"newly labelled {newly_labelled_correct}/{newly_labelled_count} correct\n"
        f"unlabeled {unlabeled_correct}/{unlabeled_count} correct\n"
        f"total labeled data {total_labeled_correct}/{total_labeled_count} correct\n\n"
    )

    log_dir = os.path.join(".", "logs", prefix)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "labelling.txt"), "a") as f:
        f.write(text)

def recreate_datasets(dataset: Subset):
    indices = dataset.indices
    dataset_base = dataset.dataset

    labeled_indices = [i for i in indices if dataset_base[i].y != -1]
    unlabeled_indices = [i for i in indices if dataset_base[i].y == -1]

    labeled_dataset = Subset(dataset_base, labeled_indices)
    unlabeled_dataset = Subset(dataset_base, unlabeled_indices)
    return labeled_dataset, unlabeled_dataset