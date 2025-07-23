import torch
from torch.utils.data import random_split, Subset
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

def label_unk(model, dataset: Subset, phase: int, device, protected_indices, prefix=None, true_labels=[]):
    model.eval()

    threshold = 0.99
    margin_min = 0.99
    unlabelling = True

    indices = [i for i in dataset.indices if i not in protected_indices]
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
            dataset_base.y[i] = int(pred_class) + 1
            newly_labelled_count += 1
            if data_sample.y == true_labels[i]:
                newly_labelled_correct += 1

        if unlabelling and confidence < threshold and margin < margin_min and i not in protected_indices and data_sample.y != -1:
            if data_sample.y == true_labels[i]:
                unlabeled_correct += 1
            dataset_base.y[i] = -1
            unlabeled_count += 1

    for i in indices:
        if i not in protected_indices:
            data_sample = dataset_base[i]
            if dataset_base.y[i] != -1:
                total_labeled_count += 1
                if data_sample.y == true_labels[i]:
                    total_labeled_correct += 1

    acc_newly = newly_labelled_correct / newly_labelled_count if newly_labelled_count != 0 else 0
    acc_unlabeled = (unlabeled_count - unlabeled_correct) / unlabeled_count if unlabeled_count != 0 else 0
    acc_total = total_labeled_correct / total_labeled_count if total_labeled_count != 0 else 0

    text = (
        f"       Phase {phase}\n"
        f"percentage of labeled data from the initial unlabeled set: {total_labeled_count / len(indices) * 100:.2f}%\n"
        f"newly labelled: {acc_newly*100:.2f}% ({newly_labelled_correct}/{newly_labelled_count} correct)\n"
        f"unlabeled: {acc_unlabeled*100:.2f}% ({unlabeled_count-unlabeled_correct}/{unlabeled_count} correctly unlabeled)\n"
        f"total labeled data: {acc_total*100:.2f}% ({total_labeled_correct}/{total_labeled_count} correct)\n\n"
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
