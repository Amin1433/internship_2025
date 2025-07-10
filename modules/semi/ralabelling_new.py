from torch.utils.data import DataLoader

def label_unk(model, dataset: Subset, phase: int, device, protected_indices, prefix=None, true_labels=[], batch_size=128):
    model.eval()

    threshold = 0.95
    margin_min = 0.5  
    unlabelling = True

    indices = dataset.indices
    dataset_base = dataset.dataset

    newly_labelled_count = 0
    newly_labelled_correct = 0
    unlabeled_count = 0
    unlabeled_correct = 0
    total_labeled_count = 0
    total_labeled_correct = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"[Phase {phase}] Relabelling")):
            # batch est un objet Data, ou un tuple (x, y, ...)
            # Adapte selon la structure de tes samples
            if hasattr(batch, 'x'):
                x = batch.x.to(device)
                y = batch.y
            else:
                x = batch[0].to(device)
                y = batch[1] if len(batch) > 1 else None

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)  # (batch_size, num_classes)
            top2 = torch.topk(probs, 2, dim=1)

            for idx_in_batch, i in enumerate(indices[batch_idx*batch_size : batch_idx*batch_size + len(x)]):
                data_sample = dataset_base[i]
                confidence = top2.values[idx_in_batch, 0].item()
                margin = (top2.values[idx_in_batch, 0] - top2.values[idx_in_batch, 1]).item()
                pred_class = top2.indices[idx_in_batch, 0].item()

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