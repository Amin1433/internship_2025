import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
import os
import numpy as np

class MIL_GCN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.dropout(F.leaky_relu(self.conv1(x, edge_index)))
        x = self.dropout(F.leaky_relu(self.conv2(x, edge_index)))
        x = self.dropout(F.leaky_relu(self.conv3(x, edge_index)))

        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        a = softmax(A, batch)

        z = global_add_pool(a * x, batch)
        out = self.classifier(z).squeeze(-1)
        return out, a.squeeze(), x

def evaluate_mil_model(model, dataloader, device, criterion):
    model.eval()
    total_correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            labels = data.y.to(device).float().view(-1)
            outputs, _, _ = model(data)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.num_graphs

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = total_correct / total if total > 0 else 0.0
    return avg_acc, avg_loss

def train_mil_model(model, train_loader, val_loader, device, n_epochs=10, lr=1e-4, prefix=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    train_labels = np.array([data.y.item() for data in train_loader.dataset])
    num_positives = train_labels.sum()
    num_negatives = len(train_labels) - num_positives
    pos_weight = torch.tensor([num_negatives / num_positives if num_positives > 0 else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    patience = 30
    patience_counter = 0
    best_val_acc = 0.0
    best_epoch = 0

    print(f"INFO: Class balance: {int(num_positives)} pos / {int(num_negatives)} neg | Loss pos_weight: {pos_weight.item():.2f}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if prefix:
        os.makedirs(f"logs/{prefix}", exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, train_correct, train_total = 0.0, 0, 0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{n_epochs}")

        for data in train_loop:
            data = data.to(device)
            labels = data.y.to(device).float().view(-1)

            optimizer.zero_grad()
            outputs, _, _ = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * data.num_graphs
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = epoch_loss / train_total
        avg_train_acc = train_correct / train_total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        val_acc, val_loss = evaluate_mil_model(model, val_loader, device, criterion)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Early stopping (optional)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_epoch = epoch + 1
        #     if prefix:
        #         model_path = os.path.join("models", f"best_{prefix}_model.pt")
        #         torch.save(model.state_dict(), model_path)
        #         print(f"** Best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f} **")
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     print(f"No improvement. Patience counter: {patience_counter}/{patience}")
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered after {patience} epochs without improvement.")
        #     break

    if prefix:
        for key, value in history.items():
            np.save(f"logs/{prefix}/{key}.npy", np.array(value))

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    return history
