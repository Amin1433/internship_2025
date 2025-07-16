import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
import os 
import numpy as np 
import torch.optim.lr_scheduler

class MIL_GCN_Attention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.dropout(x)

        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        a = softmax(A, batch)

        z = global_add_pool(a * x, batch)
        out = self.classifier(z).squeeze(-1)

        return out, a.squeeze(), x

def evaluate_mil_model(model, dataloader, device, criterion):
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0 
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            labels = data.y.to(device).float().view(-1)
            outputs, _, _ = model(data)
            
            loss = criterion(outputs, labels) 
            total_loss += loss.item() 

            predicted_probs = torch.sigmoid(outputs)
            preds = (predicted_probs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_acc = total_correct / total if total > 0 else 0.0
    return avg_acc, avg_loss

def train_mil_model(model, train_loader, val_loader, device, n_epochs=10, lr=1e-4, debug_every=5000, prefix=None):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    history = { 
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    if prefix is not None: 
        os.makedirs(f"logs/{prefix}", exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, desc=f"Epoch {epoch + 1}/{n_epochs} Training")

        for batch_idx, (data) in train_loop:
            data = data.to(device)
            labels = data.y.to(device).float().view(-1)

            optimizer.zero_grad()
            outputs, attention_weights, node_features = model(data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted_probs = torch.sigmoid(outputs)
            batch_preds = (predicted_probs > 0.5).float()
            train_correct += (batch_preds == labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % debug_every == 0 and batch_idx > 0:
                print(f"\n[DEBUG][Batch {batch_idx}]")
                print(f" Labels: {labels.tolist()}")
                print(f" Sigmoid(outputs): {predicted_probs.detach().cpu().numpy()}")
                print(f" Raw outputs: {outputs.detach().cpu().numpy()}")
                print(f" Attention shape: {attention_weights.shape}")
                print(f" Attention mean: {attention_weights.mean().item():.4f} | std: {attention_weights.std().item():.4f}")

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["train_acc"].append(train_correct / train_total)

        val_acc, val_loss = evaluate_mil_model(model, val_loader, device, criterion)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"[Epoch {epoch + 1}/{n_epochs}] Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")


    if prefix is not None:
        np.save(f"logs/{prefix}/train_loss.npy", np.array(history["train_loss"]))
        np.save(f"logs/{prefix}/train_acc.npy", np.array(history["train_acc"]))
        np.save(f"logs/{prefix}/val_loss.npy", np.array(history["val_loss"]))
        np.save(f"logs/{prefix}/val_acc.npy", np.array(history["val_acc"]))

    return history