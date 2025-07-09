
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
from torch_geometric.nn import GCNConv



class MIL_GCN_Attention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)


        self.dropout = nn.Dropout(p=dropout_rate)


        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.LeakyReLU(),
            self.dropout,  
            torch.nn.Linear(128, 1)
        )

        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x) 
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.dropout(x)


        a = self.attention(x)
        a = torch.softmax(a, dim=0)

 
        z = torch.sum(a * x, dim=0) 

        out = self.classifier(z) 

        return out, a.squeeze(), x


def evaluate_mil_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            labels = data.y.to(device).float() 
            outputs, _, _ = model(data) 
            
            predicted_probs = torch.sigmoid(outputs)
            preds = (predicted_probs.squeeze() > 0.5).float()

            total_correct += (preds == labels.squeeze()).sum().item()
            total += labels.size(0)

    return total_correct / total if total > 0 else 0.0


def train_mil_model(model, train_loader, val_loader, device, n_epochs=10, lr=1e-4):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model.train() 
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{n_epochs} Training")
        
        for data in train_loop: 
            data = data.to(device)

            labels = data.y.to(device).float() 

            optimizer.zero_grad()
            
            outputs, _, _ = model(data) 
            
            loss = criterion(outputs.squeeze(), labels.squeeze()) 

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            predicted_probs = torch.sigmoid(outputs).squeeze() 
            batch_preds = (predicted_probs > 0.5).float()
            
            train_correct += (batch_preds == labels.squeeze()).sum().item()
            train_total += labels.size(0)

            train_loop.set_postfix(loss=loss.item(), acc=train_correct/train_total if train_total > 0 else 0.0)

        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0

        val_acc = evaluate_mil_model(model, val_loader, device)

        print(f"[Epoch {epoch + 1}/{n_epochs}] Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_acc:.4f}")