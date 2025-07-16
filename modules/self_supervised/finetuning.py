import argparse
import os
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import random
import numpy as np

from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from modules.self_supervised.tower import MsAAGCN
from modules.utils import splitting_prop

def save_model(path, model, optimizer, criterion, history, batch_size):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_function': criterion.__class__.__name__,
        'history': history,
        'batch_size': batch_size,
        'timestamp': time.time()
    }, path)

DATASET_PATH = "data/nturgb+d_skeletons/"
USE_EXTENDED_DATASET = False
MODALITY = "joint"
BENCHMARK = "xsub"
ENABLE_PRE_TRANSFORM = False
FINE_TUNE_NUM_EPOCHS = 50
FINE_TUNE_BATCH_SIZE = 64
FINE_TUNE_LR = 0.0001
RANDOM_SEED = 42
LATENT_SIZE = 256

NUM_JOINT = 25
NUM_SKELETON = 2
IN_CHANNELS = 3

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ClassifierModel(nn.Module):
    def __init__(self, pretrained_tower, num_classes, latent_size):
        super().__init__()
        self.feature_extractor = pretrained_tower

        for param in self.feature_extractor.parameters():
            param.requires_grad = False 
        
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(latent_size // 2, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier_head(features)
        return logits

def train_fine_tune(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, model_name_prefix):
    os.makedirs("models", exist_ok=True)
    
    print("\n--- Démarrage du Fine-tuning Supervisé ---")
    
    scaler = torch.cuda.amp.GradScaler()

    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0
        train_start_time = time.time()

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                sys.stdout.write(f"\rEpoch {epoch+1}/{num_epochs} Train [{batch_idx+1}/{len(train_loader)}] "
                                 f"Loss: {loss.item():.4f} | Acc: {correct_train_predictions/total_train_samples:.4f}")
                sys.stdout.flush()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = correct_train_predictions / total_train_samples
        train_end_time = time.time()
        
        model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_predictions / total_val_samples

        print(f"\rEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} "
              f"| Temps/Epoch: {train_end_time - train_start_time:.2f}s")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join("models", f"{model_name_prefix}_best_classifier.pt")
            save_model(model_save_path, model, optimizer, criterion, history, FINE_TUNE_BATCH_SIZE)
            print(f"--> Nouveau meilleur modèle sauvegardé : {model_save_path} avec Val Acc: {best_val_accuracy:.4f}")

    print("\n--- Fine-tuning terminé ---")
    
    return history

def handle_fine_tune(proportion, ssl_warm_start):
    set_seed(RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Exécution du fine-tuning sur : {device}")

    pre_transformer = None
    if ENABLE_PRE_TRANSFORM:
        pre_transformer = NTU_Dataset.__nturgbd_pre_transformer__

    total_dataset = NTU_Dataset(root=DATASET_PATH,
                                pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                                pre_transform=pre_transformer,
                                modality=MODALITY,
                                benchmark=BENCHMARK,
                                part="train",
                                extended=USE_EXTENDED_DATASET
                                )
    
    print("Décalage des labels de 1-indexé à 0-indexé...")
    for i in range(len(total_dataset)):
        if total_dataset.y[i].item() >= 1:
            total_dataset.y[i] = total_dataset.y[i] - 1 
    print("Labels décalés.")

    labeled_dataset, _ = splitting_prop(total_dataset, proportion=1.0) 

    total_labeled_indices = list(range(len(labeled_dataset))) 
    random.shuffle(total_labeled_indices)

    split_point = int(len(total_labeled_indices) * 0.7)
    train_indices = total_labeled_indices[:split_point]
    val_indices = total_labeled_indices[split_point:]

    train_subset = Subset(labeled_dataset, train_indices)
    val_subset = Subset(labeled_dataset, val_indices)

    print(f"Total samples for fine-tuning: {len(labeled_dataset)}")
    print(f"Train subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=FINE_TUNE_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = 120 if USE_EXTENDED_DATASET else 60

    ssl_tower = MsAAGCN(
        num_joint=NUM_JOINT,
        num_skeleton=NUM_SKELETON,
        edge_index=EDGE_INDEX,
        in_channels=IN_CHANNELS,
        latent_size=LATENT_SIZE
    )
    
    ssl_prefix = f"ssl_{proportion*100:.0f}%_warm_start" if ssl_warm_start else f"ssl_{proportion*100:.0f}%_from_scratch"
    ssl_model_path = os.path.join("models", f"{ssl_prefix}_siamese_tower.pt")

    if os.path.exists(ssl_model_path):
        print(f"Chargement du modèle SSL pré-entraîné (MsAAGCN) depuis {ssl_model_path}...")
        try:
            checkpoint = torch.load(ssl_model_path, map_location=device)
            ssl_tower.load_state_dict(checkpoint['model_state_dict'])
            print("Modèle SSL (MsAAGCN) chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle SSL : {e}. Le fine-tuning débutera avec un MsAAGCN initialisé aléatoirement.")
    else:
        print(f"Modèle SSL non trouvé à {ssl_model_path}. Le fine-tuning débutera avec un MsAAGCN initialisé aléatoirement.")

    model = ClassifierModel(ssl_tower, num_classes=num_classes, latent_size=LATENT_SIZE).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

    fine_tune_prefix = f"finetune_{proportion*100:.0f}%_from_ssl_{'warm_start' if ssl_warm_start else 'from_scratch'}"
    
    start_time = time.time()
    history = train_fine_tune(model, train_loader, val_loader, optimizer, criterion, device, 
                              FINE_TUNE_NUM_EPOCHS, fine_tune_prefix)
    end_time = time.time()

    print(f"Temps total de fine-tuning : {end_time - start_time:.2f} secondes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-tuning of SSL pre-trained MsAAGCN model")
    parser.add_argument("--proportion", type=float, default=1.0, 
                        help="Proportion of labeled data used in the *original SSL pre-training* (determines SSL model path).")
    parser.add_argument("--ssl-warm-start", action="store_true", 
                        help="Set this flag if the SSL model itself was warm-started from a supervised model (influences SSL model path name).")
    
    args = parser.parse_args()

    handle_fine_tune(args.proportion, args.ssl_warm_start)