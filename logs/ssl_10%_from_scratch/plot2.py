import matplotlib.pyplot as plt
import numpy as np

# Charger les pertes depuis les fichiers .npy
train_losses = np.load('train_loss.npy')
val_losses = np.load('val_loss.npy')

# Tracer les courbes
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Sauvegarder le graphique
plt.savefig('loss_plot.png')
plt.close()

print("Plot enregistré dans 'loss_plot.png'")
