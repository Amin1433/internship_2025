import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(log_dir, output_dir):
    """
    Plots training and validation loss and accuracy from .npy files
    and saves them as PNG images.
    """
    
    train_loss_path = os.path.join(log_dir, "train_loss.npy")
    val_loss_path = os.path.join(log_dir, "val_loss.npy")
    train_acc_path = os.path.join(log_dir, "train_acc.npy")
    val_acc_path = os.path.join(log_dir, "val_acc.npy")

    # Check if all necessary files exist
    if not (os.path.exists(train_loss_path) and os.path.exists(val_loss_path) and
            os.path.exists(train_acc_path) and os.path.exists(val_acc_path)):
        print(f"Skipping {log_dir}: Missing one or more required .npy files.")
        return

    train_loss = np.load(train_loss_path)
    val_loss = np.load(val_loss_path)
    train_acc = np.load(train_acc_path)
    val_acc = np.load(val_acc_path)

    epochs = np.arange(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {os.path.basename(log_dir)}')
    plt.legend()
    plt.grid(True)
    loss_output_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_output_path)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve for {os.path.basename(log_dir)}')
    plt.legend()
    plt.grid(True)
    acc_output_path = os.path.join(output_dir, 'accuracy_curve.png')
    plt.savefig(acc_output_path)
    plt.close()

def main():
    log_base_dir = "logs"
    graphs_base_dir = "graphs"

    if not os.path.exists(log_base_dir):
        print(f"Error: The '{log_base_dir}' directory does not exist.")
        return

    os.makedirs(graphs_base_dir, exist_ok=True)

    for subdir_name in os.listdir(log_base_dir):
        current_log_dir = os.path.join(log_base_dir, subdir_name)

        if os.path.isdir(current_log_dir):
            output_subdir = os.path.join(graphs_base_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)
            print(f"Processing directory: {current_log_dir}")
            plot_metrics(current_log_dir, output_subdir)
    
    print("Plotting complete. Check the 'graphs' directory for the generated plots.")

if __name__ == "__main__":
    main()