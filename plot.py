import matplotlib.pyplot as plt
import numpy as np
import os

def concat_phase_arrays(log_dir):
    """
    If the directory starts with 'semi_supervised', concatenates .npy files
    from phase1, phase2, etc., subdirectories.
    Saves the concatenated arrays in log_dir (parent of phases).
    Also saves the cumulative lengths of each phase for plotting delimiters.
    """
    if not os.path.basename(log_dir).startswith("semi_supervised"):
        return

    arrays = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    phase_lengths = [] # To store the length of each phase's data

    # List subdirectories like phase1, phase2, ...
    phases = sorted([d for d in os.listdir(log_dir) if d.startswith("phase") and os.path.isdir(os.path.join(log_dir, d))])

    current_length_sum = 0
    for phase in phases:
        phase_dir = os.path.join(log_dir, phase)
        # We need to know the length of at least one array (e.g., train_loss) to get the phase length
        temp_phase_len = 0
        npy_path_train_loss = os.path.join(phase_dir, "train_loss.npy")
        if os.path.exists(npy_path_train_loss):
            data = np.load(npy_path_train_loss)
            temp_phase_len = len(data)
            current_length_sum += temp_phase_len
            phase_lengths.append(current_length_sum)
        else:
            print(f"Warning: {npy_path_train_loss} not found, cannot determine phase length for {phase}.")
            # If train_loss is missing, we can't reliably get the length for this phase.
            # We'll still try to concatenate other arrays but won't mark a delimiter.
            # You might want to adjust this logic based on how strictly you want to enforce all files existing.

        for key in arrays.keys():
            npy_path = os.path.join(phase_dir, f"{key}.npy")
            if os.path.exists(npy_path):
                arrays[key].append(np.load(npy_path))
            else:
                print(f"Warning: {npy_path} not found, skipping for {phase}.")

    # Concatenate each list if not empty, otherwise None, then save in log_dir
    for key in arrays:
        if arrays[key]:
            concat_arr = np.concatenate(arrays[key])
            out_path = os.path.join(log_dir, f"{key}.npy")
            np.save(out_path, concat_arr)
            print(f"Saved concatenated {key} to {out_path}")
        else:
            print(f"Warning: No arrays found for {key} in {log_dir}, nothing saved.")

    # Save the phase lengths for plotting vertical delimiters
    if phase_lengths:
        # Remove the last cumulative length as it marks the end of the entire series, not a division point.
        # We only need division points.
        delimiter_points = np.array(phase_lengths[:-1]) if len(phase_lengths) > 1 else np.array([])
        delimiter_path = os.path.join(log_dir, "phase_delimiters.npy")
        np.save(delimiter_path, delimiter_points)
        print(f"Saved phase delimiters to {delimiter_path}")
    else:
        print(f"Warning: No phase lengths found for {log_dir}, no delimiters saved.")


def plot_metrics(log_dir, output_dir):
    """
    Plots training and validation loss and accuracy from .npy files
    and saves them as PNG images.
    Plots only arrays present directly in log_dir.
    Adds vertical delimiters for 'semi_supervised' logs.
    """
    train_loss_path = os.path.join(log_dir, "train_loss.npy")
    val_loss_path = os.path.join(log_dir, "val_loss.npy")
    train_acc_path = os.path.join(log_dir, "train_acc.npy")
    val_acc_path = os.path.join(log_dir, "val_acc.npy")
    delimiter_path = os.path.join(log_dir, "phase_delimiters.npy")

    # Check if all necessary metric files exist
    if not (os.path.exists(train_loss_path) and os.path.exists(val_loss_path) and
            os.path.exists(train_acc_path) and os.path.exists(val_acc_path)):
        print(f"Skipping {os.path.basename(log_dir)}: Missing one or more required .npy files (metrics).")
        return

    train_loss = np.load(train_loss_path)
    val_loss = np.load(val_loss_path)
    train_acc = np.load(train_acc_path)
    val_acc = np.load(val_acc_path)

    epochs = np.arange(1, len(train_loss) + 1)

    # Load delimiters if they exist
    delimiters = []
    if os.path.exists(delimiter_path):
        delimiters = np.load(delimiter_path)
        print(f"Found delimiters for {os.path.basename(log_dir)}: {delimiters}")

    # Plot Loss
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    
    for d in delimiters:
        plt.axvline(x=d + 0.5, color='gray', linestyle='--', linewidth=1.5, label='Phase Boundary' if d == delimiters[0] else "")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {os.path.basename(log_dir)}')
    plt.legend()
    plt.grid(True)
    loss_output_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_output_path)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')

    for d in delimiters:
        plt.axvline(x=d + 0.5, color='gray', linestyle='--', linewidth=1.5, label='Phase Boundary' if d == delimiters[0] else "")

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
            # If 'semi_supervised', first concatenate phases and save in current_log_dir
            if subdir_name.startswith("semi_supervised"):
                print(f"\nConcatenating phases in: {current_log_dir}")
                concat_phase_arrays(current_log_dir)

            output_subdir = os.path.join(graphs_base_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)
            print(f"\nProcessing directory for plotting: {current_log_dir}")
            plot_metrics(current_log_dir, output_subdir)
    
    print("\nPlotting complete. Check the 'graphs' directory for the generated plots.")

if __name__ == "__main__":
    main()