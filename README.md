# Action Recognition Project

A repository for comparing Supervised, Semi-Supervised, Weakly-Supervised (MIL), and Self-Supervised learning paradigms for action recognition using an AAGCN model.

## Setup

# Create required directories for outputs
mkdir logs
mkdir models

# Structure
Files relevant to only one method are in modules/{name of the method}
The **main file** for each methods is in the project root directory.

## How to Run Experiments

All commands should be executed from the project's root directory.

### 1. Supervised (Baseline)
```bash
# Usage: python supervised.py --proportion <%_of_labeled_data>
python supervised.py --proportion 0.1
```

### 2. Semi-Supervised
```bash
# Usage: python semi_supervised.py --proportion <%_of_labeled_data>
python semi_supervised.py --proportion 0.1
```
> **Note:** Confidence thresholds must be configured manually inside the scripts in `modules/semi_supervised/`.

### 3. Weakly-Supervised (MIL)

This is a two-phase process.

**Phase 1: Train Bag Classifier**
```bash
# Usage: python weakly.py --proportion <%_of_labeled_data>
python weakly.py --proportion 0.1
```

**Phase 2: Train Final Classifier**
```bash
python final_class.py```
>  **Important:** This script may overwrite previous results. Manually rename output model files between runs.
```

### 4. Self-Supervised

This is a two-phase process.

**Phase 1: Upstream Pre-training**
```bash
# Usage: python -m modules.self_supervised.upstream --proportion <%_of_labeled_data>
python -m modules.self_supervised.upstream --proportion 0.1```
```
**Phase 2: Downstream Fine-tuning**
```bash
# Usage: python self_supervised.py --proportion <%_of_labeled_data> [--freeze-encoder]
python self_supervised.py --proportion 0.1 --freeze-encoder
```
> **Note:** The `--freeze-encoder` flag is recommended for the downstream task.

---

## Evaluating a Model

Use the `eval_model.py` script to test any saved model. 
The save model **should** be in the /models directory.

```bash
# Usage: python eval_model.py <model_file_name>
python eval_model.py "supervised_10.pt"
```

## Plotting Results

The project includes a `plot.py` script to automatically generate and save training and validation curves for all completed experiments.

The script scans the `logs/` directory. For each sub-directory (e.g., `supervised_p10`, `self_supervised_p10_frozen`), it looks for the `.npy` files containing the training history (`train_loss.npy`, `val_loss.npy`, `train_acc.npy`, `val_acc.npy`).

It then generates two plots:
-   **Loss Curve** (`loss_curve.png`)
-   **Accuracy Curve** (`accuracy_curve.png`)

These plots are saved in a newly created `graphs/` directory, mirroring the structure of your `logs/` directory.

### Special Handling for Semi-Supervised
For experiments starting with `semi_supervised`, the script will first look for `phase1`, `phase2`, etc., sub-directories. It will automatically concatenate the `.npy` files from each phase into a single continuous history and draw vertical lines on the plots to mark the boundaries between phases.

### Usage

Simply run the script from the root directory of the project. No arguments are needed.

```bash
python plot.py

