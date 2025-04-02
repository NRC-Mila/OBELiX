import argparse
import itertools
import os
import shutil
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """Calculate the mean and std of a sample tensor."""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


# Argument parsing
parser = argparse.ArgumentParser(description="CGCNN Training Workflow")
parser.add_argument(
    "data_root",
    metavar="DIR",
    help="Path to data root directory containing CIF files and id_prop files",
)
parser.add_argument(
    "--pretrained_dir",
    default="pre-trained",
    type=str,
    help="Directory containing pre-trained files",
)
parser.add_argument(
    "--task",
    choices=["regression", "classification"],
    default="regression",
    help="Task type (default: regression)",
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="Number of data loading workers (default: 0)",
)
parser.add_argument(
    "--lr-milestones",
    default=[10, 20, 30, 40],
    nargs="+",
    type=int,
    metavar="N",
    help="milestones for scheduler (default: [15, 35])",
)
parser.add_argument(
    "--resume", default="", type=str, metavar="PATH", help="Path to latest checkpoint"
)
parser.add_argument(
    "--n-folds", default=5, type=int, help="Number of folds for cross-validation"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument(
    "--num-cases", default=100, type=int, help="Number of hyperparameter cases to test"
)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == "regression":
    best_mae_error = 1e10
else:
    best_mae_error = 0.0


def train(train_loader, model, criterion, optimizer, normalizer):

    model.train()
    losses_train, mae_errors_train = [], []
    preds_train, targets_train = [], []

    for inputs, target, _ in train_loader:
        inputs = [input.cuda() for input in inputs] if args.cuda else inputs
        target = target.cuda() if args.cuda else target

        # Normalize targets
        target_normed = normalizer.norm(target)

        # Forward pass
        optimizer.zero_grad()
        output = model(*inputs)
        loss = criterion(output, target_normed)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Record loss
        losses_train.append(loss.item())

        # Denormalize predictions and compute MAE
        output_denormed = normalizer.denorm(output.detach().cpu())
        preds_train.extend(output_denormed.numpy())
        targets_train.extend(target.cpu().numpy())
        mae_errors_train.append(mae(output_denormed.numpy(), target.cpu().numpy()))

    avg_loss_train = np.mean(losses_train)
    avg_mae_train = np.mean(mae_errors_train)
    return avg_loss_train, avg_mae_train, np.array(preds_train), np.array(targets_train)


def validate(val_loader, model, criterion, normalizer):

    model.eval()
    losses_val, mae_errors_val = [], []
    preds_val, targets_val = [], []

    with torch.no_grad():
        for inputs, target, _ in val_loader:
            inputs = [input.cuda() for input in inputs] if args.cuda else inputs
            target = target.cuda() if args.cuda else target

            # Normalize targets
            target_normed = normalizer.norm(target)

            # Forward pass
            output_val = model(*inputs)
            loss_val = criterion(output_val, target_normed)

            # Record loss
            losses_val.append(loss_val.item())

            # Denormalize predictions and collect targets
            output_denormed = normalizer.denorm(output_val.cpu())
            preds_val.extend(output_denormed.numpy())
            targets_val.extend(target.cpu().numpy())

            # Compute MAE for this batch
            mae_errors_val.append(mae(output_denormed.numpy(), target.cpu().numpy()))

    avg_loss_val = np.mean(losses_val)
    avg_mae_val = np.mean(mae_errors_val)
    return avg_loss_val, avg_mae_val, np.array(preds_val), np.array(targets_val)


# Hyperparameter search function
def hyperparameter_search(args, dataset, normalizer, pretrained_file=None):
    hyperparameter_space = {
        "atom_fea_len": [32, 64, 128, 256],
        "h_fea_len": [32, 64, 128, 256],
        "n_conv": [1, 2, 3, 4],
        "n_h": [1, 2, 3, 4],
        "lr": [1e-2, 1e-3, 5e-4, 1e-4],
        "batch_size": [35, 50, 70, 100],
        "epochs": [35, 50, 70, 100],
    }

    all_combinations = [
        dict(zip(hyperparameter_space, v))
        for v in itertools.product(*hyperparameter_space.values())
    ]
    selected_combinations = sample(all_combinations, args.num_cases)

    best_hyperparams = None
    best_val_mae = float("inf")
    best_fold_maes = None
    best_fold_indices = None

    worst_case_hyperparams = None
    worst_case_val_mae = float("-inf")
    worst_case_fold_indices = None

    avg_val_mae_with_pretraining = None

    # Loop through each hyperparameter combination
    for idx, hyperparams in enumerate(selected_combinations):
        print(f"Testing combination {idx + 1}/{args.num_cases}: {hyperparams}")

        # Perform cross-validation and calculate average validation MAE
        _, _, val_mae_per_fold, avg_val_mae, fold_indices, _ = cross_validate(
            args,
            dataset,
            hyperparams,
            normalizer,
            case_number=idx + 1,
            pretrained_file=None,
        )

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_hyperparams = hyperparams
            best_fold_maes = val_mae_per_fold
            best_fold_indices = fold_indices

        if max(val_mae_per_fold) > worst_case_val_mae:
            worst_case_val_mae = max(val_mae_per_fold)
            worst_case_hyperparams = hyperparams
            worst_case_fold_indices = fold_indices

    if pretrained_file:
        print("\nTesting best hyperparameters with pretraining:")
        _, _, _, avg_val_mae_with_pretraining, _, _ = cross_validate(
            args,
            dataset,
            best_hyperparams,
            normalizer,
            case_number="Best (Pretrained)",
            pretrained_file=pretrained_file,
        )
        print(f"Validation MAE with pretraining: {avg_val_mae_with_pretraining:.4f}")
    else:
        avg_val_mae_with_pretraining = None

    return (
        best_hyperparams,
        best_val_mae,
        best_fold_maes,
        best_fold_indices,
        worst_case_hyperparams,
        worst_case_val_mae,
        worst_case_fold_indices,
        avg_val_mae_with_pretraining,
    )


def cross_validate(
    args, dataset, hyperparams, normalizer, case_number, pretrained_file=None
):
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # Load pre-trained model if specified
    if pretrained_file:
        pretrained_path = os.path.join(args.pretrained_dir, pretrained_file)
        if args.cuda:
            checkpoint = torch.load(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

    # Lists to store losses and MAEs for all folds
    train_loss_all, val_loss_all = [], []
    train_mae_all, val_mae_all = [], []
    train_mae_per_fold = []
    val_mae_per_fold = []
    fold_indices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Case {case_number} - Starting fold {fold + 1}/{args.n_folds}")

        # Split dataset into training and validation subsets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # DataLoader for training and validation
        train_loader = DataLoader(
            train_subset,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            collate_fn=collate_pool,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            collate_fn=collate_pool,
        )

        # Initialize the model, criterion, optimizer, and scheduler
        model = build_model(hyperparams, dataset)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=0)
        scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams["epochs"])

        # Variables to store losses and MAEs per epoch
        train_loss_per_epoch, val_loss_per_epoch = [], []
        train_mae_per_epoch, val_mae_per_epoch = [], []

        # Variables to store predictions and targets for plotting
        train_preds, train_targets = None, None
        val_preds, val_targets = None, None

        # Train and validate for the specified number of epochs
        for epoch in range(1, hyperparams["epochs"] + 1):
            # Training phase
            train_loss, train_mae, train_preds, train_targets = train(
                train_loader, model, criterion, optimizer, normalizer
            )
            # Validation phase
            val_loss, val_mae, val_preds, val_targets = validate(
                val_loader, model, criterion, normalizer
            )

            # Record per-epoch metrics
            train_loss_per_epoch.append(train_loss)
            train_mae_per_epoch.append(train_mae)
            val_loss_per_epoch.append(val_loss)
            val_mae_per_epoch.append(val_mae)

            scheduler.step()

            # Print per-epoch metrics
            print(
                f"Case {case_number} - Epoch {epoch}/{hyperparams['epochs']} - Fold {fold + 1}: "
                f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
            )

        # Append metrics for the fold
        train_loss_all.append(train_loss_per_epoch)
        val_loss_all.append(val_loss_per_epoch)
        train_mae_all.append(train_mae_per_epoch)
        val_mae_all.append(val_mae_per_epoch)
        fold_indices.append((train_idx, val_idx))

        # Record the final MAE for this fold
        train_mae_per_fold.append(train_mae_per_epoch[-1])
        val_mae_per_fold.append(val_mae_per_epoch[-1])

        # Generate Predicted vs Actual plot for the fold
        plot_pred_vs_actual(
            train_preds=train_preds,
            train_targets=train_targets,
            val_preds=val_preds,
            val_targets=val_targets,
            fold=fold,
            args=args,
            hyperparams=hyperparams,
            case_number=case_number,
            pretrained_file=None,
        )

        # Plot MAE vs. Epochs for this fold
        epochs = range(1, hyperparams["epochs"] + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_mae_per_epoch, label="Train MAE", marker="o")
        plt.plot(epochs, val_mae_per_epoch, label="Validation MAE", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.title(f"MAE vs Epochs - Fold {fold + 1}")
        plt.legend()
        plt.grid()
        plt.savefig(f"mae_vs_epochs_case_{case_number}_fold_{fold + 1}.png", dpi=300)
        plt.close()

    worst_fold_index = np.argmax(val_mae_per_fold)

    # Compute the average and standard deviation across folds
    avg_train_mae = np.mean(train_mae_per_fold)
    std_train_mae = np.std(train_mae_per_fold)
    avg_val_mae = np.mean(val_mae_per_fold)
    std_val_mae = np.std(val_mae_per_fold)

    # Compute average loss and MAE per epoch across folds
    epochs = range(1, hyperparams["epochs"] + 1)
    avg_train_loss = np.mean(train_loss_all, axis=0)
    avg_val_loss = np.mean(val_loss_all, axis=0)
    avg_train_mae_per_epoch = np.mean(train_mae_all, axis=0)
    avg_val_mae_per_epoch = np.mean(val_mae_all, axis=0)

    # Plot Average Loss vs. Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, avg_val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Loss vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"avg_loss_vs_epochs_case_{case_number}.png", dpi=300)
    plt.close()

    # Plot Average MAE vs. Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_mae_per_epoch, label="Train MAE", marker="o")
    plt.plot(epochs, avg_val_mae_per_epoch, label="Validation MAE", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("Average MAE vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"avg_mae_vs_epochs_case_{case_number}.png", dpi=300)
    plt.close()

    # Print cross-validation summary
    print(f"\nCross-validation results:")
    print(f"Avg Train MAE: {avg_train_mae:.4f} ± {std_train_mae:.4f}")
    print(f"Avg Validation MAE: {avg_val_mae:.4f} ± {std_val_mae:.4f}")

    return (
        train_mae_per_fold,
        avg_train_mae,
        val_mae_per_fold,
        avg_val_mae,
        fold_indices,
        worst_fold_index,
    )


# Model builder
def build_model(hyperparams, dataset):
    structures, _, _ = dataset[0]
    orig_atom_fea_len, nbr_fea_len = structures[0].shape[-1], structures[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=hyperparams["atom_fea_len"],
        h_fea_len=hyperparams["h_fea_len"],
        n_conv=hyperparams["n_conv"],
        n_h=hyperparams["n_h"],
    )
    return model.cuda() if torch.cuda.is_available() else model


def mae(preds, targets):
    return np.mean(np.abs(preds - targets))


# Update the plot_pred_vs_actual function
def plot_pred_vs_actual(
    train_preds=None,
    train_targets=None,
    val_preds=None,
    val_targets=None,
    fold=None,
    args=None,
    hyperparams=None,
    case_number=None,
    pretrained_file=None,
):

    plt.figure(figsize=(8, 8))

    if pretrained_file:
        suffix = f"_{pretrained_file.split('.')[0]}"
        label = "Test (Pretrained)"
    elif fold == "worst":
        suffix = "_worst"
        label = "Test (Worst)"
    elif fold == "avg":
        suffix = "_avg"
        label = "Test (Average)"
    else:
        suffix = f"_case_{case_number}_fold_{fold+1}"
        label = "Validation"

    filename = f"pred_vs_actual{suffix}.png"

    # Plot Train data points
    if train_preds is not None and train_targets is not None:
        plt.scatter(train_targets, train_preds, alpha=0.5, label="Train", color="blue")

    # Plot Validation/Test data points
    if val_preds is not None and val_targets is not None:
        plt.scatter(val_targets, val_preds, alpha=0.5, label=label, color="orange")

    all_targets = []
    if train_targets is not None:
        all_targets.extend(train_targets)
    if val_targets is not None:
        all_targets.extend(val_targets)

    if len(all_targets) > 0:
        min_val, max_val = min(all_targets), max(all_targets)
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()


def train_best_model(args, best_hyperparams, dataset, normalizer, pretrained_file=None):

    # Initialize DataLoader
    train_loader = DataLoader(
        dataset, batch_size=best_hyperparams["batch_size"], collate_fn=collate_pool
    )

    # Build the model using the given hyperparameters
    model = build_model(best_hyperparams, dataset)

    # Load pre-trained model if specified
    if pretrained_file:
        pretrained_path = os.path.join(args.pretrained_dir, pretrained_file)
        if args.cuda:
            checkpoint = torch.load(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

        # Load matching layers from pre-trained weights
        pretrained_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()
        filtered_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(
            f"Loaded pre-trained model from {pretrained_file} with matching layers only."
        )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=best_hyperparams["lr"], weight_decay=0
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=best_hyperparams["epochs"])

    # Initialize tracking for the best MAE
    best_mae_error = float("inf")

    # Training loop
    for epoch in range(1, best_hyperparams["epochs"] + 1):
        train_loss, train_mae, train_preds, train_targets = train(
            train_loader, model, criterion, optimizer, normalizer
        )

        scheduler.step()

        # Check if this epoch has the best MAE
        is_best = train_mae < best_mae_error
        best_mae_error = min(train_mae, best_mae_error)

        # Save the checkpoint if this is the best model
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_mae_error": best_mae_error,
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "args": vars(args),
            },
            is_best,
        )

        # Print epoch summary
        print(
            f"Epoch {epoch}/{best_hyperparams['epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}"
        )

    return model


def evaluate_on_test_set(
    model, test_dataset, normalizer, batch_size, worst_fold=False, pretrained_file=None
):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_pool
    )

    val_loss, val_mae, preds, targets = validate(
        val_loader=test_loader,
        model=model,
        criterion=nn.MSELoss(),
        normalizer=normalizer,
    )

    # Create DataFrame for saving results
    results = pd.DataFrame({"Actual": targets.flatten(), "Predicted": preds.flatten()})

    # Save to Excel file
    suffix = (
        f"_{pretrained_file.split('.')[0]}"
        if pretrained_file
        else ("_worst" if worst_fold else "_avg")
    )
    file_name = f"test_results{suffix}.xlsx"
    results.to_excel(file_name, index=False)
    print(f"Saved test results to {file_name}")

    fold = (
        pretrained_file.split(".")[0]
        if pretrained_file
        else ("worst" if worst_fold else "avg")
    )
    plot_pred_vs_actual(
        train_preds=None,
        train_targets=None,
        val_preds=preds,
        val_targets=targets,
        fold=fold,
        pretrained_file=pretrained_file,
    )

    return val_mae


# Checkpoint
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


# Main function
def main():
    global args, best_mae_error
    set_random_seed(args.seed)

    # Load datasets for training and testing
    train_file = os.path.join(args.data_root, "train", "id_prop_train.csv")
    test_file = os.path.join(args.data_root, "test", "id_prop_test.csv")
    dataset = CIFData(root_dir=args.data_root, id_prop_file=train_file)
    test_dataset = CIFData(root_dir=args.data_root, id_prop_file=test_file)

    # Initialize normalizer with a sample of target values
    normalizer = Normalizer(torch.tensor([sample[1] for sample in dataset]))

    # Pretrained file
    pretrained_file = ["efermi.pth.tar"]

    # Hyperparameter search
    (
        best_hyperparams,
        best_val_mae,
        best_fold_maes,
        best_fold_indices,
        worst_case_hyperparams,
        worst_case_val_mae,
        worst_case_fold_indices,
        avg_val_mae_with_pretraining,
    ) = hyperparameter_search(
        args, dataset, normalizer, pretrained_file=pretrained_file[0]
    )
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Validation MAE (Best Case): {best_val_mae:.4f}")

    print(f"Worst Hyperparameters: {worst_case_hyperparams}")
    print(f"Worst Validation MAE: {worst_case_val_mae:.4f}")

    if avg_val_mae_with_pretraining is not None:
        print(f"Validation MAE (with pretraining): {avg_val_mae_with_pretraining:.4f}")

    # Train and evaluate on the worst fold's parameters
    print("\nTraining with Worst Hyperparameters:")
    worst_fold_index = np.argmax(best_fold_maes)
    worst_train_idx, _ = worst_case_fold_indices[worst_fold_index]
    worst_train_data = Subset(dataset, worst_train_idx)
    model = train_best_model(args, worst_case_hyperparams, worst_train_data, normalizer)
    test_mae_worst = evaluate_on_test_set(
        model,
        test_dataset,
        normalizer,
        worst_case_hyperparams["batch_size"],
        worst_fold=True,
        pretrained_file=None,
    )
    print(f"Test MAE (Worst Hyperparameters): {test_mae_worst:.4f}")

    # Train on the full dataset and evaluate on test set
    model = train_best_model(args, best_hyperparams, dataset, normalizer)
    avg_test_mae = evaluate_on_test_set(
        model,
        test_dataset,
        normalizer,
        best_hyperparams["batch_size"],
        worst_fold=False,
        pretrained_file=None,
    )
    print(f"Test MAE (Average): {avg_test_mae:.4f}")

    for pretrained_file in pretrained_file:
        print(f"\nEvaluating pretraining with file: {pretrained_file}")
        model = train_best_model(
            args, best_hyperparams, dataset, normalizer, pretrained_file=pretrained_file
        )
        pretrained_test_mae = evaluate_on_test_set(
            model,
            test_dataset,
            normalizer,
            best_hyperparams["batch_size"],
            worst_fold=False,
            pretrained_file=pretrained_file,
        )
        print(f"Test MAE with {pretrained_file}: {pretrained_test_mae:.4f}")


if __name__ == "__main__":
    main()
