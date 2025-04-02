from __future__ import annotations

import gc
import itertools
import os
import random
import sys
from functools import partial

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matgl import load_model
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, collate_fn_graph
from matgl.models import M3GNet, SO3Net
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# Load the YAML config file
def load_config(file: str = "config.yaml"):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config


# Define seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Load the dataset from CSV and CIF files
def load_dataset(csv_file, csv_split, cif_dir, dataset_type, conductivity_column):
    """Load datasets from a CSV file and corresponding CIF files."""
    data = pd.read_csv(csv_file)
    split = pd.read_csv(csv_split, index_col="ID")
    data = data[data["ID"].isin(split.index)]
    structures = []
    ids = []
    print(f"\nLoading {dataset_type} dataset...")
    for i, row in tqdm(data.iterrows(), total=len(data)):
        try:
            struct = Structure.from_file(os.path.join(cif_dir, f'{row["ID"]}.cif'))
            structures.append(struct)
            ids.append(row["ID"])
        except OSError as e:
            print(f"Error loading structure for ID {row['ID']}: {e}")
    ionic_conductivities = np.log10(data[conductivity_column].values)
    return structures, ionic_conductivities, ids


# Create the model (M3GNet or SO3Net)
def get_model(config, model_type, structures):
    elem_list = get_element_list(structures)

    # Initialize the model based on the model_type
    if model_type == "m3gnet":
        model = M3GNet(
            element_types=elem_list,
            is_intensive=config["m3gnet"]["is_intensive"],
            readout_type=config["m3gnet"]["readout_type"],
            nblocks=config["m3gnet"]["nblocks"],
            dim_node_embedding=config["m3gnet"]["dim_node_embedding"],
            dim_edge_embedding=config["m3gnet"]["dim_edge_embedding"],
            units=config["m3gnet"]["units"],
            threebody_cutoff=config["m3gnet"]["threebody_cutoff"],
            cutoff=config["m3gnet"]["cutoff"],
        )
    elif model_type == "so3net":
        model = SO3Net(
            element_types=elem_list,
            is_intensive=config["so3net"]["is_intensive"],
            target_property=config["so3net"]["target_property"],
            readout_type=config["so3net"]["readout_type"],
            nmax=config["so3net"]["nmax"],
            lmax=config["so3net"]["lmax"],
            nblocks=config["so3net"]["nblocks"],
            dim_node_embedding=config["so3net"]["dim_node_embedding"],
            units=config["so3net"]["units"],
            cutoff=config["so3net"]["cutoff"],
            nlayers_readout=config["so3net"]["nlayers_readout"],
        )
    return model


# Get the latest version folder
def get_latest_version(log_dir):
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if versions:
        return max(versions, key=lambda x: int(x.split("_")[-1]))
    return "version_0"


# Run cross-validation with proper file naming
def run_cross_validation(
    config,
    model_type,
    max_epochs,
    lr,
    batch_size,
    train_dataset,
    train_structures,
    case_number=None,
):
    """Run k-fold cross-validation."""
    kf = KFold(
        n_splits=config["split"]["k_folds"],
        shuffle=config["split"]["shuffle"],
        random_state=config["split"]["random_state"],
    )

    mae_train = []
    mae_val = []
    loss_train = []
    loss_val = []
    epochs_range = range(1, max_epochs + 1)

    mae_train_folds = []
    mae_val_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
        print(f"\nCase Number: {case_number}, Fold {fold}/{config['split']['k_folds']}")

        train_data = Subset(train_dataset, train_idx)
        val_data = Subset(train_dataset, val_idx)

        collate_fn = partial(
            collate_fn_graph, include_line_graph=config["graph"]["include_line_graph"]
        )

        train_loader = DataLoader(
            dataset=train_data,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=config["split"]["num_workers"],
            shuffle=config["split"]["train_shuffle"],
        )
        val_loader = DataLoader(
            dataset=val_data,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=config["split"]["num_workers"],
        )

        model = get_model(config, model_type, structures=train_structures)
        lit_module = ModelLightningModule(
            model=model,
            include_line_graph=config["graph"]["include_line_graph"],
            lr=lr,
            weight_decay=config["hyperparameters"]["weight_decay"],
        )

        if model_type == "m3gnet":
            file_suffix = (
                f"case_{case_number}_m3gnet_rt_{config['m3gnet']['readout_type']}_nb_{config['m3gnet']['nblocks']}_"
                f"dimnode_{config['m3gnet']['dim_node_embedding']}_dimedge_{config['m3gnet']['dim_edge_embedding']}_"
                f"units_{config['m3gnet']['units']}_threebody_cutoff_{config['m3gnet']['threebody_cutoff']}_cutoff_{config['m3gnet']['cutoff']}_epochs_{max_epochs}_lr_{lr}"
            )
        elif model_type == "so3net":
            file_suffix = (
                f"case_{case_number}_so3net_rt_{config['so3net']['readout_type']}_nb_{config['so3net']['nblocks']}_dimnode_{config['so3net']['dim_node_embedding']}_"
                f"units_{config['so3net']['units']}_nlayers_readout_{config['so3net']['nlayers_readout']}_"
                f"nmax_{config['so3net']['nmax']}_lmax_{config['so3net']['lmax']}_cutoff_{config['so3net']['cutoff']}_"
                f"epochs_{max_epochs}_lr_{lr}"
            )

        log_dir = os.path.join(
            config["logger"]["save_dir"], f"fold_{fold}_{file_suffix}"
        )
        logger = CSVLogger(save_dir=log_dir, name="", version=None)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=config["training"]["accelerator"],
            logger=logger,
        )

        try:
            print(f"Starting training for fold {fold}")
            trainer.fit(
                model=lit_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
        except RuntimeError as e:
            print(f"Error encountered during training in fold {fold}: {e}")
            break

        torch.cuda.empty_cache()
        gc.collect()

        version_dir = get_latest_version(log_dir)
        metrics_path = os.path.join(log_dir, version_dir, "metrics.csv")

        if not os.path.exists(metrics_path):
            print(f"Metrics file for fold {fold} not found at {metrics_path}")
            continue

        metrics = pd.read_csv(metrics_path)
        val_metrics = metrics.iloc[::2].reset_index(drop=True)
        train_metrics = metrics.iloc[1::2].reset_index(drop=True)

        # Calculate MAE (pred vs actual)
        train_preds, train_actuals = [], []
        for batch in train_loader:
            graph, lat, line_graph, state_attr, labels = batch
            outputs = lit_module(
                g=graph, lat=lat, l_g=line_graph, state_attr=state_attr
            )
            train_preds.extend(outputs.detach().cpu().numpy())
            train_actuals.extend(labels.detach().cpu().numpy())

        val_preds, val_actuals = [], []
        for batch in val_loader:
            graph, lat, line_graph, state_attr, labels = batch
            outputs = lit_module(
                g=graph, lat=lat, l_g=line_graph, state_attr=state_attr
            )
            val_preds.extend(outputs.detach().cpu().numpy())
            val_actuals.extend(labels.detach().cpu().numpy())

        val_mae_fold = mean_absolute_error(val_actuals, val_preds)
        train_mae_fold = mean_absolute_error(train_actuals, train_preds)

        # Store MAEs
        mae_train_folds.append(train_mae_fold)
        mae_val_folds.append(val_mae_fold)

        # Store full MAE metrics for all epochs
        mae_train.append(train_metrics["train_MAE"].values)
        mae_val.append(val_metrics["val_MAE"].values)
        loss_train.append(train_metrics["train_Total_Loss"].values)
        loss_val.append(val_metrics["val_Total_Loss"].values)

        # Plot predictions vs actual values
        plt.figure(figsize=tuple(config["plot"]["fig_size"]))
        plt.scatter(train_actuals, train_preds, label="Train", alpha=0.7)
        plt.scatter(val_actuals, val_preds, label="Validation", alpha=0.7)
        plt.plot(
            [min(train_actuals + val_actuals), max(train_actuals + val_actuals)],
            [min(train_actuals + val_actuals), max(train_actuals + val_actuals)],
            color="black",
            linestyle="--",
            label="Ideal",
        )
        plt.xlabel("Actual Ionic Conductivity")
        plt.ylabel("Predicted Ionic Conductivity")
        plt.title(f"Predictions vs Actuals for Fold {fold}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"pred_vs_actual_fold_{fold}_case_{case_number}.png",
            dpi=config["plot"]["dpi"],
        )
        plt.close()

        print(f"Case {case_number}, Fold {fold} Train MAE: {train_mae_fold:.4f}")
        print(f"Case {case_number}, Fold {fold} Validation MAE: {val_mae_fold:.4f}")

    # Average and standard deviation for each epoch
    mae_train_avg = np.mean(mae_train, axis=0)
    mae_train_std = np.std(mae_train, axis=0)
    mae_val_avg = np.mean(mae_val, axis=0)
    mae_val_std = np.std(mae_val, axis=0)
    loss_train_avg = np.mean(loss_train, axis=0)
    loss_train_std = np.std(loss_train, axis=0)
    loss_val_avg = np.mean(loss_val, axis=0)
    loss_val_std = np.std(loss_val, axis=0)

    # MAE average and std
    avg_mae_train = np.mean(mae_train_folds)
    avg_mae_train_std = np.std(mae_train_folds)
    avg_mae_val = np.mean(mae_val_folds)
    avg_mae_val_std = np.std(mae_val_folds)

    # Plot average loss vs. epochs
    plt.figure(figsize=tuple(config["plot"]["fig_size"]))
    plt.plot(
        epochs_range, loss_train_avg, label="Avg Train Loss", linewidth=2, marker="o"
    )
    plt.plot(
        epochs_range, loss_val_avg, label="Avg Validation Loss", linewidth=2, marker="o"
    )
    plt.fill_between(
        epochs_range,
        loss_train_avg - loss_train_std,
        loss_train_avg + loss_train_std,
        alpha=0.3,
    )
    plt.fill_between(
        epochs_range,
        loss_val_avg - loss_val_std,
        loss_val_avg + loss_val_std,
        alpha=0.3,
    )
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Total Loss", fontsize=14)
    plt.title(f"Average Loss vs Epochs (lr={lr}, epochs={max_epochs})", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"Average_Loss_vs_Epochs_case_{case_number}.png", dpi=config["plot"]["dpi"]
    )

    # Plot Average Validation MAE vs. Epochs
    plt.figure(figsize=tuple(config["plot"]["fig_size"]))
    plt.plot(
        epochs_range, mae_train_avg, label="Avg Train MAE", linewidth=2, marker="o"
    )
    plt.plot(
        epochs_range, mae_val_avg, label="Avg Validation MAE", linewidth=2, marker="o"
    )
    plt.fill_between(
        epochs_range,
        mae_train_avg - mae_train_std,
        mae_train_avg + mae_train_std,
        alpha=0.3,
        label="Train MAE Std",
    )
    plt.fill_between(
        epochs_range,
        mae_val_avg - mae_val_std,
        mae_val_avg + mae_val_std,
        alpha=0.3,
        label="Validation MAE Std",
    )
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MAE", fontsize=14)
    plt.title(
        f"Average Validation MAE vs Epochs (lr={lr}, epochs={max_epochs})", fontsize=16
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"Average_Train_Val_MAE_vs_Epochs_case_{case_number}.png",
        dpi=config["plot"]["dpi"],
    )

    print(f"\nResults for Case {case_number}:")
    print(f"Average Train MAE: {avg_mae_train:.4f} ± {avg_mae_train_std:.4f}")
    print(f"Average Validation MAE: {avg_mae_val:.4f} ± {avg_mae_val_std:.4f}")

    max_val_mae_fold = max(mae_val_folds)

    return avg_mae_val, mae_val_folds, max_val_mae_fold


def fine_tune_pretrained_best_case(
    config,
    model_type,
    train_dataset,
    train_structures,
    best_hyperparameters,
    model_name,
):
    """Fine-tune the pretrained model using the best hyperparameters and recompute the average validation MAE."""
    kf = KFold(
        n_splits=config["split"]["k_folds"],
        shuffle=config["split"]["shuffle"],
        random_state=config["split"]["random_state"],
    )
    val_mae_folds = []

    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), start=1):
        print(
            f"Fine-tuning Pretrained Model for Fold {fold}/{config['split']['k_folds']}"
        )

        # Create train and validation subsets
        train_data = Subset(train_dataset, train_idx)
        val_data = Subset(train_dataset, val_idx)

        # DataLoaders
        collate_fn = partial(
            collate_fn_graph, include_line_graph=config["graph"]["include_line_graph"]
        )
        train_loader = DataLoader(
            dataset=train_data,
            collate_fn=collate_fn,
            batch_size=best_hyperparameters["batch_size"],
            num_workers=config["split"]["num_workers"],
        )
        val_loader = DataLoader(
            dataset=val_data,
            collate_fn=collate_fn,
            batch_size=best_hyperparameters["batch_size"],
            num_workers=config["split"]["num_workers"],
        )

        # Load pretrained model
        model, _ = load_pretrained_model(model_name, model_type, config)
        if model is None:
            print(f"Skipping Fold {fold}: Pretrained model could not be loaded.")
            continue

        # Fine-tune the model
        lit_module = ModelLightningModule(
            model=model,
            lr=best_hyperparameters["lr"],
            include_line_graph=config["graph"]["include_line_graph"],
        )
        logger = CSVLogger(
            save_dir=config["logger"]["save_dir"], name=f"finetuning_fold_{fold}"
        )
        trainer = pl.Trainer(
            max_epochs=best_hyperparameters["max_epochs"],
            accelerator=config["training"]["accelerator"],
            logger=logger,
        )
        trainer.fit(model=lit_module, train_dataloaders=train_loader)

        # Evaluate on validation set
        val_preds, val_actuals = [], []
        for batch in val_loader:
            graph, lat, line_graph, state_attr, labels = batch
            outputs = lit_module(
                g=graph, lat=lat, l_g=line_graph, state_attr=state_attr
            )
            val_preds.extend(outputs.detach().cpu().numpy())
            val_actuals.extend(labels.detach().cpu().numpy())

        # Calculate MAE for this fold
        val_mae_fold = mean_absolute_error(val_actuals, val_preds)
        val_mae_folds.append(val_mae_fold)
        print(f"Fold {fold} Validation MAE after Fine-Tuning: {val_mae_fold:.4f}")

    # Calculate the average and std MAE across folds
    avg_val_mae = np.mean(val_mae_folds)
    std_val_mae = np.std(val_mae_folds)

    print(
        f"Validation MAE After Fine-Tuning (Avg): {avg_val_mae:.4f} ± {std_val_mae:.4f}"
    )
    return avg_val_mae, std_val_mae, lit_module


# Search for the best hyperparameter combination
def run_hyperparameter_search(config, model_type, train_dataset, train_structures):
    """Search for the best hyperparameter combination and return the best hyperparameters and validation MAE."""
    if model_type == "m3gnet":
        hyperparameters = itertools.product(
            config["m3gnet"]["is_intensive"],
            config["m3gnet"]["readout_type"],
            config["m3gnet"]["nblocks"],
            config["m3gnet"]["dim_node_embedding"],
            config["m3gnet"]["dim_edge_embedding"],
            config["m3gnet"]["units"],
            config["m3gnet"]["threebody_cutoff"],
            config["m3gnet"]["cutoff"],
            config["hyperparameters"]["batch_size"],
            config["hyperparameters"]["lr"],
            config["hyperparameters"]["max_epochs"],
        )
    elif model_type == "so3net":
        hyperparameters = itertools.product(
            config["so3net"]["is_intensive"],
            config["so3net"]["target_property"],
            config["so3net"]["readout_type"],
            config["so3net"]["nmax"],
            config["so3net"]["lmax"],
            config["so3net"]["nblocks"],
            config["so3net"]["dim_node_embedding"],
            config["so3net"]["units"],
            config["so3net"]["cutoff"],
            config["so3net"]["nlayers_readout"],
            config["hyperparameters"]["batch_size"],
            config["hyperparameters"]["lr"],
            config["hyperparameters"]["max_epochs"],
        )

    best_val_mae = float("inf")
    best_folds_mae = None
    best_hyperparameters = None
    all_mae_values = []

    worst_case_min_max_val_mae = float("inf")
    worst_case_hyperparameters = None
    worst_case_folds_mae = None

    random_hyperparameters = random.sample(
        list(hyperparameters), config["hyperparameters"]["num_cases"]
    )

    # Initialize case number counter
    case_num = 1

    # Iterate over all hyperparameter combinations
    for params in random_hyperparameters:
        if model_type == "m3gnet":
            (
                is_intensive,
                readout_type,
                nblocks,
                dim_node_embedding,
                dim_edge_embedding,
                units,
                threebody_cutoff,
                cutoff,
                batch_size,
                lr,
                max_epochs,
            ) = params

            print(
                f"Running cross-validation with: is_intensive={is_intensive}, readout_type={readout_type}, nblocks={nblocks},dim_node_embedding={dim_node_embedding}, "
                f"dim_edge_embedding={dim_edge_embedding}, units={units}, threebody_cutoff={threebody_cutoff}, cutoff={cutoff},"
                f"batch_size={batch_size}, lr={lr}, max_epochs={max_epochs}"
            )

            # Update the config with current hyperparameters
            config["m3gnet"]["is_intensive"] = is_intensive
            config["m3gnet"]["readout_type"] = readout_type
            config["m3gnet"]["nblocks"] = nblocks
            config["m3gnet"]["dim_node_embedding"] = dim_node_embedding
            config["m3gnet"]["dim_edge_embedding"] = dim_edge_embedding
            config["m3gnet"]["units"] = units
            config["m3gnet"]["threebody_cutoff"] = threebody_cutoff
            config["m3gnet"]["cutoff"] = cutoff

        elif model_type == "so3net":
            (
                is_intensive,
                target_property,
                readout_type,
                nmax,
                lmax,
                nblocks,
                dim_node_embedding,
                units,
                cutoff,
                nlayers_readout,
                batch_size,
                lr,
                max_epochs,
            ) = params

            print(
                f"Running cross-validation with: is_intensive={is_intensive}, target_property={target_property} ,readout_type={readout_type}, nmax={nmax}, lmax={lmax}, "
                f"nblocks={nblocks}, dim_node_embedding={dim_node_embedding}, units={units}, cutoff={cutoff},"
                f"nlayers_readout={nlayers_readout}, batch_size={batch_size}, lr={lr}, max_epochs={max_epochs}"
            )

            # Update the config with current hyperparameters
            config["so3net"]["is_intensive"] = is_intensive
            config["so3net"]["target_property"] = target_property
            config["so3net"]["readout_type"] = readout_type
            config["so3net"]["nmax"] = nmax
            config["so3net"]["lmax"] = lmax
            config["so3net"]["nblocks"] = nblocks
            config["so3net"]["dim_node_embedding"] = dim_node_embedding
            config["so3net"]["units"] = units
            config["so3net"]["cutoff"] = cutoff
            config["so3net"]["nlayers_readout"] = nlayers_readout

        avg_val_mae, folds_val_mae, max_val_mae_fold = run_cross_validation(
            config,
            model_type,
            max_epochs,
            lr,
            batch_size,
            train_dataset,
            train_structures,
            case_number=case_num,
        )

        # Add hyperparameters and results to all_mae_values
        all_mae_values.append(
            {
                "is_intensive": is_intensive,
                "readout_type": readout_type,
                "nblocks": nblocks,
                "dim_node_embedding": dim_node_embedding,
                "dim_edge_embedding": (
                    dim_edge_embedding if model_type == "m3gnet" else None
                ),
                "units": units,
                "threebody_cutoff": (
                    threebody_cutoff if model_type == "m3gnet" else None
                ),
                "cutoff": cutoff,
                "target_property": target_property if model_type == "so3net" else None,
                "nmax": nmax if model_type == "so3net" else None,
                "lmax": lmax if model_type == "so3net" else None,
                "nlayers_readout": nlayers_readout if model_type == "so3net" else None,
                "batch_size": batch_size,
                "lr": lr,
                "max_epochs": max_epochs,
                "avg_val_mae": avg_val_mae,
            }
        )

        # Update best hyperparameters
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_folds_mae = folds_val_mae
            best_hyperparameters = {
                "is_intensive": is_intensive,
                "readout_type": readout_type,
                "nblocks": nblocks,
                "dim_node_embedding": dim_node_embedding,
                "dim_edge_embedding": (
                    dim_edge_embedding if model_type == "m3gnet" else None
                ),
                "units": units,
                "threebody_cutoff": (
                    threebody_cutoff if model_type == "m3gnet" else None
                ),
                "cutoff": cutoff,
                "target_property": target_property if model_type == "so3net" else None,
                "nmax": nmax if model_type == "so3net" else None,
                "lmax": lmax if model_type == "so3net" else None,
                "nlayers_readout": nlayers_readout if model_type == "so3net" else None,
                "batch_size": batch_size,
                "lr": lr,
                "max_epochs": max_epochs,
            }

        if max_val_mae_fold < worst_case_min_max_val_mae:
            worst_case_min_max_val_mae = max_val_mae_fold
            worst_case_folds_mae = folds_val_mae
            worst_case_hyperparameters = {
                "is_intensive": is_intensive,
                "readout_type": readout_type,
                "nblocks": nblocks,
                "dim_node_embedding": dim_node_embedding,
                "dim_edge_embedding": (
                    dim_edge_embedding if model_type == "m3gnet" else None
                ),
                "units": units,
                "threebody_cutoff": (
                    threebody_cutoff if model_type == "m3gnet" else None
                ),
                "cutoff": cutoff,
                "target_property": target_property if model_type == "so3net" else None,
                "nmax": nmax if model_type == "so3net" else None,
                "lmax": lmax if model_type == "so3net" else None,
                "nlayers_readout": nlayers_readout if model_type == "so3net" else None,
                "batch_size": batch_size,
                "lr": lr,
                "max_epochs": max_epochs,
            }

        # Increment case number
        case_num += 1

    # Plot the comparison of hyperparameter combinations
    plot_hyperparameter_comparison(all_mae_values, model_type)

    print(f"Best Hyperparameters for {model_type}: {best_hyperparameters}")
    print(f"Lowest Validation MAE: {best_val_mae:.4f}")

    print(f"Hyperparameters for the worst case: {worst_case_hyperparameters}")
    print(f"Worst Validation MAE: {worst_case_min_max_val_mae:.4f}")

    return (
        best_hyperparameters,
        best_val_mae,
        best_folds_mae,
        worst_case_hyperparameters,
        worst_case_min_max_val_mae,
        worst_case_folds_mae,
    )


def plot_hyperparameter_comparison(all_mae_values, model_type):
    """Simplified plot: Validation MAE for Different Hyperparameter Cases."""
    mae_list = [item["avg_val_mae"] for item in all_mae_values]
    case_numbers = list(range(1, len(mae_list) + 1))

    # Find the best case
    best_case_index = np.argmin(mae_list)
    best_case_number = case_numbers[best_case_index]
    best_mae = mae_list[best_case_index]

    fig, ax = plt.subplots(figsize=tuple(config["plot"]["fig_size"]))

    # Scatter plot for all cases
    ax.scatter(case_numbers, mae_list, color="blue", s=100, label="Cases")

    # Highlight the best case
    ax.scatter(
        best_case_number,
        best_mae,
        color="gold",
        edgecolors="black",
        s=200,
        marker="*",
        label=f"Best Case: {best_case_number} (MAE: {best_mae:.4f})",
    )

    # Annotate the best case
    ax.annotate(
        f"Best Case\n#{best_case_number}\nMAE: {best_mae:.4f}",
        xy=(best_case_number, best_mae),
        xytext=(best_case_number + 0.5, best_mae + 0.05),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

    # Axis labels and title
    ax.set_xticks(case_numbers)
    ax.set_xticklabels(case_numbers, fontsize=10)
    ax.set_xlabel("Case Number", fontsize=14)
    ax.set_ylabel("Validation MAE", fontsize=14)
    ax.set_title(
        f"Validation MAE for Different Hyperparameter Cases ({model_type})", fontsize=16
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        f"Hyperparameter_Comparison_MAE_{model_type}.png", dpi=config["plot"]["dpi"]
    )
    plt.show()


# Retrain the model on the full dataset using the best hyperparameters
def retrain_on_full_dataset(
    config, model_type, best_hyperparameters, train_dataset, train_structures
):
    collate_fn = partial(
        collate_fn_graph, include_line_graph=config["graph"]["include_line_graph"]
    )
    full_train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=best_hyperparameters["batch_size"],
        num_workers=config["split"]["num_workers"],
    )

    # Instantiate the model with the best hyperparameters
    if model_type == "m3gnet":
        config["m3gnet"].update(best_hyperparameters)
    elif model_type == "so3net":
        config["so3net"].update(best_hyperparameters)

    model = get_model(config, model_type, structures=train_structures)

    lit_module = ModelLightningModule(
        model=model,
        include_line_graph=config["graph"]["include_line_graph"],
        lr=best_hyperparameters["lr"],
        weight_decay=config["hyperparameters"]["weight_decay"],
    )

    logger = CSVLogger(
        save_dir=config["logger"]["save_dir"], name=f"{model_type}_full_training"
    )

    trainer = pl.Trainer(
        max_epochs=best_hyperparameters["max_epochs"],
        accelerator=config["training"]["accelerator"],
        logger=logger,
    )

    # Retrain the model on the full training set
    trainer.fit(model=lit_module, train_dataloaders=full_train_loader)
    return lit_module


# Test the model with test set and best hyperparameters
def run_test_evaluation(
    lit_module, test_dataset, config, best_hyperparameters, test_ids, file_suffix=""
):
    """Evaluate the model configuration on the test set and save predictions with IDs."""

    collate_fn = partial(
        collate_fn_graph, include_line_graph=config["graph"]["include_line_graph"]
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=collate_fn,
        batch_size=best_hyperparameters["batch_size"],
        num_workers=config["split"]["num_workers"],
        shuffle=config["split"]["test_shuffle"],
    )

    trainer = pl.Trainer(accelerator=config["training"]["accelerator"])

    # Generate predictions for the test set
    test_preds, test_actuals = [], []
    for batch in test_loader:
        graph, lat, line_graph, state_attr, labels = batch
        outputs = lit_module(g=graph, lat=lat, l_g=line_graph, state_attr=state_attr)

        # Collect predictions and actual labels
        test_preds.extend(outputs.detach().cpu().numpy())
        test_actuals.extend(labels.detach().cpu().numpy())

    # Ensure the number of IDs matches the actuals and predictions
    test_ids = test_ids[: len(test_actuals)]

    # Save predictions and actual values to a CSV file
    test_results_df = pd.DataFrame(
        {"ID": test_ids, "Actual": test_actuals, "Predicted": test_preds}
    )

    test_results_csv_path = f"test_predictions_{file_suffix}.csv"
    test_results_df.to_csv(test_results_csv_path, index=False)
    print(f"Test predictions saved to {test_results_csv_path}")

    # Plot predictions vs. actual values for the test set
    plt.figure(figsize=tuple(config["plot"]["fig_size"]))
    plt.scatter(test_actuals, test_preds, alpha=0.7, label="Test")
    plt.plot(
        [min(test_actuals), max(test_actuals)],
        [min(test_actuals), max(test_actuals)],
        color="black",
        linestyle="--",
        label="Ideal",
    )
    plt.xlabel("Actual Ionic Conductivity")
    plt.ylabel("Predicted Ionic Conductivity")
    plt.title("Predictions vs Actuals for Test Set")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"pred_vs_actual_test_{file_suffix}.png", dpi=config["plot"]["dpi"])
    plt.close()

    # Test on the test set
    test_results = trainer.test(model=lit_module, dataloaders=test_loader)
    test_mae = test_results[0]["test_MAE"]

    return test_mae


def load_pretrained_model(model_name, model_type, config):
    """Load a pretrained model for fine-tuning."""
    if model_type == "m3gnet":
        print(f"Loading pretrained M3GNet model: {model_name}")
        model_pretrained = load_model(model_name)

        # Handle element_refs gracefully
        property_offset = getattr(model_pretrained, "element_refs", None)
        if property_offset:
            property_offset = property_offset.property_offset
        else:
            print(
                "Warning: 'element_refs' not found in pretrained model. Proceeding without it."
            )
            property_offset = None

        model = model_pretrained.model
        return model, property_offset
    elif model_type == "so3net":
        print("Pretrained SO3Net models are not yet supported")
        return None, None
    else:
        print(f"Unknown model type: {model_type}")
        return None, None


def run_finetuning_with_hyperparameters(
    config,
    model_type,
    train_dataset,
    train_structures,
    best_hyperparameters,
    model_name,
):
    """Fine-tune the model using the best hyperparameters."""
    model, _ = load_pretrained_model(model_name, model_type, config)
    if model is None:
        print(
            "Skipping pretraining due to unsupported model or unavailable pretrained files."
        )
        return None

    collate_fn = partial(
        collate_fn_graph, include_line_graph=config["graph"]["include_line_graph"]
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=best_hyperparameters["batch_size"],
        num_workers=config["split"]["num_workers"],
    )

    lit_module_finetune = ModelLightningModule(
        model=model,
        lr=best_hyperparameters["lr"],
        include_line_graph=config["graph"]["include_line_graph"],
    )

    logger = CSVLogger(
        save_dir=config["logger"]["save_dir"],
        name=f"{model_type}_finetuning_best_hyperparameters",
    )
    trainer = pl.Trainer(
        max_epochs=best_hyperparameters["max_epochs"],
        accelerator=config["training"]["accelerator"],
        logger=logger,
    )
    trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader)

    finetuned_model_path = os.path.join(
        config["output"]["finetuned_model_dir"],
        "finetuned_model_with_best_hyperparameters",
    )
    lit_module_finetune.model.save(finetuned_model_path)
    print(f"Fine-tuned model with best hyperparameters saved to {finetuned_model_path}")

    return lit_module_finetune


# Main function
if __name__ == "__main__":
    config = load_config()
    set_random_seed(config["seed"]["seed_value"])

    # Load training and test datasets
    train_structures, train_ionic_conductivities, train_ids = load_dataset(
        config["data"]["csv_data_file"],
        config["data"]["csv_train_idx"],
        config["data"]["cif_directory"],
        "training",
        config["data"]["ionic_conductivity_column"],
    )
    test_structures, test_ionic_conductivities, test_ids = load_dataset(
        config["data"]["csv_data_file"],
        config["data"]["csv_test_idx"],
        config["data"]["cif_directory"],
        "testing",
        config["data"]["ionic_conductivity_column"],
    )

    # Convert datasets to MGLDataset format
    train_converter = Structure2Graph(
        element_types=get_element_list(train_structures),
        cutoff=config["graph"]["cutoff"],
    )

    test_converter = Structure2Graph(
        element_types=get_element_list(test_structures),
        cutoff=config["graph"]["cutoff"],
    )

    train_dataset = MGLDataset(
        threebody_cutoff=config["graph"]["threebody_cutoff"],
        structures=train_structures,
        converter=train_converter,
        labels={"ionic_conductivity": train_ionic_conductivities},
        include_line_graph=config["graph"]["include_line_graph"],
        raw_dir=config["output"]["raw_data_dir_train"],
    )

    test_dataset = MGLDataset(
        threebody_cutoff=config["graph"]["threebody_cutoff"],
        structures=test_structures,
        converter=test_converter,
        labels={"ionic_conductivity": test_ionic_conductivities},
        include_line_graph=config["graph"]["include_line_graph"],
        raw_dir=config["output"]["raw_data_dir_test"],
    )

    # Pass model type from the command line
    model_type = sys.argv[1] if len(sys.argv) > 1 else "m3gnet"

    # Run hyperparameter search to get the best hyperparameters and validation MAE
    (
        best_hyperparameters,
        best_val_mae,
        best_folds_mae,
        worst_case_hyperparameters,
        worst_case_min_max_val_mae,
        worst_case_folds_mae,
    ) = run_hyperparameter_search(config, model_type, train_dataset, train_structures)

    # Retrain the model on the full dataset with the best hyperparameters
    lit_module = retrain_on_full_dataset(
        config=config,
        model_type=model_type,
        best_hyperparameters=best_hyperparameters,
        train_dataset=train_dataset,
        train_structures=train_structures,
    )

    # Evaluate the model on the test set
    test_mae = run_test_evaluation(
        lit_module,
        test_dataset,
        config,
        best_hyperparameters,
        test_ids,
        file_suffix="avg",
    )
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Validation MAE: {best_val_mae:.4f} (best hyperparameters)")

    lit_module_worst_fold = retrain_on_full_dataset(
        config=config,
        model_type=model_type,
        best_hyperparameters=worst_case_hyperparameters,
        train_dataset=Subset(
            train_dataset,
            list(KFold(config["split"]["k_folds"]).split(train_dataset))[
                np.argmax(worst_case_folds_mae)
            ][0],
        ),
        train_structures=train_structures,
    )

    # Evaluate test set for worst fold retraining
    test_mae_worst = run_test_evaluation(
        lit_module_worst_fold,
        test_dataset,
        config,
        worst_case_hyperparameters,
        test_ids,
        file_suffix="worst",
    )
    print(f"Test MAE (worst fold): {test_mae_worst:.4f}")

    if config["pretraining"]["use_pretrained"]:
        print("Fine-tuning pretrained model for the best case...")
        avg_val_mae, std_val_mae, lit_module = fine_tune_pretrained_best_case(
            config=config,
            model_type=model_type,
            train_dataset=train_dataset,
            train_structures=train_structures,
            best_hyperparameters=best_hyperparameters,
            model_name=config["pretraining"]["model_name"],
        )
        print(
            f"Validation MAE After Fine-Tuning (Best Case): {avg_val_mae:.4f} ± {std_val_mae:.4f}"
        )

        # Evaluate the fine-tuned model on the test dataset
        test_mae = run_test_evaluation(
            lit_module,
            test_dataset,
            config,
            best_hyperparameters,
            test_ids,
            file_suffix="finetuned",
        )
        print(f"Test MAE after fine-tuning: {test_mae:.4f}")

    else:
        print("Pretraining not enabled in configuration.")

    if config["pretraining"]["use_pretrained"]:
        print("Using pretrained model with worst-case hyperparameters...")
        lit_module_worst_pretrained = run_finetuning_with_hyperparameters(
            config=config,
            model_type=model_type,
            train_dataset=train_dataset,
            train_structures=train_structures,
            best_hyperparameters=worst_case_hyperparameters,
            model_name=config["pretraining"]["model_name"],
        )

        if lit_module_worst_pretrained:
            test_mae_worst_pretrained = run_test_evaluation(
                lit_module_worst_pretrained,
                test_dataset,
                config,
                worst_case_hyperparameters,
                test_ids,
                file_suffix="finetuned_worst",
            )
            print(
                f"Test MAE after fine-tuning (worst fold): {test_mae_worst_pretrained:.4f}"
            )
        else:
            print(
                "Pretraining skipped for worst hyperparameters; test MAE after fine-tuning will not be calculated."
            )
    else:
        print("Pretraining not enabled in configuration.")
