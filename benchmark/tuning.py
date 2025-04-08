import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parse import read_xy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "benchmark"


def process_data(cif_only, partial):
    xy = read_xy(DATA_PATH / "processed.csv", partial=partial)

    train_idx = [l.strip() for l in open(BASE_PATH / "data/train_idx.csv")][1:]
    test_idx = [l.strip() for l in open(BASE_PATH / "data/test_idx.csv")][1:]

    scaler = StandardScaler()
    xy["Ionic conductivity (S cm-1)"] = xy["Ionic conductivity (S cm-1)"].map(np.log10)

    train_xy = xy.loc[train_idx]
    test_xy = xy.loc[test_idx]
    # standardise the lattice params by fitting with training data
    scaler = scaler.fit(train_xy.iloc[:, -8:-2])
    train_xy.iloc[:, -8:-2] = scaler.transform(train_xy.iloc[:, -8:-2])
    test_xy.iloc[:, -8:-2] = scaler.transform(test_xy.iloc[:, -8:-2])

    if cif_only:
        m = train_xy[train_xy["CIF"] == "Match"]
        cm = train_xy[train_xy["CIF"] == "Close Match"]
        train_xy = pd.concat([m, cm], axis=0)
    train_xy = train_xy.drop("CIF", axis=1)

    x_train = train_xy.iloc[:, :-1].to_numpy()
    y_train = train_xy.iloc[:, -1].to_numpy()

    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train = np.array(x_train)[idx]
    y_train = np.array(y_train)[idx]

    return x_train, y_train, test_xy


def model_tune(model, x_train, y_train, cif_only, partial):
    print("Experiment Config")
    print("Model:", model)
    print("Is this CIF only (False means whole dataset)", cif_only)
    print("Are partially occupancies considered? (False means rounded values)", partial)
    print("-------------------------------------------------")
    model_str = model[:]
    if model == "MLP":
        hparams = {
            "hidden_layer_sizes": [32, 32],
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.01,
            "early_stopping": True,
            "batch_size": 32,
            "max_iter": 1000,
            "learning_rate": "adaptive",
            "n_iter_no_change": 100,
        }

        model = MLPRegressor(**hparams)

        hparams = {
            "hidden_layer_sizes": [
                [32, 32],
                [16, 16],
                [64, 64],
                [128, 128],
                [256, 256],
                [32, 32, 32],
                [16, 16, 16],
                [64, 64, 64],
                [128, 128, 128],
                [256, 256, 256],
                [32, 32, 32, 32],
                [16, 16, 16, 16],
                [64, 64, 64, 64],
                [128, 128, 128, 128],
                [256, 256, 256, 256],
                [64, 64, 64, 64, 64],
            ],
            "activation": ["relu"],
            "solver": ["adam"],
            "learning_rate_init": [0.003, 0.01, 0.03, 0.1, 0.3],
            "early_stopping": [True],
            "batch_size": [16, 32, 64],
            "max_iter": [1000],
            "learning_rate": ["adaptive"],
            "n_iter_no_change": [100],
        }

    elif model == "RF":
        hparams = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [12, 24, 36, 48, 56, 64, None],
            "max_features": ["log2", "sqrt", None],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }

        hparams_ex = {
            "max_depth": 12,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "n_estimators": 50,
        }

        model = RandomForestRegressor(**hparams_ex, oob_score="neg_mean_absolute_error")

    gs = GridSearchCV(
        estimator=model, param_grid=hparams, cv=5, scoring="neg_mean_absolute_error"
    )
    gs.fit(x_train, y_train)

    print("Best parameters:", gs.best_params_)
    print(
        f"Best {model_str} result:",
        abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
        "±",
        gs.cv_results_["std_test_score"][gs.best_index_],
    )
    plt_name = f"{model_str}_benchmark_"
    if cif_only:
        plt_name += "cifonly"
    else:
        plt_name += "whole"
    if not partial:
        plt_name += "roundPO"
    plt_name += ".png"
    try:
        plt.figure()
        plt.plot(gs.best_estimator_.loss_curve_)
        plt.yscale("log")
        plt.savefig(plt_name)
    except AttributeError:
        pass
    plt.figure()
    plt.scatter(y_train, gs.best_estimator_.predict(x_train))
    plt.plot(y_train, y_train)
    plt.savefig("parity+" + plt_name)


def test(model, params, x_train, y_train, test_xy, cif_only, partial):
    print("Experiment Config")
    print("Model:", model)
    print("Is this CIF only (False means whole dataset)", cif_only)
    print("Are partially occupancies considered? (False means rounded values)", partial)
    print("-------------------------------------------------")
    if model == "MLP":
        model = MLPRegressor(**params)
    elif model == "RF":
        model = RandomForestRegressor(**params, oob_score="neg_mean_absolute_error")

    model.fit(x_train, y_train)

    for cif_only in [True, False]:
        if cif_only:
            m = test_xy[test_xy["CIF"] == "Match"]
            cm = test_xy[test_xy["CIF"] == "Close Match"]
            test = pd.concat([m, cm], axis=0)
            test = test.drop("CIF", axis=1)
        else:
            test = test_xy.drop("CIF", axis=1)

        x_test = test.iloc[:, :-1].to_numpy()
        y_test = test.iloc[:, -1].to_numpy()
        y_pred = model.predict(x_test)
        loss = mean_absolute_error(y_test, y_pred)
        if cif_only:
            print("CIF Only", loss)
        else:
            print("Whole dataset", loss)


if __name__ == "__main__":
    model = "MLP"
    cif_only = True
    partial = True
    # params = {
    #     "activation": "relu",
    #     "batch_size": 16,
    #     "early_stopping": True,
    #     "hidden_layer_sizes": [64, 64, 64, 64, 64],
    #     "learning_rate": "adaptive",
    #     "learning_rate_init": 0.003,
    #     "max_iter": 1000,
    #     "n_iter_no_change": 100,
    #     "solver": "adam",
    # }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_train, y_train, test_xy = process_data(cif_only=cif_only, partial=partial)
        model_tune(model, x_train, y_train, cif_only=cif_only, partial=partial)
        # test(model, params, x_train, y_train, test_xy, cif_only, partial)
