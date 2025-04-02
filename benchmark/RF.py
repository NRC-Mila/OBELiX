from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parse import read_xy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH / "benchmark"

cif_only = False
xy = read_xy(DATA_PATH / "processed.csv")  # , partial=False)

train_idx = [l.strip() for l in open(BASE_PATH / "data/train_idx.csv")][1:]
test_idx = [l.strip() for l in open(BASE_PATH / "data/test_idx.csv")][1:]

scaler = StandardScaler()
xy["Ionic conductivity (S cm-1)"] = xy["Ionic conductivity (S cm-1)"].map(np.log10)
train_xy = xy.loc[train_idx]
test_xy = xy.loc[test_idx]
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
    "Best RF result:",
    abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
    "±",
    gs.cv_results_["std_test_score"][gs.best_index_],
)

# Drop CIF == True
# best_params = {
#     "max_depth": 64,
#     "max_features": None,
#     "min_samples_leaf": 1,
#     "n_estimators": 200,
# }

# Drop CIF == False
# best_params = {
#     "max_depth": 36,
#     "max_features": "sqrt",
#     "min_samples_leaf": 1,
#     "n_estimators": 50,
# }

# Partial == False
# best_params = {
#     "max_depth": 36,
#     "max_features": "sqrt",
#     "min_samples_leaf": 1,
#     "n_estimators": 100,
# }

# model = RandomForestRegressor(**best_params, oob_score="neg_mean_absolute_error")
# model.fit(x_train, y_train)
# plt.plot(model.loss_curve_)

# for cif_only in [True, False]:
#     if cif_only:
#         m = test_xy[test_xy["CIF"] == "Match"]
#         cm = test_xy[test_xy["CIF"] == "Close Match"]
#         test = pd.concat([m, cm], axis=0)
#         test = test.drop("CIF", axis=1)
#     else:
#         test = test_xy.drop("CIF", axis=1)

#     x_test = test.iloc[:, :-1].to_numpy()
#     y_test = test.iloc[:, -1].to_numpy()
#     y_pred = model.predict(x_test)
#     loss = mean_absolute_error(y_test, y_pred)
#     if cif_only:
#         print("CIF Only", loss)
#     else:
#         print("Whole dataset", loss)
