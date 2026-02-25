import sys

import numpy as np
import pandas as pd


def isnan(x):
    try:
        return np.isnan(x)
    except:
        return False


old = sys.argv[1]
old_data = pd.read_excel(old, index_col="ID")

new = sys.argv[2]
new_data = pd.read_excel(new, index_col="ID")

try:
    compare = old_data.compare(new_data, result_names=("Old", "New"))
except ValueError:

    if len(old_data.columns) < len(new_data.columns):
        print("There are new columns:")
        print([c for c in new_data.columns if c not in old_data.columns])
        common_columns = [c for c in new_data.columns if c in old_data.columns]

    elif len(old_data.columns) > len(new_data.columns):
        print("There are missing columns:")
        print([c for c in old_data.columns if c not in new_data.columns])
        common_columns = [c for c in old_data.columns if c in new_data.columns]
    else:
        newcols = new_data.columns.to_list().copy()
        common_columns = []
        for c in old_data.columns.to_list():
            if c not in newcols:
                print(f"{c} not in new")
            else:
                newcols.remove(c)
                common_columns.append(c)
        for c in newcols:
            print(f"{c} not in old")

    common_rows = set(old_data.index).intersection(set(new_data.index))
    if len(common_rows) < len(old_data):
        extra_idx = list(set(old_data.index) - common_rows)
        print("Old has the following extra rows:")
        print(old_data.loc[extra_idx])
        old_data.drop(extra_idx)

    if len(common_rows) < len(new_data):
        extra_idx = list(set(new_data.index) - common_rows)
        print("New has the following extra rows:")
        print(new_data.loc[extra_idx])
        new_data.drop(extra_idx)

    common_rows = list(common_rows)
    old_data = old_data.loc[common_rows, common_columns]
    new_data_orig = new_data.copy()
    new_data = new_data.loc[common_rows, common_columns]

    compare = old_data.compare(new_data, result_names=("Old", "New"))

changed_rows = set()
for row in compare.index:
    for c in compare.loc[row].index:
        if ~isnan(compare.loc[row][c]):
            print(f"{row} - {c} - {compare.loc[row][c]}")
            changed_rows = changed_rows.union({row})
    print()
