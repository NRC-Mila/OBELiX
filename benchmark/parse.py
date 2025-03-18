import pandas as pd
from mendeleev import get_all_elements
from pymatgen.core import Composition
from pathlib import Path


def flatten_comp(true_comp, compdf):
    for i, item in true_comp.items():
        comp = Composition(item).get_el_amt_dict()
        for k, v in comp.items():
            compdf[k][i] = v
    return compdf


def read_xy(csv_fptr, partial=True):
    data = pd.read_csv(
        csv_fptr,
        index_col="ID",
        usecols=[
            "ID",
            "Composition",
            "Space group number",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "CIF",
            "Ionic conductivity (S cm-1)",
        ],
    )

    comp_df = {}
    for e in get_all_elements():
        comp_df[e.symbol] = [0 for _ in range(len(data))]
    comp_df = pd.DataFrame(comp_df)
    comp_df = comp_df.set_index(data.index)
    comp_df = flatten_comp(data["Composition"], comp_df)
    comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]
    if not partial:
        comp_df = comp_df.round(0)
    data = data.drop("Composition", axis=1)
    data = pd.concat([comp_df, data], axis=1)
    return data


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    print(
        read_xy(
            "/home/mila/d/divya.sharma/ionic-conductivity/data/processed.csv",
            partial=False,
        )
    )
