"""
Splits the data set intro train and test by keeping approximately the same distribution
of the target variable (IC) in both splits, as well disjoint papers and compositions(+spg) in each split.

Example:

    python train_test_split.py \
            --input processed.csv
            --output split.csv
            --use_log
            --do_plot
            --max_iters 100000
            --max_distance 0.42 --max_distance_cif 0.8 --ratio 0.6
            --seed 33
            --composition_only
"""

import random
import signal
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from tqdm import tqdm

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        help="Input CSV containing the data set.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output file name containing the string 'split' which will be replaced "
        "by train/test",
    )
    parser.add_argument(
        "--pct_test",
        default=0.25,
        type=float,
        help="Percentage of data points for the test set.",
    )
    parser.add_argument(
        "--min_pct_test",
        default=0.2,
        type=float,
        help="Minimum percentage of data points for the test set.",
    )
    parser.add_argument(
        "--max_pct_test",
        default=0.3,
        type=float,
        help="Maximum percentage of data points for the test set.",
    )
    parser.add_argument(
        "--target",
        default="Ionic conductivity (S cm-1)",
        type=str,
        help="Name of the column containing the target variable",
    )
    parser.add_argument(
        "--max_distance",
        default=0.5,
        type=float,
        help="Maximum Earth Mover's distance (Wasserstein) between the train and test "
        "distribution of the target variable",
    )
    parser.add_argument(
        "--max_distance_cif",
        default=0.5,
        type=float,
        help="Maximum Earth Mover's distance (Wasserstein) between the train and test "
        "distribution of the target variable for entries with CIFs",
    )
    parser.add_argument(
        "--use_log",
        default=False,
        action="store_true",
        help="If True, take the log of the target variable before stratifying.",
    )
    parser.add_argument(
        "--plain_stratify",
        default=False,
        action="store_true",
        help="If True, do a stratified split of the target variable wihout using paper, composition or spacegroup info.",
    )
    parser.add_argument(
        "--n_bins_stratify",
        default=10,
        type=int,
        help="Number of bins for the stratification of the target variable.",
    )
    parser.add_argument(
        "--composition_only",
        default=False,
        action="store_true",
        help="Use composition only instead of composition and spacegroup number.",
    )
    parser.add_argument(
        "--max_iters",
        default=10000,
        type=int,
        help="Maximum number of attempts to get a stratified split.",
    )
    parser.add_argument(
        "--do_plot",
        default=False,
        action="store_true",
        help="If True, plot distributions of the target variable for each split.",
    )

    parser.add_argument(
        "--smart_permutation",
        default=False,
        action="store_true",
        help="Whether to use smart permutation or not.",
    )
    parser.add_argument(
        "--monte_carlo",
        default=False,
        action="store_true",
        help="Whether to randomly swap two groups",
    )
    parser.add_argument(
        "--hist_dist",
        default=False,
        action="store_true",
        help="Use the histgram distance",
    )
    parser.add_argument(
        "--use_sets",
        default=False,
        action="store_true",
        help="Whether to use sets or not.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--ratio",
        default=0,
        type=float,
        help="Relative importance of the DISTANCE of the full dataset and the DISTANCE of the CIFs",
    )
    return parser


def plot(df, args, name):
    fig, axes = plt.subplots(figsize=(20, 10))
    sns.displot(data=df, x="y", hue="split", kde=True)
    plt.tight_layout()
    plt.savefig(name)


def stratified_split(y, n_bins, seed):
    indices = np.arange(len(y))
    bins = np.linspace(start=min(y), stop=max(y), num=n_bins)
    y_binned = np.digitize(y, bins)
    indices_tr, indices_tt, _, _ = train_test_split(
        indices, y, test_size=args.pct_test, stratify=y_binned, random_state=seed
    )
    return indices_tr, indices_tt


def get_directly_connected_entries(df, i, properties):

    # Entries with the same paper (there can be more than one)
    connected = []
    for paper in df["DOI"].loc[i].split("|"):
        connected += list(df.loc[df["DOI"].str.contains(paper, regex=False)].index)

    # OR

    # Entries with the same composition (AND space group number)
    connected += list(df.loc[(df[properties] == df[properties].loc[i]).all(1)].index)

    return set(connected)


def get_all_connected_entries(df, idx, i, properties):
    for j in get_directly_connected_entries(df, i, properties):
        if j not in idx:
            idx.append(j)
            idx = get_all_connected_entries(df, idx, j, properties)
    return idx


def get_groups(df, properties):
    idx = []
    allidx = set()
    while allidx != set(df.index):
        rest = set(df.index) - allidx
        el = rest.pop()
        idx.append(get_all_connected_entries(df, [el], el, properties))
        allidx = set().union(*idx)
    return idx


def find_groups_to_move(
    y, y_tr, y_tt, n_bins, groups_lookup, rng, worst=True, plot=False
):

    bins = np.histogram_bin_edges(y, bins=n_bins)
    hist_tr = np.histogram(y_tr, bins=bins)[0] / len(y_tr)
    hist_tt = np.histogram(y_tt, bins=bins)[0] / len(y_tt)

    if plot:
        plt.bar(bins[:-1], hist_tr, label="Train", alpha=0.5)
        plt.bar(bins[:-1], hist_tt, label="Test", alpha=0.5)

    nonzero_tr = np.nonzero(hist_tr)[0]
    nonzero_tt = np.nonzero(hist_tt)[0]

    if worst:
        worst_tr_bin = nonzero_tt[np.argmax(hist_tr[nonzero_tt] - hist_tt[nonzero_tt])]
    else:
        diff = hist_tr[nonzero_tt] - hist_tt[nonzero_tt]
        p = diff[diff > 0] / np.sum(diff[diff > 0])
        if len(nonzero_tt[diff > 0]) == 0:
            worst_tr_bin = len(hist_tr)  # not actually an index
        else:
            worst_tr_bin = rng.choice(nonzero_tt[diff > 0], p=p)

    tr_bin_idx = np.digitize(y_tr, bins, right=True) - 1
    tr_y_idx = y_tr.index[tr_bin_idx == worst_tr_bin]

    tr_groups = groups_lookup.loc[tr_y_idx]

    if worst:
        worst_tt_bin = nonzero_tr[np.argmax(hist_tt[nonzero_tr] - hist_tr[nonzero_tr])]
    else:
        diff = hist_tt[nonzero_tr] - hist_tr[nonzero_tr]
        p = diff[diff > 0] / np.sum(diff[diff > 0])
        if len(nonzero_tr[diff > 0]) == 0:
            worst_tt_bin = len(hist_tt)  # not actually an index
        else:
            worst_tt_bin = rng.choice(nonzero_tr[diff > 0], p=p)

    tt_bin_idx = np.digitize(y_tt, bins, right=True) - 1
    tt_y_idx = y_tt.index[tt_bin_idx == worst_tt_bin]

    tt_groups = groups_lookup.loc[tt_y_idx]

    return tr_groups[0], tt_groups[0]


def smart_permutation(
    permutation,
    n_groups_tt,
    groups,
    groups_shuffled,
    groups_lookup,
    df,
    y,
    rng,
    ratio,
    n_bins=25,
    use_sets=True,
    plot=False,
):
    groups_tt = groups_shuffled[:n_groups_tt]
    groups_tr = groups_shuffled[n_groups_tt:]

    y_tr = y.loc[[i for g in groups_tr for i in g]]
    y_tt = y.loc[[i for g in groups_tt for i in g]]

    tr_worst_groups, tt_worst_groups = find_groups_to_move(
        y, y_tr, y_tt, n_bins, groups_lookup, rng, plot=plot
    )

    y_tr_cif = y.loc[y_tr.index[df["CIF"].loc[y_tr.index] != "No Match"]]
    y_tt_cif = y.loc[y_tt.index[df["CIF"].loc[y_tt.index] != "No Match"]]
    y_cif = y.loc[df["CIF"] != "No Match"]

    cif_tr_worst_groups, cif_tt_worst_groups = find_groups_to_move(
        y_cif,
        y_tr_cif,
        y_tt_cif,
        int(n_bins * len(y_cif) / len(y)),
        groups_lookup,
        rng,
        plot=False,
    )

    if use_sets:
        tr_intersection = set(tr_worst_groups).intersection(set(cif_tr_worst_groups))
        if len(tr_intersection) > 0:
            tr_group_to_move = rng.choice(list(tr_intersection))
        else:
            p = [ratio] * len(set(tr_worst_groups)) + [1 - ratio] * len(
                set(cif_tr_worst_groups)
            )
            if np.sum(p) == 0:
                tr_group_to_move = rng.choice(permutation[n_groups_tt:])
            else:
                p = p / np.sum(p)
                tr_group_to_move = rng.choice(
                    list(set(tr_worst_groups)) + list(set(cif_tr_worst_groups)), p=p
                )

        tt_intersection = set(tt_worst_groups).intersection(set(cif_tt_worst_groups))
        if len(tt_intersection) > 0:
            tt_group_to_move = rng.choice(list(tt_intersection))
        else:
            p = [ratio] * len(set(tt_worst_groups)) + [1 - ratio] * len(
                set(cif_tt_worst_groups)
            )
            if np.sum(p) == 0:
                tt_group_to_move = rng.choice(permutation[:n_groups_tt])
            else:
                p = p / np.sum(p)
                tt_group_to_move = rng.choice(
                    list(set(tt_worst_groups)) + list(set(cif_tt_worst_groups)), p=p
                )
    else:
        p = [ratio] * len(tr_worst_groups) + [1 - ratio] * len(cif_tr_worst_groups)
        if np.sum(p) == 0:
            tr_group_to_move = rng.choice(permutation[n_groups_tt:])
        else:
            p = p / np.sum(p)
            tr_group_to_move = rng.choice(
                list(tr_worst_groups) + list(cif_tr_worst_groups), p=p
            )

        p = [ratio] * len(tt_worst_groups) + [1 - ratio] * len(cif_tt_worst_groups)
        if np.sum(p) == 0:
            tt_group_to_move = rng.choice(permutation[:n_groups_tt])
        else:
            p = p / np.sum(p)
            tt_group_to_move = rng.choice(
                list(tt_worst_groups) + list(cif_tt_worst_groups), p=p
            )

    if plot:

        bins = np.histogram_bin_edges(y, bins=n_bins)
        y_tr_to_move = y.loc[groups[tr_group_to_move]]
        y_tt_to_move = y.loc[groups[tt_group_to_move]]

        hist_tr_to_move = np.histogram(y_tr_to_move, bins=bins)[0] / len(y_tr)
        hist_tt_to_move = np.histogram(y_tt_to_move, bins=bins)[0] / len(y_tt)

        plt.bar(bins[:-1], -hist_tr_to_move, label="Train to move", alpha=0.5)
        plt.bar(bins[:-1], -hist_tt_to_move, label="Test to move", alpha=0.5)

        plt.legend()

        plt.show()

    tr_group_idx = np.argwhere(permutation == tr_group_to_move)
    tt_group_idx = np.argwhere(permutation == tt_group_to_move)

    permutation[tr_group_idx] = tt_group_to_move
    permutation[tt_group_idx] = tr_group_to_move

    return permutation


def hist_distance(y, y_tr, y_tt, n_bins=25):
    bins = np.histogram_bin_edges(y, bins=n_bins)
    hist_tr = np.histogram(y_tr, bins=bins)[0] / len(y_tr)
    hist_tt = np.histogram(y_tt, bins=bins)[0] / len(y_tt)
    return np.mean(np.abs(hist_tr - hist_tt))


def main(args):

    # Read input CSV
    df = pd.read_csv(args.input, index_col="ID")
    n_data = len(df)

    # Check duplicates
    dup_rows = df.duplicated(keep="first")

    duplicated_df = df[dup_rows]
    df = df[~dup_rows]

    n_duplicates = n_data - len(df)
    if n_duplicates > 0:
        print(f"ATTENTION: The original data set contained {n_duplicates} duplicates!")
        print(duplicated_df)

    n_data = len(df)
    n_data_w_CIF = (df["CIF"] != "No Match").sum()

    # Get target variable
    if args.use_log:
        y = np.log(df[args.target])
    else:
        y = df[args.target]

    # If plain_startify, do a plain stratified split by the target variable
    if args.plain_stratify:
        indices_tr, indices_tt = stratified_split(
            y, n_bins=args.n_bins_stratify, seed=args.seed
        )
        y_tr = y[indices_tr]
        y_tt = y[indices_tt]
        distance = wasserstein_distance(y_tr, y_tt)
        print(f"DISTANCE between splits: {distance}")

        if args.do_plot:
            split = np.array(["train"] * len(y))
            split[indices_tt] = "test"
            df_plot = pd.DataFrame.from_dict({"y": y, "split": split})
            plot(df_plot, args)
            return

    # Otherwise, try brute force splits until the desired stratification is achieved
    success = False
    min_distance = 100
    min_distance_cif = 100
    seed = args.seed

    if args.composition_only:
        groups = get_groups(df, ["Composition"])
    else:
        groups = get_groups(df, ["Composition", "Space group number"])

    print("Number of groups:", len(groups))
    print("Number of entries per group:", [len(g) for g in groups])

    n_groups = len(groups)
    n_groups_tt = int(n_groups * args.pct_test)
    groups_lookup = pd.DataFrame.from_dict(
        {idx: i for i, g in enumerate(groups) for idx in g}, orient="index"
    )
    rng = np.random.default_rng(seed=seed)
    distances = []
    distances_cif = []
    for it in tqdm(range(args.max_iters)):

        if it > 0 and args.smart_permutation:
            permutation = smart_permutation(
                best_permutation,
                n_groups_tt,
                groups,
                best_groups_shuffled,
                groups_lookup,
                df,
                y,
                rng,
                args.ratio,
                use_sets=args.use_sets,
            )
        elif it > 0:
            permutation = best_permutation.copy()

        if it > 0 and args.monte_carlo:

            tt_group_idx = rng.integers(0, n_groups_tt - 1)
            tr_group_idx = rng.integers(n_groups_tt, n_groups - 1)

            tr_group_to_move = deepcopy(permutation[tr_group_idx])

            permutation[tr_group_idx] = deepcopy(permutation[tt_group_idx])
            permutation[tt_group_idx] = tr_group_to_move

        if it == 0 or (not args.smart_permutation and not args.monte_carlo):
            permutation = rng.permutation(n_groups)
            best_permutation = deepcopy(permutation)
            best_groups_shuffled = [groups[i] for i in permutation]

        groups_shuffled = [groups[i] for i in permutation]
        groups_tt = groups_shuffled[:n_groups_tt]

        df_tt_tmp = df.loc[[i for g in groups_tt for i in g]]

        pct_test = len(df_tt_tmp) / n_data
        # If the number of data points is outside the requested range, try again
        if pct_test < args.min_pct_test or pct_test > args.max_pct_test:
            continue

        pct_test_cif = (df_tt_tmp["CIF"] != "No Match").sum() / n_data_w_CIF
        # If the number of data points with CIFs is outside the requested range, try again
        if pct_test_cif < args.min_pct_test or pct_test_cif > args.max_pct_test:
            continue

        # Check distribution of target variable
        groups_tr = groups_shuffled[n_groups_tt:]
        df_tr_tmp = df.loc[[i for g in groups_tr for i in g]]

        y_tr = y.loc[df_tr_tmp.index]
        y_tt = y.loc[df_tt_tmp.index]
        y_tr_cif = y.loc[df_tr_tmp.index[df_tr_tmp["CIF"] != "No Match"]]
        y_tt_cif = y.loc[df_tt_tmp.index[df_tt_tmp["CIF"] != "No Match"]]
        if not args.hist_dist:
            distance = wasserstein_distance(y_tr, y_tt)
            distance_cif = wasserstein_distance(y_tr_cif, y_tt_cif)
        else:
            distance = hist_distance(y, y_tr, y_tt)
            distance_cif = hist_distance(y, y_tr_cif, y_tt_cif)

        distances.append(distance)
        distances_cif.append(distance_cif)

        if (
            args.ratio * distance + (1 - args.ratio) * distance_cif
            < args.ratio * min_distance + (1 - args.ratio) * min_distance_cif
        ):
            print(
                f"New min DISTANCE: {distance}, {distance_cif}, {args.ratio*distance + (1 - args.ratio)*distance_cif}"
            )
            best_permutation = deepcopy(permutation)
            best_groups_shuffled = deepcopy(groups_shuffled)
            min_distance = distance
            min_distance_cif = distance_cif
            df_tr = df_tr_tmp.copy()
            df_tt = df_tt_tmp.copy()
        # If the DISTANCE between the train and test distribution of the target value is
        # larger than the requested minimum, try again
        if distance > args.max_distance or distance_cif > args.max_distance_cif:
            continue
        else:
            success = True
            break

    plt.plot(distances)
    plt.plot(distances_cif)

    plt.show()

    print("Average DISTANCE:", np.mean(distances), np.std(distances))
    print("Average DISTANCE CIF:", np.mean(distances_cif), np.std(distances_cif))
    if success:
        print(f"Found split with DISTANCE = {distance}, {distance_cif}!")
    else:
        print(
            f"No successful split found after {args.max_iters} iterations. Minimum "
            f"DISTANCE found: {min_distance}, {min_distance_cif}"
        )
    y_tr = y.loc[df_tr.index]
    y_tt = y.loc[df_tt.index]
    y_tr_cif = y.loc[df_tr.index[df_tr["CIF"] != "No Match"]]
    y_tt_cif = y.loc[df_tt.index[df_tt["CIF"] != "No Match"]]

    emd = wasserstein_distance(y_tr, y_tt)
    emd_cif = wasserstein_distance(y_tr_cif, y_tt_cif)

    hist_dist = hist_distance(y, y_tr, y_tt)
    hist_dist_cif = hist_distance(y, y_tr_cif, y_tt_cif)

    print()
    print(f"Final metrics full dataset EMD: {emd}, hist dist: {hist_dist}")
    print(f"Final metrics cifs: EMD: {emd_cif}, hist dist: {hist_dist_cif}")

    print(
        f"Number of training data points:",
        len(df_tr),
        (df_tr["CIF"] != "No Match").sum(),
    )
    print(
        f"Number of testing  data points:",
        len(df_tt),
        (df_tt["CIF"] != "No Match").sum(),
    )
    print(
        "Test percentage:",
        len(df_tt) / (len(df_tt) + len(df_tr)),
        (df_tt["CIF"] != "No Match").sum()
        / ((df_tt["CIF"] != "No Match").sum() + (df_tr["CIF"] != "No Match").sum()),
    )

    # Sanity checks
    # The sum of the lengths of train and test is equal to the original length
    if not len(df_tr) + len(df_tt) == len(df):
        print(
            "ATTENTION: The sum of the lengths of the train and test sets is not "
            "equal to the original length!"
        )
    # The length of the concatenation after dropping duplicates is equal to the
    # original length
    df_concat = pd.concat([df_tr, df_tt]).drop_duplicates(keep=False)
    if not len(df_concat) == len(df):
        print("ATTENTION: The train and the test set contain full duplicates!")

    # The sets of papers are disjoint
    papers_tr = set(df_tr["DOI"].unique())
    papers_tt = set(df_tt["DOI"].unique())
    papers_duplicated = papers_tr.intersection(papers_tt)
    if not len(papers_duplicated) == 0:
        print("ATTENTION: The train and the test set contain duplicate papers!")
        print(papers_duplicated)

    # The sets compositions are disjoint
    compositions_tr = set(df_tr["Composition"].unique())
    compositions_tt = set(df_tt["Composition"].unique())
    compositions_duplicated = compositions_tr.intersection(compositions_tt)
    if not len(compositions_duplicated) == 0:
        print("ATTENTION: The train and the test set contain duplicate compositions!")
        print(compositions_duplicated)

    # The sets compositions+spg are disjoint
    spg_compositions_tr = df_tr[["Composition", "Space group number"]].drop_duplicates()
    spg_compositions_tt = df_tt[["Composition", "Space group number"]].drop_duplicates()
    compositions_duplicated = pd.merge(
        spg_compositions_tr, spg_compositions_tt, how="inner"
    )
    if not len(compositions_duplicated) == 0:
        print(
            "ATTENTION: The train and the test set contain duplicate compositions+spg!"
        )
        print(compositions_duplicated)

    if args.do_plot:
        split = np.array(["train"] * len(df_tr) + ["test"] * len(df_tt))
        y = np.concatenate((np.log(df_tr[args.target]), np.log(df_tt[args.target])))
        df_plot = pd.DataFrame.from_dict({"y": y, "split": split})
        plot(df_plot, args, args.output.replace("csv", "pdf"))

        split = np.array(
            ["train cifs"] * (df_tr["CIF"] != "No Match").sum()
            + ["test cifs"] * (df_tt["CIF"] != "No Match").sum()
        )
        y = np.concatenate(
            (
                np.log(df_tr[args.target][df_tr["CIF"] != "No Match"]),
                np.log(df_tt[args.target][df_tt["CIF"] != "No Match"]),
            )
        )
        df_plot = pd.DataFrame.from_dict({"y": y, "split": split})
        plot(df_plot, args, "cif_" + args.output.replace("csv", "pdf"))

    output_tr = args.output.replace("split", "train")
    output_tt = args.output.replace("split", "test")
    df_tr.to_csv(output_tr, index=True)
    df_tt.to_csv(output_tt, index=True)


def args2yaml(args):
    """
    Writes the arguments in  YAML file with the name of the output file, with extension
    .yaml.

    Args:
        args (argparse.Namespace): the parsed arguments
    """
    output = Path(args.output)
    output_yaml = output.parent / ("arguments_" + output.stem + ".yaml")
    with open(output_yaml, "w") as f:
        yaml.dump(dict(vars(args)), f)


def print_args(args):
    """
    Prints the arguments

    Args:
        args (argparse.Namespace): the parsed arguments
    """
    print("Arguments:")
    darg = vars(args)
    max_k = max([len(k) for k in darg])
    for k in darg:
        print(f"\t{k:{max_k}}: {darg[k]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    print_args(args)
    main(args)
    args2yaml(args)
    sys.exit()
