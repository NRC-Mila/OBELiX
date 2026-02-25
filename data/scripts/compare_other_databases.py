import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3
from preprocessing import get_paper_from_laskowski, get_paper_from_liion
from venn import venn


def add_datasets_labels(data):

    laskowski_database = "misc/laskowski_semi-fromatted.csv"
    liion_database = "misc/LiIonDatabase.csv"
    uni_datatbase = "misc/unidentified_with_refs.csv"
    doi_lookup_table_file = "misc/doi_lookup_table.csv"
    hand_database = "misc/checked_by_hand.csv"

    laskowski_data = np.genfromtxt(laskowski_database, delimiter=",", dtype=str)
    doi_lookup_table = dict(
        np.genfromtxt(doi_lookup_table_file, delimiter=",", dtype=str, skip_header=1)
    )
    liion_data = np.genfromtxt(liion_database, delimiter=",", dtype=str, skip_header=1)

    data["ICSD ID"] = None
    data["Laskowski ID"] = None
    data["Liion ID"] = None
    data["ICSD used"] = None

    for i, row in data.iterrows():

        formula = row["Reduced Composition"]

        # From databases
        la_matches, la_potential_matches = get_paper_from_laskowski(
            row["Reduced Composition"],
            row["Ionic conductivity (S cm-1)"],
            laskowski_data,
            doi_lookup_table,
            return_idx=True,
        )

        li_matches, li_potential_matches = get_paper_from_liion(
            row["Reduced Composition"],
            row["Ionic conductivity (S cm-1)"],
            liion_data,
            return_idx=True,
        )

        # From comments
        ref_id = re.findall(r"D([1,2])\s*-\s*([0-9]+)", str(row["Ref"]))
        ref_ICSD = re.findall(r"([0-9]+)-ICSD", str(row["Cif ref_1"]))
        # From cifs
        if row["Cif ID"] == "done":

            filename = "new_cifs/cifs/" + i + ".cif"
            with open(filename, "r") as f:
                s = f.read()
            cif_ICSD = re.findall(r"_database_code_ICSD\s*([0-9]+)", s)

            # Actually used the cif

            filename = "cifs/" + i + ".cif"
            with open(filename, "r") as f:
                s = f.read()
            true_cif_ICSD = re.findall(r"_database_code_ICSD\s*([0-9]+)", s)
            modified = re.findall(r"[Cc]orrected [Mm]anually", s)

        else:
            cif_ICSD = []
            true_cif_ICSD = []

        # Laskowski

        if len(la_matches) >= 1:
            if len(ref_id) != 0 and int(ref_id[0][0]) == 2:
                for match in la_matches:
                    if ref_id[0][1] in match:
                        lask_id = match
                        break
                else:
                    print(
                        f"Laskowski: Inconsistent Ref for {i}: {la_matches[0]} - {ref_id[0][1]}"
                    )
                    lask_id = la_matches[0]
            else:
                print(f"Laskowski: Multiple matches for {i} chosing first one")
                lask_id = la_matches[0]
        else:
            if len(ref_id) != 0 and int(ref_id[0][0]) == 2:
                for pot_match in la_potential_matches:
                    for match in pot_match[0]:
                        if ref_id[0][1] in match:
                            lask_id = match
                            break
                    else:
                        continue
                    break
                else:
                    print(
                        f"Laskowski: No match for {i} but reference was given. Using reference."
                    )
                    lask_id = ref_id[0][1]
            else:
                if len(la_potential_matches) > 0:
                    print(
                        f"Laskowski: No match for {i} and no reference given but potential matches were found."
                    )
                lask_id = None

        # LiIon

        if len(li_matches) >= 1:
            if len(ref_id) != 0 and int(ref_id[0][0]) == 1:
                for match in li_matches:
                    if ref_id[0][1] in match:
                        liion_id = match
                        break
                else:
                    print(
                        f"Liion: Inconsistent Ref for {i}: {li_matches[0]} - {ref_id[0][1]}"
                    )
                    liion_id = li_matches[0]
            else:
                print(f"Liion: Multiple matches for {i} chosing first one")
                liion_id = li_matches[0]
        else:
            if len(ref_id) != 0 and int(ref_id[0][0]) == 1:
                for match in la_potential_matches:
                    if ref_id[0][1] in match[0]:
                        liion_id = match[0]
                        break
                else:
                    print(
                        f"Liion: No match for {i} but reference was given. Using reference."
                    )
                    liion_id = ref_id[0][1]
            else:
                if len(li_potential_matches) > 0:
                    print(
                        f"Liion: No match for {i} and no reference given but potential matches were found."
                    )
                liion_id = None

        # ICSD

        if len(ref_ICSD) == 1 and len(cif_ICSD) == 1:
            if ref_ICSD[0] != cif_ICSD[0]:
                print(f"ICSD mismatch for {i}: {ref_ICSD[0]} - {cif_ICSD[0]}")
            ICSD_id = cif_ICSD[0]
        elif len(ref_ICSD) == 1 and len(true_cif_ICSD) == 0:
            ICSD_id = ref_ICSD[0]
        elif len(ref_ICSD) == 0 and len(true_cif_ICSD) == 1:
            ICSD_id = true_cif_ICSD[0]
        else:
            ICSD_id = None

        # ICSD used

        ICSD_used = len(true_cif_ICSD) == 1 and len(modified) == 0

        data.loc[i, "ICSD ID"] = ICSD_id
        data.loc[i, "Laskowski ID"] = lask_id
        data.loc[i, "Liion ID"] = liion_id
        data.loc[i, "ICSD used"] = ICSD_used

        if not ICSD_used and row["close match"] == True and modified == False:
            print("{i} does not use ICSD but has a close match")

    return data


if __name__ == "__main__":

    data = pd.read_excel("raw.xlsx", index_col="ID")

    data = add_datasets_labels(data)

    data.drop(columns=["ICSD used"]).to_excel("raw.xlsx")

    ICSD_entries = set(data.index[data["ICSD ID"].notna()])
    Laskowski_entries = set(data.index[data["Laskowski ID"].notna()])
    Liion_entries = set(data.index[data["Liion ID"].notna()])

    ICSD_used = set(data.index[data["ICSD used"]])

    venn3(
        (ICSD_entries, Laskowski_entries, Liion_entries), ("ICSD", "Laskowski", "Liion")
    )

    print("Number of ICSD files used as is:", len(ICSD_used))
    print("Number of ICSD files:", len(ICSD_entries))
    print(
        "Number of files not in any dataset:",
        len(data) - len(ICSD_entries | Laskowski_entries | Liion_entries),
    )

    plt.show()
