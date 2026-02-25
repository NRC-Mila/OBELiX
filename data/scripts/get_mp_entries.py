import re

import numpy as np
import pandas as pd
import requests
from mp_api.client import MPRester


def has_partial(formula_string):
    formula = re.findall("([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string)
    for element, amount in formula:
        if amount != "" and float(amount) != float(int(amount)):
            return True
    return False


def is_close(target_abc, target_angles, abc, angles, rtol=3e-2, atol=0.5):
    return (
        (
            np.isclose(abc, target_abc, rtol=rtol).all()
            and np.isclose(angles, target_angles, atol=atol).all()
        )
        or (
            np.isclose(abc[[0, 2, 1]], target_abc, rtol=rtol).all()
            and np.isclose(angles[[2, 1, 0]], target_angles, atol=atol).all()
        )
        or (
            np.isclose(abc[[1, 0, 2]], target_abc, rtol=rtol).all()
            and np.isclose(angles[[0, 2, 1]], target_angles, atol=atol).all()
        )
        or (
            np.isclose(abc[[1, 2, 0]], target_abc, rtol=rtol).all()
            and np.isclose(angles[[1, 2, 0]], target_angles, atol=atol).all()
        )
        or (
            np.isclose(abc[[2, 0, 1]], target_abc, rtol=rtol).all()
            and np.isclose(angles[[2, 0, 1]], target_angles, atol=atol).all()
        )
        or (
            np.isclose(abc[[2, 1, 0]], target_abc, rtol=rtol).all()
            and np.isclose(angles[[1, 0, 2]], target_angles, atol=atol).all()
        )
    )


data = pd.read_excel("raw_from_google.xlsx")

number_of_matches = 0
number_of_multi_matches = 0
number_of_partial_compositions = 0
with MPRester("MP_API_Key ") as mpr:

    for i, row in data.iterrows():

        if row["Cif ID"] == "done":
            continue

        possible_matches = mpr.materials.search(
            formula=row["Reduced Composition"], spacegroup_number=row["Space group #"]
        )

        if len(possible_matches) == 0:
            # print("No possible matches found for", row["ID"])
            continue

        matches = []
        for pm in possible_matches:
            for j, initial_structure in enumerate(
                pm.initial_structures + [pm.structure]
            ):
                if is_close(
                    [row["a"], row["b"], row["c"]],
                    [row["alpha"], row["beta"], row["gamma"]],
                    np.array(initial_structure.lattice.abc),
                    np.array(initial_structure.lattice.angles),
                    rtol=3e-2,
                    atol=0.5,
                ):
                    break
            else:
                continue
            matches.append((pm, initial_structure))

        if len(matches) == 1:
            print(
                "Match found for",
                row["ID"],
                row["Reduced Composition"],
                matches[0][0].material_id,
            )
            number_of_matches += 1
            if not has_partial(row["Reduced Composition"]):
                print(matches[0][1].lattice.abc, [row["a"], row["b"], row["c"]])
                matches[0][1].to(fmt="cif", filename="mp_cifs/" + row["ID"] + ".cif")
                data.at[i, "MP near match"] = matches[0][0].material_id
            else:
                print("Partial composition found for", row["ID"])
                number_of_partial_compositions += 1

        elif len(matches) > 1:
            print(
                "Multiple matches found for",
                row["ID"],
                row["Reduced Composition"],
                [m.material_id for m in matches],
            )
            number_of_multi_matches += 1

        else:
            # print("No matches found for", row["ID"])
            for pm in possible_matches:
                print(pm.material_id)
                for initial_structure in pm.initial_structures + [pm.structure]:
                    print(initial_structure.lattice.abc, [row["a"], row["b"], row["c"]])
                    print(
                        initial_structure.lattice.angles,
                        [row["alpha"], row["beta"], row["gamma"]],
                    )
                    print("---")

    print("Number of matches found:", number_of_matches)
    print("Number of multiple matches found:", number_of_multi_matches)
    print("Number of partial compositions found:", number_of_partial_compositions)

    data.to_excel("raw_from_google_mp.xlsx")
