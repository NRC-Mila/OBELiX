import shutil
from sys import argv

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifFile
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

noise_level = 0.01
neighbor_cutoff = 1
max_attempts = 2

input_folder = "anon_cifs/"
output_folder = "randomized_cifs/"


def closest_shell(atom, structure, neighbor_cutoff=neighbor_cutoff):
    dists = [
        n.nn_distance
        for n in structure.get_sites_in_sphere(atom.coords, neighbor_cutoff)
        if n.label == atom.label
    ]
    return structure.get_neighbors_in_shell(atom.coords, min(dists), 1e-5)


def symmetrize(structure, new_structure, neighbor_cutoff=neighbor_cutoff):
    for atom in structure:
        coords = []
        indices = []
        for n in closest_shell(atom, new_structure, neighbor_cutoff):
            indices.append(n.index)
            coords.append(n.coords)
        new_coords = np.array(coords).mean(axis=0)
        new_structure[indices[0]].coords = new_coords
        new_structure.remove_sites(indices[1:])

    return new_structure


def add_random_noise(filename, rng):

    with open(filename, "r") as f:
        s = f.read()

    cif = CifFile.from_str(s)

    assert len(cif.data) == 1

    k = next(iter(cif.data.keys()))

    vals = []
    for dim in ["x", "y", "z"]:
        vals.append(np.array(cif.data[k].data[f"_atom_site_fract_{dim}"], dtype=float))

    vals = np.array(vals)
    mask_idx = np.arange(vals.shape[1])
    for i, pos in enumerate(vals.T):
        for j in range(i + 1, vals.shape[1]):
            if np.allclose(np.mod(pos, 1), np.mod(vals[:, j], 1), atol=1e-5):
                mask_idx[j] = mask_idx[i]

    noise = rng.normal(0, noise_level, vals.shape)
    new_vals = vals + noise[:, mask_idx]

    for i, dim in enumerate(["x", "y", "z"]):
        cif.data[k].data[f"_atom_site_fract_{dim}"] = (
            new_vals[i, :].astype(str).tolist()
        )

    new_structure = Structure.from_str(str(cif), fmt="cif")

    if len(new_structure) > len(structure):
        new_structure = symmetrize(structure, new_structure)
    elif len(new_structure) < len(structure):
        print("WARNING: number of atoms decreased!")

    return new_structure


if __name__ == "__main__":

    rng = np.random.default_rng(seed=int(argv[1]))

    data = pd.read_excel("raw.xlsx", index_col="ID")
    problems = []
    for i, row in data.iterrows():

        filename = input_folder + i + ".cif"

        if row["Cif ID"] == "done" and not np.isfinite(row["ICSD ID"]):
            shutil.copyfile(filename, output_folder + i + ".cif")
            continue
        elif row["Cif ID"] != "done":
            continue

        print("Processing", i)

        structure = Structure.from_file(filename)

        for attempt in range(max_attempts):

            new_structure = add_random_noise(filename, rng)

            if (
                len(structure) == len(new_structure)
                and structure.composition == new_structure.composition
                and SpacegroupAnalyzer(structure, symprec=2e-3).get_space_group_number()
                == SpacegroupAnalyzer(
                    new_structure, symprec=2e-3
                ).get_space_group_number()
            ):
                new_structure.to(output_folder + i + ".cif", symprec=2e-3)
                break

        else:
            new_structure.to(output_folder + i + "_problem.cif", symprec=2e-3)
            print(f"PROBLEM: {i}")
            problems.append(i)
            continue

    print(f"There were {len(problems)} problems.")
    print(problems)
