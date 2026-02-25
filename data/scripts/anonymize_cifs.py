import pathlib
import re
import warnings

import numpy as np
import pandas as pd
from ase.io import read, write
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

symprec = 2e-3
make_np = True


def remove_partial_occ(structure, random=False, seed=0):
    rng = np.random.RandomState(seed)
    structure = structure.copy()
    to_remove = []
    for i, site in enumerate(structure):
        if random:
            if abs(sum(site.species.as_dict().values()) - 1) < 3e-2:
                p = list(site.species.as_dict().values()) / sum(
                    site.species.as_dict().values()
                )
                k = rng.choice(site.species.as_dict().keys(), p=p)
                new_occ = {k: 1}
                structure[i]._species = Composition(new_occ)
            else:
                p = list(site.species.as_dict().values()) + [
                    1 - sum(site.species.as_dict().values())
                ]
                keys = list(site.species.as_dict().keys()) + ["remove"]
                k = rng.choice(keys, p=p)
                if k == "remove":
                    to_remove.append(i)
                else:
                    new_occ = {k: 1}
                    structure[i]._species = Composition(new_occ)
        else:
            for k, v in site.species.as_dict().items():
                if random:
                    v = int(rng.binomial(1, min(v, 1)))
                else:
                    v = int(round(v))
                if v == 1:
                    new_occ = {k: 1}
                    structure[i]._species = Composition(new_occ)
                    break
            else:
                to_remove.append(i)
    structure.remove_sites(to_remove)
    return structure


def get_occ(atoms):
    symbols = list(atoms.symbols)
    coords = list(atoms.get_positions())
    occ_info = atoms.info.get("occupancy")
    kinds = atoms.arrays.get("spacegroup_kinds")
    occupancies = []
    if occ_info is not None and kinds is not None:
        occupancies = [occ_info[str(k)] for k in kinds]
    else:
        occupancies = [{s: 1.0} for s in symbols]
    return occupancies


def rescale_occs(atoms, tol=1e-3):
    occ_info = atoms.info.get("occupancy")
    if occ_info is None:
        return atoms

    for i, occ in occ_info.items():
        if (sum(list(occ.values())) > 1 + Composition.amount_tolerance) and (
            sum(list(occ.values())) < 1 + tol
        ):
            occ = rescale_occs_dict(occ)
            atoms.info["occupancy"][i] = occ
    return atoms


def rescale_occs_dict(occ):
    total = sum(list(occ.values()))
    print(f"Rescaling total occupancy of {total} to 1")
    for k in occ.keys():
        occ[k] /= total
    return occ


def atoms_to_structure(atoms):
    structure = AseAtomsAdaptor().get_structure(atoms)
    to_remove = []
    assert len(structure) == len(atoms)
    for i, occ in enumerate(get_occ(atoms)):
        if all(np.array(list(occ.values())) == 0):
            to_remove.append(i)
        else:
            if (sum(list(occ.values())) > 1 + Composition.amount_tolerance) and (
                sum(list(occ.values())) < 1 + comp_tol
            ):
                occ = rescale_occs(occ)
            structure[i]._species = structure[i].species.from_dict(occ)
    structure.remove_sites(to_remove)
    return structure


def ase_read(filename, **kwargs):
    filename = pathlib.Path(filename)
    with open(filename, "r") as f:
        s = f.read()
    # s = re.sub("\([0-9]+\)", "", s)
    s = re.sub("#.*", "", s)
    s = s[max(s.find("data_"), 0) :]
    tmpname = filename.with_stem(filename.stem + "_tmp")
    with open(tmpname, "w") as f:
        f.write(s)
    atoms = read(tmpname, **kwargs)
    pathlib.Path(tmpname).unlink()
    return atoms


def is_fractional(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    for elem, number in broken_down_formula:
        if number == "":
            number = "1"
        if float(number) != float(int(number)):
            return True
    return False


def process(i, folder, make_np=False):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="crystal system .*")
        atoms = ase_read(folder + i + ".cif", fractional_occupancies=True, format="cif")

    atoms = rescale_occs(atoms, tol=1.1e-3)

    structure = atoms_to_structure(atoms)

    structure.to("anon_cifs/" + i + ".cif", symprec=symprec)

    if make_np:
        np_structure = remove_partial_occ(structure)
        np_structure.to("np_cifs/" + i + ".cif", symprec=symprec)

    return atoms, structure


def close_composition(comp1, comp2, Z, tol):
    if len(comp1) != len(comp2):
        return False
    for k in comp1.keys():
        if k not in comp2.keys():
            return False
        if abs(comp1[k] / Z - comp2[k] / Z) > tol:
            return False
    return True


if __name__ == "__main__":

    data = pd.read_excel("raw.xlsx", index_col="ID")
    folder = "cifs/"
    problems = dict()
    for i, row in data.iterrows():

        if row["Cif ID"] != "done":
            continue

        print("Processing", i)

        try:
            atoms, structure = process(i, folder, make_np=make_np)
        except Exception as e:
            print("PROBELM:", e)
            problems[i] = str(e)

        if atoms is None:
            continue

        new_atoms = read("anon_cifs/" + i + ".cif", fractional_occupancies=True)

        occ_info = atoms.info.get("occupancy")
        kinds = atoms.arrays.get("spacegroup_kinds")

        new_occ_info = new_atoms.info.get("occupancy")
        new_kinds = new_atoms.arrays.get("spacegroup_kinds")

        # assert SpacegroupAnalyzer(structure, symprec=2e-3).get_space_group_number() == row["Space group #"]

        n_zero_occ = len([1 for occ in get_occ(atoms) if sum(list(occ.values())) == 0])

        if len(atoms) - n_zero_occ != len(new_atoms):
            problems[i] = "Different number of atoms"
            continue

        if new_occ_info is None and occ_info is not None:
            problems[i] = "Fractional occupancies, but none found in anonymized cif"
            continue
        elif occ_info is None:
            if is_fractional(row["Reduced Composition"]):
                problems[i] = "Fractional occupancies, but none found in original cif"
            continue

        for site, atom in occ_info.items():
            for a in new_occ_info.values():
                for k, v in atom.copy().items():
                    if v == 0.0:
                        atom.pop(k)
                if a == atom or atom == {}:
                    break
            else:
                problems[i] = "Different occupancies"
                continue

        if row["close match"] == "Yes":
            tol = 1
        else:
            tol = 0.5

        if (
            SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
            != row["Space group #"]
        ):

            if (
                SpacegroupAnalyzer(
                    structure, symprec=symprec * 2
                ).get_space_group_number()
                != row["Space group #"]
            ):
                structure.to("anon_cifs/" + i + ".cif", symprec=symprec * 2)

                if make_np:
                    np_structure = remove_partial_occ(structure)
                    np_structure.to("np_cifs/" + i + ".cif", symprec=symprec * 2)

            else:
                problems[i] = (
                    "Space group mismatch",
                    row["Space group #"],
                    SpacegroupAnalyzer(
                        structure, symprec=2e-3
                    ).get_space_group_number(),
                )
                continue

        if not close_composition(
            Composition(row["True Composition"]), structure.composition, row["Z"], tol
        ):
            problems[i] = (
                "Composition mismatch",
                row["True Composition"],
                str(structure.composition),
                row["close match"],
            )
            continue

        if row["close match"] == "Yes":
            rtol = 1e-2
        else:
            rtol = 1e-5

        if not np.allclose(
            structure.lattice.abc, (row["a"], row["b"], row["c"]), rtol=rtol
        ):
            problems[i] = (
                "Lattice mismatch",
                (row["a"], row["b"], row["c"]),
                structure.lattice.abc,
                row["close match"],
            )
            continue

        if not np.allclose(
            structure.lattice.angles,
            (row["alpha"], row["beta"], row["gamma"]),
            rtol=rtol,
        ):
            problems[i] = (
                "Angles mismatch",
                (row["alpha"], row["beta"], row["gamma"]),
                structure.lattice.angles,
                row["close match"],
            )
            continue

    print(len(problems), "problems found")
    for k, v in problems.items():
        print(k, ":", v)
