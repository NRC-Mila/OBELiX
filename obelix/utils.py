import numpy as np
from pymatgen.core import Composition


def round_partial_occ(structure):

    structure = structure.copy()
    to_remove = []
    for i, site in enumerate(structure):
        for k, v in site.species.as_dict().items():
            v = int(round(v))
            if v == 1:
                new_occ = {k: 1}
                structure[i]._species = Composition(new_occ)
                break
        else:
            to_remove.append(i)
    structure.remove_sites(to_remove)
    return structure


def replace_text_IC(cond, value=1e-15):
    if cond == "<1E-10" or cond == "<1E-8":
        return value
    else:
        try:
            return float(cond)
        except ValueError:
            print("WARNING: IC is not a float:", cond)
    return cond
