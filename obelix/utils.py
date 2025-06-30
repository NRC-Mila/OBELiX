import numpy as np
from pymatgen.core import Composition
import re


def round_partial_occ(structure):

    structure = structure.copy()
    to_remove = []
    for i, site in enumerate(structure):   
        for k,v in site.species.as_dict().items():
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
    if cond == '<1E-10' or cond == '<1E-8':
        return value
    else:
        try:
            return float(cond)
        except ValueError:
            print("WARNING: IC is not a float:", cond)
    return cond

def is_same_formula(formula_string1, formula_string2):
    """
    Compares two formulas to determine if they represent the same composition.
    """

    f1 = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string1)
    f2 = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string2)

    f1_dict = {}
    for elem, count in f1:
        count = float(count) if count else 1  # Convert to number, default to 1
        f1_dict[elem] = f1_dict.get(elem, 0) + count

    f2_dict = {}
    for elem, count in f2:
        count = float(count) if count else 1
        f2_dict[elem] = f2_dict.get(elem, 0) + count

    return f1_dict == f2_dict

