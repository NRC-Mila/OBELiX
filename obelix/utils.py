from pymatgen.core import Composition


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

    Uses pymatgen's Composition class for robust comparison, handling
    element reordering, implicit subscripts, scaled formulas (e.g. Li2O
    vs Li4O2), and parenthetical groups (e.g. Ca(OH)2).

    Returns False gracefully for invalid, empty, None, or NaN inputs.
    """
    try:
        c1 = Composition(formula_string1)
        c2 = Composition(formula_string2)
        return c1.reduced_formula == c2.reduced_formula
    except Exception:
        return False

