import numpy as np
import pytest

from obelix.utils import is_same_formula, replace_text_IC


# ---------------------------------------------------------------------------
# is_same_formula tests
# ---------------------------------------------------------------------------

def test_is_same_formula_identical_strings():
    """Two identical formula strings should be recognized as the same."""
    assert is_same_formula("Li7La3Zr2O12", "Li7La3Zr2O12") is True


def test_is_same_formula_reordered_elements():
    """Element order should not matter; only composition counts."""
    assert is_same_formula("Li7La3Zr2O12", "Zr2La3Li7O12") is True


def test_is_same_formula_different_stoichiometry():
    """Different element counts must be detected as different."""
    assert is_same_formula("Li7La3Zr2O12", "Li6La3Zr2O12") is False


def test_is_same_formula_implicit_vs_explicit_one():
    """Omitting the subscript '1' is equivalent to writing it explicitly."""
    assert is_same_formula("LiCl", "Li1Cl1") is True


def test_is_same_formula_scaled_formulas():
    """Formulas that reduce to the same ratio should be considered equal.

    Li2O and Li4O2 both have a 2:1 Li-to-O ratio, so their reduced
    formulas are identical.
    """
    assert is_same_formula("Li2O", "Li4O2") is True


def test_is_same_formula_fractional_coefficients():
    """Fractional stoichiometric coefficients should be handled correctly."""
    assert is_same_formula("Li0.5La0.5TiO3", "Li0.5La0.5TiO3") is True


def test_is_same_formula_single_element():
    """A single-element formula compared to itself should return True."""
    assert is_same_formula("Li", "Li") is True


def test_is_same_formula_completely_different():
    """Completely unrelated compositions must return False."""
    assert is_same_formula("NaCl", "KBr") is False


def test_is_same_formula_empty_string():
    """An empty string is not a valid formula; should return False."""
    assert is_same_formula("", "Li") is False
    assert is_same_formula("Li", "") is False
    assert is_same_formula("", "") is False


def test_is_same_formula_nonsense_string():
    """A non-chemical string should return False without raising."""
    assert is_same_formula("not_a_formula", "Li") is False
    assert is_same_formula("Li", "not_a_formula") is False


def test_is_same_formula_none_input():
    """None inputs should return False without raising."""
    assert is_same_formula(None, "Li") is False
    assert is_same_formula("Li", None) is False
    assert is_same_formula(None, None) is False


def test_is_same_formula_nan_input():
    """np.nan inputs should return False without raising."""
    assert is_same_formula(np.nan, "Li") is False
    assert is_same_formula("Li", np.nan) is False
    assert is_same_formula(np.nan, np.nan) is False


def test_is_same_formula_parenthetical_groups():
    """Parenthetical notation should be expanded correctly by pymatgen.

    Ca(OH)2 = Ca1 O2 H2, while CaO2H4 = Ca1 O2 H4.
    These have different hydrogen counts and should NOT be equal.
    """
    assert is_same_formula("Ca(OH)2", "CaO2H4") is False


def test_is_same_formula_parenthetical_equivalent():
    """Ca(OH)2 expands to CaO2H2, so it should match CaH2O2."""
    assert is_same_formula("Ca(OH)2", "CaH2O2") is True


# ---------------------------------------------------------------------------
# replace_text_IC tests
# ---------------------------------------------------------------------------

def test_replace_text_IC_below_detection_1e10():
    """The string '<1E-10' should be replaced with the default value."""
    assert replace_text_IC("<1E-10") == 1e-15


def test_replace_text_IC_below_detection_1e8():
    """The string '<1E-8' should be replaced with the default value."""
    assert replace_text_IC("<1E-8") == 1e-15


def test_replace_text_IC_numeric_string():
    """A plain numeric string should be converted to float."""
    assert replace_text_IC("0.001") == 0.001


def test_replace_text_IC_custom_value():
    """When a custom replacement value is given, it should be used."""
    assert replace_text_IC("<1E-10", value=1e-12) == 1e-12
