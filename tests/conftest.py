"""
Shared pytest fixtures for the OBELiX test suite.

These fixtures provide small, deterministic, synthetic datasets for unit
testing deduplication logic, merging, formula equivalence, and edge cases.
All fixtures return plain ``pandas.DataFrame`` objects unless noted otherwise.

Fixtures that return ``Dataset`` objects are explicitly named with the
``dataset_obj_`` prefix so the distinction is always clear at the call site.
"""

import pandas as pd
import pytest

from obelix.dataset import Dataset

# ---------------------------------------------------------------------------
# Upstream pytest configuration (--rundev flag for dev tests)
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--rundev",
        action="store_true",
        help="Run the dev tests (requires GitPython and access to the private git repository)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "dev: mark test as dev test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rundev"):
        return
    skip_dev = pytest.mark.skip(
        reason="Dev only. Needs access to private repository to run. Use the --rundev option if you have access."
    )
    for item in items:
        if "dev" in item.keywords:
            item.add_marker(skip_dev)


# ---------------------------------------------------------------------------
# Core synthetic DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset_a():
    """A small synthetic dataset for unit testing.

    Contains 5 entries with a mix of compositions. One DOI (index 3, KBr)
    is intentionally ``None`` to exercise missing-DOI handling.
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl", "Li3PS4", "KBr", "Li2O"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6, 1e-3, 1e-5, 1e-7],
            "DOI": ["10.1234/a", "10.1234/b", "10.1234/c", None, "10.1234/e"],
            "Space group #": [142, 225, 36, 225, 166],
        }
    )
    return df


@pytest.fixture
def small_dataset_b():
    """A second synthetic dataset that partially overlaps with ``small_dataset_a``.

    Overlapping entries (by composition):
    - Li7La3Zr2O12 shares DOI '10.1234/a' with dataset_a  --> true duplicate
    - Li3PS4 has a *different* DOI ('10.1234/g')           --> same composition, different source

    Non-overlapping entries: LiF, CaF2.
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": ["Li7La3Zr2O12", "LiF", "Li3PS4", "CaF2"],
            "Ionic conductivity (S cm-1)": [2e-4, 5e-6, 1e-3, 3e-5],
            "DOI": ["10.1234/a", "10.1234/f", "10.1234/g", "10.1234/h"],
        }
    )
    return df


@pytest.fixture
def dataset_with_nan_dois():
    """Dataset where some DOIs are NaN -- for testing dedup with missing DOIs.

    Layout:
    - Li7La3Zr2O12 appears twice: once with DOI, once without.
    - NaCl appears twice: once with DOI, once without.

    Useful for verifying that the dedup logic correctly propagates removal to
    rows whose DOI is missing but whose composition matches a known duplicate.
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": ["Li7La3Zr2O12", "Li7La3Zr2O12", "NaCl", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 2e-4, 1e-6, 3e-6],
            "DOI": ["10.1234/a", None, "10.1234/b", None],
        }
    )
    return df


@pytest.fixture
def dataset_with_equivalent_formulas():
    """Dataset with formulas that are equivalent when reduced.

    Pairs:
    - Li2O  / Li4O2   (same reduced composition)
    - NaCl  / Na2Cl2  (same reduced composition)

    Both pairs share DOIs within the pair, so dedup should recognise them as
    duplicates if the implementation normalises formulas.
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": ["Li2O", "Li4O2", "Na2Cl2", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-7, 2e-7, 1e-6, 3e-6],
            "DOI": ["10.1234/a", "10.1234/a", "10.1234/b", "10.1234/b"],
        }
    )
    return df


@pytest.fixture
def empty_dataset():
    """An empty DataFrame with the expected column schema.

    Useful for verifying that methods handle zero-row inputs gracefully
    without raising exceptions.
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": pd.Series(dtype=str),
            "Ionic conductivity (S cm-1)": pd.Series(dtype=float),
            "DOI": pd.Series(dtype=str),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Additional edge-case DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def single_entry_dataset():
    """A DataFrame with exactly one row.

    Boundary-condition fixture: tests that iteration, dedup, indexing, and
    merge logic all handle the minimal non-empty case correctly.
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [1e-4],
            "DOI": ["10.1234/a"],
        }
    )
    return df


@pytest.fixture
def dataset_with_all_nan_dois():
    """A dataset where every DOI is NaN.

    When no DOI values are available, DOI-based dedup should be unable to
    confirm any matches and therefore should *not* remove any rows (the
    current implementation requires at least one non-NaN DOI match).
    """
    df = pd.DataFrame(
        {
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl", "Li3PS4", "Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6, 1e-3, 2e-4],
            "DOI": [None, None, None, None],
        }
    )
    return df


# ---------------------------------------------------------------------------
# Dataset *objects* (wrapping DataFrames)
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_obj_a(small_dataset_a):
    """A ``Dataset`` object wrapping ``small_dataset_a``.

    Use this when testing methods that accept a ``Dataset`` instance (as
    opposed to a raw DataFrame), such as ``remove_matching_entries``.
    """
    return Dataset(small_dataset_a)


@pytest.fixture
def dataset_obj_b(small_dataset_b):
    """A ``Dataset`` object wrapping ``small_dataset_b``.

    Use this when testing methods that accept a ``Dataset`` instance (as
    opposed to a raw DataFrame), such as ``remove_matching_entries``.
    """
    return Dataset(small_dataset_b)
