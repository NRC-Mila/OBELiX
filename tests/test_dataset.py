"""
Unit tests for the Dataset base class.

These tests use synthetic DataFrames only -- no data downloads required.
Tests are written in TDD style: some target the NEW API (e.g., __add__,
union) and will fail until those methods are implemented.
"""

import pytest
import pandas as pd
import numpy as np

from obelix.dataset import Dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """A minimal DataFrame with 4 entries and the standard columns."""
    return pd.DataFrame({
        "Reduced Composition": ["Li7La3Zr2O12", "NaCl", "Li3PS4", "KBr"],
        "Ionic conductivity (S cm-1)": [1e-4, 1e-6, 1e-3, 1e-5],
        "DOI": ["10.1234/a", "10.1234/b", "10.1234/c", None],
    })


@pytest.fixture
def sample_dataset(sample_df):
    """Dataset wrapping sample_df."""
    return Dataset(sample_df)


@pytest.fixture
def other_df():
    """A second DataFrame for merge / remove tests."""
    return pd.DataFrame({
        "Reduced Composition": ["Li7La3Zr2O12", "MgO", "Li3PS4"],
        "Ionic conductivity (S cm-1)": [2e-4, 5e-7, 3e-3],
        "DOI": ["10.1234/a", "10.1234/d", "10.1234/x"],
    })


@pytest.fixture
def other_dataset(other_df):
    return Dataset(other_df)


@pytest.fixture
def extra_cols_df():
    """DataFrame with an extra column not present in sample_df."""
    return pd.DataFrame({
        "Reduced Composition": ["CaF2", "BaTiO3"],
        "Ionic conductivity (S cm-1)": [1e-8, 2e-7],
        "DOI": ["10.1234/e", "10.1234/f"],
        "Space group #": [225, 221],
    })


@pytest.fixture
def extra_cols_dataset(extra_cols_df):
    return Dataset(extra_cols_df)


@pytest.fixture
def empty_df():
    """An empty DataFrame with the standard columns."""
    return pd.DataFrame({
        "Reduced Composition": pd.Series([], dtype="object"),
        "Ionic conductivity (S cm-1)": pd.Series([], dtype="float64"),
        "DOI": pd.Series([], dtype="object"),
    })


@pytest.fixture
def empty_dataset(empty_df):
    return Dataset(empty_df)


@pytest.fixture
def duplicate_composition_df():
    """DataFrame where two rows have compositions that reduce to the same
    canonical formula: Li2O and Li4O2 both reduce to Li2O."""
    return pd.DataFrame({
        "Reduced Composition": ["Li2O", "Li4O2", "NaCl"],
        "Ionic conductivity (S cm-1)": [1e-4, 2e-4, 3e-6],
        "DOI": ["10.1234/a", "10.1234/a", "10.1234/b"],
    })


@pytest.fixture
def duplicate_composition_dataset(duplicate_composition_df):
    return Dataset(duplicate_composition_df)


@pytest.fixture
def space_group_df():
    """DataFrame that includes both DOI and Space group # columns."""
    return pd.DataFrame({
        "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
        "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
        "DOI": ["10.1234/a", "10.1234/b"],
        "Space group #": [230, 225],
    })


@pytest.fixture
def space_group_dataset(space_group_df):
    return Dataset(space_group_df)


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_returns_correct_count(self, sample_dataset):
        assert len(sample_dataset) == 4

    def test_len_empty_dataset(self, empty_dataset):
        assert len(empty_dataset) == 0

    def test_len_matches_dataframe(self, sample_dataset, sample_df):
        assert len(sample_dataset) == len(sample_df)


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_getitem_by_int_returns_dict(self, sample_dataset):
        entry = sample_dataset[0]
        assert isinstance(entry, dict)

    def test_getitem_by_int_has_id_key(self, sample_dataset):
        entry = sample_dataset[0]
        assert "ID" in entry

    def test_getitem_by_int_has_correct_composition(self, sample_dataset):
        entry = sample_dataset[0]
        assert entry["Reduced Composition"] == "Li7La3Zr2O12"

    def test_getitem_last_element(self, sample_dataset):
        entry = sample_dataset[3]
        assert entry["Reduced Composition"] == "KBr"

    def test_getitem_by_label(self):
        """Indexing by a label that exists in the DataFrame index."""
        df = pd.DataFrame(
            {
                "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
                "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            },
            index=["abc", "def"],
        )
        ds = Dataset(df)
        entry = ds["abc"]
        assert isinstance(entry, dict)
        assert entry["ID"] == "abc"
        assert entry["Reduced Composition"] == "Li7La3Zr2O12"

    def test_getitem_int_out_of_range_raises(self, sample_dataset):
        with pytest.raises((IndexError, KeyError)):
            sample_dataset[100]

    def test_getitem_missing_label_raises(self, sample_dataset):
        with pytest.raises(KeyError):
            sample_dataset["nonexistent_label"]


# ---------------------------------------------------------------------------
# __iter__
# ---------------------------------------------------------------------------


class TestIter:
    def test_iter_yields_all_entries(self, sample_dataset):
        entries = list(sample_dataset)
        assert len(entries) == 4

    def test_iter_yields_dicts(self, sample_dataset):
        for entry in sample_dataset:
            assert isinstance(entry, dict)

    def test_iter_entries_have_expected_keys(self, sample_dataset):
        for entry in sample_dataset:
            assert "Reduced Composition" in entry
            assert "Ionic conductivity (S cm-1)" in entry

    def test_iter_empty_dataset(self, empty_dataset):
        entries = list(empty_dataset)
        assert entries == []

    def test_iter_preserves_order(self, sample_dataset):
        entries = list(sample_dataset)
        expected_comps = ["Li7La3Zr2O12", "NaCl", "Li3PS4", "KBr"]
        actual_comps = [e["Reduced Composition"] for e in entries]
        assert actual_comps == expected_comps


# ---------------------------------------------------------------------------
# __add__  (NEW API -- will fail until implemented)
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_same_columns_combined_length(
        self, sample_dataset, other_dataset
    ):
        """Concatenating two datasets with the same columns gives the sum of
        their lengths."""
        combined = sample_dataset + other_dataset
        assert len(combined) == len(sample_dataset) + len(other_dataset)

    def test_add_returns_dataset_instance(
        self, sample_dataset, other_dataset
    ):
        combined = sample_dataset + other_dataset
        assert isinstance(combined, Dataset)

    def test_add_different_columns_keeps_common(
        self, sample_dataset, extra_cols_dataset
    ):
        """When columns differ, only the common columns are kept."""
        combined = sample_dataset + extra_cols_dataset
        assert "Reduced Composition" in combined.labels
        assert "Ionic conductivity (S cm-1)" in combined.labels
        assert "DOI" in combined.labels
        # "Space group #" is only in extra_cols_dataset, not in sample
        assert "Space group #" not in combined.labels

    def test_add_originals_unchanged(self, sample_dataset, other_dataset):
        """Adding datasets does not mutate either operand."""
        original_len_self = len(sample_dataset)
        original_len_other = len(other_dataset)
        original_cols_self = list(sample_dataset.labels)
        _ = sample_dataset + other_dataset
        assert len(sample_dataset) == original_len_self
        assert len(other_dataset) == original_len_other
        assert list(sample_dataset.labels) == original_cols_self

    def test_add_empty_to_nonempty(self, sample_dataset, empty_dataset):
        """Adding an empty dataset returns a copy with the same rows."""
        combined = sample_dataset + empty_dataset
        assert len(combined) == len(sample_dataset)

    def test_add_nonempty_to_empty(self, sample_dataset, empty_dataset):
        combined = empty_dataset + sample_dataset
        assert len(combined) == len(sample_dataset)

    def test_add_two_empty(self, empty_dataset):
        combined = empty_dataset + empty_dataset
        assert len(combined) == 0

    def test_add_no_dedup(self, sample_dataset):
        """__add__ does NOT deduplicate -- duplicates should be preserved."""
        combined = sample_dataset + sample_dataset
        assert len(combined) == 2 * len(sample_dataset)

    def test_add_data_values_preserved(self, sample_dataset, other_dataset):
        """All data values from both datasets appear in the result."""
        combined = sample_dataset + other_dataset
        comps = list(combined.dataframe["Reduced Composition"])
        assert "Li7La3Zr2O12" in comps
        assert "NaCl" in comps
        assert "MgO" in comps


# ---------------------------------------------------------------------------
# union  (NEW API -- will fail until implemented)
# ---------------------------------------------------------------------------


class TestUnion:
    def test_union_deduplicates_identical_composition(self):
        """Two datasets with the same composition string are deduplicated."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "KBr"],
            "Ionic conductivity (S cm-1)": [2e-4, 3e-5],
        })
        ds1 = Dataset(df1)
        ds2 = Dataset(df2)
        result = ds1.union(ds2)
        assert len(result) == 3  # Li2O, NaCl, KBr

    def test_union_reduces_equivalent_formulas(self):
        """'Li2O' and 'Li4O2' reduce to the same canonical formula and should
        be treated as duplicates."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["Li4O2", "KBr"],
            "Ionic conductivity (S cm-1)": [2e-4, 3e-5],
        })
        ds1 = Dataset(df1)
        ds2 = Dataset(df2)
        result = ds1.union(ds2)
        assert len(result) == 3  # Li2O (from ds1), NaCl, KBr

    def test_union_keeps_first_occurrence(self):
        """When a composition appears in both datasets, the row from the first
        dataset (self) is kept."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O"],
            "Ionic conductivity (S cm-1)": [1e-4],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["Li2O"],
            "Ionic conductivity (S cm-1)": [9e-9],
        })
        ds1 = Dataset(df1)
        ds2 = Dataset(df2)
        result = ds1.union(ds2)
        assert len(result) == 1
        # The conductivity value from ds1 should be kept
        assert result.dataframe["Ionic conductivity (S cm-1)"].iloc[0] == pytest.approx(1e-4)

    def test_union_non_duplicates_all_preserved(self):
        """Rows with unique compositions are all preserved."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["KBr", "MgO"],
            "Ionic conductivity (S cm-1)": [3e-5, 5e-7],
        })
        ds1 = Dataset(df1)
        ds2 = Dataset(df2)
        result = ds1.union(ds2)
        assert len(result) == 4

    def test_union_returns_dataset(self):
        df1 = pd.DataFrame({
            "Reduced Composition": ["NaCl"],
            "Ionic conductivity (S cm-1)": [1e-6],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["KBr"],
            "Ionic conductivity (S cm-1)": [3e-5],
        })
        result = Dataset(df1).union(Dataset(df2))
        assert isinstance(result, Dataset)

    def test_union_originals_unchanged(self):
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O"],
            "Ionic conductivity (S cm-1)": [1e-4],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["Li2O"],
            "Ionic conductivity (S cm-1)": [2e-4],
        })
        ds1 = Dataset(df1)
        ds2 = Dataset(df2)
        _ = ds1.union(ds2)
        assert len(ds1) == 1
        assert len(ds2) == 1


# ---------------------------------------------------------------------------
# merge_datasets  (static method)
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def test_merge_two_datasets(self, sample_dataset, other_dataset):
        """Merging two datasets produces a result with rows from both."""
        result = Dataset.merge_datasets(sample_dataset, other_dataset,
                                        remove_duplicates=False)
        # merge_datasets returns a Dataset (or DataFrame -- test both)
        total = len(result)
        assert total == len(sample_dataset) + len(other_dataset)

    def test_merge_keeps_shared_relevant_columns(self):
        """Only columns that appear in ALL datasets and are in the relevant
        set ('Reduced Composition', 'Ionic conductivity (S cm-1)',
        'Space group #', 'DOI') are kept."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["NaCl"],
            "Ionic conductivity (S cm-1)": [1e-6],
            "DOI": ["10.1234/a"],
            "Space group #": [225],
            "Extra Column": ["x"],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["KBr"],
            "Ionic conductivity (S cm-1)": [3e-5],
            "DOI": ["10.1234/b"],
            "Space group #": [221],
            "Another Extra": [42],
        })
        result = Dataset.merge_datasets(Dataset(df1), Dataset(df2),
                                        remove_duplicates=False)
        if isinstance(result, Dataset):
            cols = list(result.dataframe.columns)
        else:
            cols = list(result.columns)
        assert "Reduced Composition" in cols
        assert "Ionic conductivity (S cm-1)" in cols
        # These should be kept because they are in the relevant set AND
        # present in all datasets
        assert "DOI" in cols
        assert "Space group #" in cols
        # Extra columns not in the relevant set should be dropped
        assert "Extra Column" not in cols
        assert "Another Extra" not in cols

    def test_merge_remove_duplicates_true(self):
        """With remove_duplicates=True, rows with the same reduced composition
        are deduplicated."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "KBr"],
            "Ionic conductivity (S cm-1)": [2e-4, 3e-5],
        })
        result = Dataset.merge_datasets(Dataset(df1), Dataset(df2),
                                        remove_duplicates=True)
        result_len = len(result) if isinstance(result, Dataset) else len(result)
        # Li2O appears in both but should appear only once after dedup
        assert result_len == 3

    def test_merge_remove_duplicates_false(self):
        """With remove_duplicates=False, all rows are kept even if
        compositions overlap."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["Li2O", "KBr"],
            "Ionic conductivity (S cm-1)": [2e-4, 3e-5],
        })
        result = Dataset.merge_datasets(Dataset(df1), Dataset(df2),
                                        remove_duplicates=False)
        result_len = len(result) if isinstance(result, Dataset) else len(result)
        assert result_len == 4

    def test_merge_three_datasets(self):
        df1 = pd.DataFrame({
            "Reduced Composition": ["Li2O"],
            "Ionic conductivity (S cm-1)": [1e-4],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["NaCl"],
            "Ionic conductivity (S cm-1)": [1e-6],
        })
        df3 = pd.DataFrame({
            "Reduced Composition": ["KBr"],
            "Ionic conductivity (S cm-1)": [3e-5],
        })
        result = Dataset.merge_datasets(Dataset(df1), Dataset(df2),
                                        Dataset(df3),
                                        remove_duplicates=False)
        result_len = len(result) if isinstance(result, Dataset) else len(result)
        assert result_len == 3

    def test_merge_four_datasets(self):
        dfs = [
            pd.DataFrame({
                "Reduced Composition": [f"A{i}B"],
                "Ionic conductivity (S cm-1)": [float(i)],
            })
            for i in range(4)
        ]
        datasets = [Dataset(df) for df in dfs]
        result = Dataset.merge_datasets(*datasets, remove_duplicates=False)
        result_len = len(result) if isinstance(result, Dataset) else len(result)
        assert result_len == 4

    def test_merge_optional_doi_column_missing_in_one(self):
        """If 'DOI' is missing from one dataset, it should not appear in the
        merged result (only columns present in ALL datasets are kept)."""
        df1 = pd.DataFrame({
            "Reduced Composition": ["NaCl"],
            "Ionic conductivity (S cm-1)": [1e-6],
            "DOI": ["10.1234/a"],
        })
        df2 = pd.DataFrame({
            "Reduced Composition": ["KBr"],
            "Ionic conductivity (S cm-1)": [3e-5],
            # No DOI column at all
        })
        result = Dataset.merge_datasets(Dataset(df1), Dataset(df2),
                                        remove_duplicates=False)
        if isinstance(result, Dataset):
            cols = list(result.dataframe.columns)
        else:
            cols = list(result.columns)
        assert "DOI" not in cols


# ---------------------------------------------------------------------------
# remove_matching_entries
# ---------------------------------------------------------------------------


class TestRemoveMatchingEntries:
    def test_matching_comp_and_matching_doi_removed(self):
        """Both datasets have 'Li7La3Zr2O12' with DOI '10.1234/a' -- the
        entry should be removed from self."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 1
        assert result.dataframe["Reduced Composition"].iloc[0] == "NaCl"

    def test_matching_comp_self_doi_nan_removed(self):
        """Self has 'Li7La3Zr2O12' with NaN DOI, other has it with a real
        DOI.  The entry should be removed from self because we cannot verify
        it is a distinct measurement."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": [None, "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 1
        assert result.dataframe["Reduced Composition"].iloc[0] == "NaCl"

    def test_matching_comp_different_dois_kept(self):
        """Same composition but different (both non-null) DOIs -- the entry
        should NOT be removed because they are distinct measurements."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/z"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 2

    def test_no_match_kept(self):
        """Composition not in the other dataset at all -- entry is kept."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["MgO"],
            "Ionic conductivity (S cm-1)": [5e-7],
            "DOI": ["10.1234/d"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 2

    def test_returns_new_dataset(self):
        """remove_matching_entries returns a new Dataset; the original's
        dataframe is unchanged."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        original_len = len(ds)
        result = ds.remove_matching_entries(other)
        # Result is a Dataset
        assert isinstance(result, Dataset)
        # Original is unchanged
        assert len(ds) == original_len

    def test_empty_other_nothing_removed(self):
        """If the other dataset is empty, nothing is removed."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": pd.Series([], dtype="object"),
            "Ionic conductivity (S cm-1)": pd.Series([], dtype="float64"),
            "DOI": pd.Series([], dtype="object"),
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 2

    def test_empty_self_returns_empty(self):
        """If self is empty, result is also empty."""
        self_df = pd.DataFrame({
            "Reduced Composition": pd.Series([], dtype="object"),
            "Ionic conductivity (S cm-1)": pd.Series([], dtype="float64"),
            "DOI": pd.Series([], dtype="object"),
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 0
        assert isinstance(result, Dataset)

    def test_formula_equivalence_reduced(self):
        """'Li2O' in self and 'Li4O2' in other should be treated as the same
        composition (pymatgen reduces both to 'Li2O').  When DOIs also match,
        the entry should be removed from self."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li2O", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li4O2"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        assert len(result) == 1
        assert result.dataframe["Reduced Composition"].iloc[0] == "NaCl"

    def test_multiple_matches_all_removed(self):
        """When self has multiple rows matching the same composition in other,
        rows with matching DOI or NaN DOI should all be removed."""
        self_df = pd.DataFrame({
            "Reduced Composition": [
                "Li7La3Zr2O12",
                "Li7La3Zr2O12",
                "Li7La3Zr2O12",
                "NaCl",
            ],
            "Ionic conductivity (S cm-1)": [1e-4, 2e-4, 3e-4, 1e-6],
            "DOI": ["10.1234/a", None, "10.1234/z", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [5e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        remaining_comps = list(result.dataframe["Reduced Composition"])
        # Row 0 (DOI match) removed, Row 1 (NaN DOI) removed,
        # Row 2 (different DOI "10.1234/z") kept, Row 3 (NaCl) kept
        assert "NaCl" in remaining_comps
        assert len(result) == 2

    def test_accepts_dataframe_as_other(self):
        """remove_matching_entries should also accept a raw pandas DataFrame
        as the 'other' argument (not just a Dataset)."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": ["10.1234/a", "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": ["10.1234/a"],
        })
        ds = Dataset(self_df)
        result = ds.remove_matching_entries(other_df)
        assert len(result) == 1

    def test_both_dois_nan_not_removed(self):
        """If both self and other have NaN DOIs for the same composition,
        the match is ambiguous but per current logic, self's NaN DOI row
        is removed when other has a matching composition row that triggers
        the removal chain.  When OTHER's DOI is also NaN, there is no
        confirmed DOI match so the row should be kept."""
        self_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12", "NaCl"],
            "Ionic conductivity (S cm-1)": [1e-4, 1e-6],
            "DOI": [None, "10.1234/b"],
        })
        other_df = pd.DataFrame({
            "Reduced Composition": ["Li7La3Zr2O12"],
            "Ionic conductivity (S cm-1)": [2e-4],
            "DOI": [None],
        })
        ds = Dataset(self_df)
        other = Dataset(other_df)
        result = ds.remove_matching_entries(other)
        # Both DOIs are NaN -- no confirmed match, row should be kept
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Attribute correctness
# ---------------------------------------------------------------------------


class TestAttributes:
    def test_entries_is_list_of_index(self, sample_dataset, sample_df):
        assert sample_dataset.entries == list(sample_df.index)

    def test_labels_is_list_of_columns(self, sample_dataset, sample_df):
        assert sample_dataset.labels == list(sample_df.columns)

    def test_dataframe_stored(self, sample_dataset, sample_df):
        pd.testing.assert_frame_equal(sample_dataset.dataframe, sample_df)
