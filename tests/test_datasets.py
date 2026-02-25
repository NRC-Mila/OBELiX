"""
Integration tests for OBELiX dataset subclasses and their interactions.

These tests download real data from the internet, so they require network
access and are best understood as integration tests. Session-scoped fixtures
ensure that each dataset is downloaded only once per test session, keeping
total runtime manageable.

Tested classes:
    - OBELiX  (obelix/__init__.py)
    - LiIon   (obelix/liion.py)
    - Laskowski (obelix/laskowski.py)
    - ShonAndMin (obelix/shonandmin.py)
    - McHaffie (obelix/mchaffie.py)
    - Dataset  (obelix/dataset.py)
"""

import numpy as np
import pandas as pd
import pytest

from obelix import OBELiX, LiIon, Laskowski, ShonAndMin, McHaffie, Dataset


# ---------------------------------------------------------------------------
# Session-scoped fixtures -- download each dataset exactly once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def obelix_data():
    """Load the OBELiX dataset without CIF files for speed."""
    return OBELiX(no_cifs=True)


@pytest.fixture(scope="session")
def liion_data(tmp_path_factory):
    """Load the LiIon dataset with default settings (room_temp_only=True).

    Uses local=True to download from the OBELiX GitHub mirror, avoiding
    dependence on the external Liverpool server.
    """
    path = str(tmp_path_factory.mktemp("liion"))
    return LiIon(data_path=path, local=True)


@pytest.fixture(scope="session")
def liion_raw_data(tmp_path_factory):
    """Load the LiIon dataset with all temperatures included."""
    path = str(tmp_path_factory.mktemp("liion_raw"))
    return LiIon(data_path=path, room_temp_only=False, local=True)


@pytest.fixture(scope="session")
def laskowski_data(tmp_path_factory):
    """Load the Laskowski dataset with default settings.

    Uses local=True to read from the repo's bundled data directory,
    avoiding dependence on the remote GitHub raw URL.
    """
    path = str(tmp_path_factory.mktemp("laskowski"))
    return Laskowski(data_path=path, local=True)


@pytest.fixture(scope="session")
def shonandmin_data(tmp_path_factory):
    """Load the ShonAndMin dataset with default settings
    (clean_data=True, keep_min_conductivity=True).

    Uses local=True to read from the repo's bundled xlsx file,
    avoiding the ACS paywall (403 Forbidden).
    """
    path = str(tmp_path_factory.mktemp("shonandmin"))
    return ShonAndMin(data_path=path, local=True)


@pytest.fixture(scope="session")
def shonandmin_raw_data(tmp_path_factory):
    """Load the ShonAndMin dataset without cleaning or deduplication."""
    path = str(tmp_path_factory.mktemp("shonandmin_raw"))
    return ShonAndMin(data_path=path, clean_data=False,
                      keep_min_conductivity=False, local=True)


@pytest.fixture(scope="session")
def mchaffie_data(tmp_path_factory):
    """Load the McHaffie dataset with default settings.

    Uses local=True to read from the repo's bundled data directory,
    avoiding dependence on the remote Caltech Data repository.
    """
    path = str(tmp_path_factory.mktemp("mchaffie"))
    return McHaffie(data_path=path, local=True)


# ===================================================================
# LiIon tests
# ===================================================================

class TestLiIon:
    """Tests for the LiIon dataset class."""

    def test_liion_loads(self, liion_data):
        """LiIon dataset loads successfully and contains entries."""
        assert len(liion_data) > 0
        assert len(liion_data.dataframe) > 0

    def test_liion_columns(self, liion_data):
        """After rename, LiIon has the standard composition and conductivity columns."""
        cols = liion_data.dataframe.columns
        assert "Reduced Composition" in cols, (
            "Missing 'Reduced Composition' column after rename"
        )
        assert "Ionic conductivity (S cm-1)" in cols, (
            "Missing 'Ionic conductivity (S cm-1)' column after rename"
        )

    def test_liion_raw_vs_filtered(self, liion_data, liion_raw_data):
        """The unfiltered dataset (all temperatures) has strictly more rows
        than the room-temperature-only dataset."""
        assert len(liion_raw_data) > len(liion_data), (
            f"Expected raw ({len(liion_raw_data)}) > filtered ({len(liion_data)})"
        )

    def test_liion_room_temp_filter(self, liion_data):
        """Every row in the room-temperature-filtered dataset has a temperature
        within the expected range of 25 +/- 7 degrees Celsius (i.e. [18, 32])."""
        df = liion_data.dataframe
        if "temperature" in df.columns:
            temps = df["temperature"]
            assert temps.min() >= 18, (
                f"Found temperature below 18: {temps.min()}"
            )
            assert temps.max() <= 32, (
                f"Found temperature above 32: {temps.max()}"
            )

    def test_liion_remove_obelix_returns_dataset(self, liion_data, obelix_data):
        """LiIon.remove_obelix returns a Dataset instance, not a raw DataFrame."""
        result = liion_data.remove_obelix(obelix_data)
        assert isinstance(result, Dataset), (
            f"Expected Dataset, got {type(result).__name__}"
        )

    def test_liion_remove_obelix_reduces_count(self, liion_data, obelix_data):
        """Removing OBELiX entries from LiIon produces fewer rows than the
        original LiIon dataset."""
        result = liion_data.remove_obelix(obelix_data)
        assert len(result) < len(liion_data), (
            f"Expected fewer rows after removal: got {len(result)} vs original {len(liion_data)}"
        )

    def test_liion_remove_obelix_correctness(self, liion_data, obelix_data):
        """After removing OBELiX entries, none of the Liion IDs present in the
        OBELiX dataset should remain in the result.

        This works because OBELiX carries a 'Liion ID' column that maps its
        entries back to original LiIon indices.  After removal, the result's
        index should have no intersection with those IDs.
        """
        ob_liion_ids = (
            obelix_data.dataframe["Liion ID"]
            .dropna()
            .astype(int)
            .tolist()
        )
        result = liion_data.remove_obelix(obelix_data)
        remaining_ids = set(result.dataframe.index)
        overlap = remaining_ids.intersection(ob_liion_ids)
        assert len(overlap) == 0, (
            f"Found {len(overlap)} Liion IDs from OBELiX still in result: "
            f"{sorted(list(overlap))[:10]}..."
        )


# ===================================================================
# Laskowski tests
# ===================================================================

class TestLaskowski:
    """Tests for the Laskowski dataset class."""

    def test_laskowski_loads(self, laskowski_data):
        """Laskowski dataset loads successfully and contains entries."""
        assert len(laskowski_data) > 0
        assert len(laskowski_data.dataframe) > 0

    def test_laskowski_columns(self, laskowski_data):
        """After rename, Laskowski has the expected standard columns."""
        cols = laskowski_data.dataframe.columns
        assert "Reduced Composition" in cols, (
            "Missing 'Reduced Composition' column after rename"
        )
        assert "Ionic conductivity (S cm-1)" in cols, (
            "Missing 'Ionic conductivity (S cm-1)' column after rename"
        )
        assert "Space group" in cols, (
            "Missing 'Space group' column after rename"
        )

    def test_laskowski_remove_obelix_returns_dataset(self, laskowski_data, obelix_data):
        """Laskowski.remove_obelix returns a Dataset instance."""
        result = laskowski_data.remove_obelix(obelix_data)
        assert isinstance(result, Dataset), (
            f"Expected Dataset, got {type(result).__name__}"
        )

    def test_laskowski_remove_obelix_reduces_count(self, laskowski_data, obelix_data):
        """Removing OBELiX entries from Laskowski produces fewer rows."""
        result = laskowski_data.remove_obelix(obelix_data)
        assert len(result) < len(laskowski_data), (
            f"Expected fewer rows after removal: got {len(result)} vs original {len(laskowski_data)}"
        )

    def test_laskowski_remove_obelix_correctness(self, laskowski_data, obelix_data):
        """After removing OBELiX entries, none of the Laskowski IDs present in
        the OBELiX dataset should remain in the result.

        OBELiX carries a 'Laskowski ID' column that maps its entries back to
        original Laskowski indices.  After removal, the result's index should
        have no intersection with those IDs.
        """
        ob_lask_ids = (
            obelix_data.dataframe["Laskowski ID"]
            .dropna()
            .astype(int)
            .tolist()
        )
        result = laskowski_data.remove_obelix(obelix_data)
        remaining_ids = set(result.dataframe.index)
        overlap = remaining_ids.intersection(ob_lask_ids)
        assert len(overlap) == 0, (
            f"Found {len(overlap)} Laskowski IDs from OBELiX still in result: "
            f"{sorted(list(overlap))[:10]}..."
        )


# ===================================================================
# ShonAndMin tests
# ===================================================================

class TestShonAndMin:
    """Tests for the ShonAndMin dataset class."""

    def test_shonandmin_loads(self, shonandmin_data):
        """ShonAndMin dataset loads successfully and contains entries."""
        assert len(shonandmin_data) > 0
        assert len(shonandmin_data.dataframe) > 0

    def test_shonandmin_columns(self, shonandmin_data):
        """After rename, ShonAndMin has the expected standard columns."""
        cols = shonandmin_data.dataframe.columns
        assert "Reduced Composition" in cols, (
            "Missing 'Reduced Composition' column after rename"
        )
        assert "Ionic conductivity (S cm-1)" in cols, (
            "Missing 'Ionic conductivity (S cm-1)' column after rename"
        )

    def test_shonandmin_raw_vs_cleaned(self, shonandmin_data, shonandmin_raw_data):
        """The raw (uncleaned, no dedup) dataset has strictly more rows than
        the cleaned and deduplicated dataset."""
        assert len(shonandmin_raw_data) > len(shonandmin_data), (
            f"Expected raw ({len(shonandmin_raw_data)}) > cleaned ({len(shonandmin_data)})"
        )

    def test_shonandmin_cleaning_validity(self, shonandmin_data):
        """After cleaning, every row must have a valid positive numeric ionic
        conductivity value."""
        df = shonandmin_data.dataframe
        cond_col = "Ionic conductivity (S cm-1)"
        conductivities = pd.to_numeric(df[cond_col], errors="coerce")

        assert conductivities.notna().all(), (
            f"Found {conductivities.isna().sum()} non-numeric conductivity values after cleaning"
        )
        assert (conductivities > 0).all(), (
            f"Found {(conductivities <= 0).sum()} non-positive conductivity values after cleaning"
        )

    def test_shonandmin_min_conductivity(self, shonandmin_data):
        """When keep_min_conductivity=True (default), there should be no
        duplicate (Reduced Composition, DOI) pairs -- each such pair retains
        only the entry with the minimum conductivity."""
        df = shonandmin_data.dataframe
        if "DOI" in df.columns:
            grouped = df.groupby(["Reduced Composition", "DOI"]).size()
            duplicates = grouped[grouped > 1]
            assert len(duplicates) == 0, (
                f"Found {len(duplicates)} duplicate (Reduced Composition, DOI) pairs: "
                f"{duplicates.head().to_dict()}"
            )

    def test_shonandmin_remove_obelix_returns_dataset(self, shonandmin_data, obelix_data):
        """ShonAndMin.remove_obelix returns a Dataset instance.

        Since OBELiX has no dedicated ID column for ShonAndMin, this method
        should internally use remove_matching_entries for composition+DOI based
        deduplication."""
        result = shonandmin_data.remove_obelix(obelix_data)
        assert isinstance(result, Dataset), (
            f"Expected Dataset, got {type(result).__name__}"
        )


# ===================================================================
# McHaffie tests
# ===================================================================

class TestMcHaffie:
    """Tests for the McHaffie dataset class."""

    def test_mchaffie_loads(self, mchaffie_data):
        """McHaffie dataset loads successfully and contains entries."""
        assert len(mchaffie_data) > 0
        assert len(mchaffie_data.dataframe) > 0

    def test_mchaffie_expected_size(self, mchaffie_data):
        """McHaffie dataset contains exactly 571 entries."""
        assert len(mchaffie_data) == 571, (
            f"Expected 571 entries, got {len(mchaffie_data)}"
        )

    def test_mchaffie_columns(self, mchaffie_data):
        """After rename, McHaffie has the expected standard columns."""
        cols = mchaffie_data.dataframe.columns
        assert "Reduced Composition" in cols, (
            "Missing 'Reduced Composition' column after rename"
        )
        assert "Ionic conductivity (S cm-1)" in cols, (
            "Missing 'Ionic conductivity (S cm-1)' column after rename"
        )
        assert "DOI" in cols, (
            "Missing 'DOI' column after rename"
        )
        assert "ICSD Collection Code" in cols, (
            "Missing 'ICSD Collection Code' column after rename"
        )

    def test_mchaffie_no_nulls(self, mchaffie_data):
        """Critical columns (conductivity, DOI, ICSD Collection Code) have no
        null values."""
        df = mchaffie_data.dataframe
        assert df["Ionic conductivity (S cm-1)"].notna().all(), (
            f"Found {df['Ionic conductivity (S cm-1)'].isna().sum()} null conductivity values"
        )
        assert df["DOI"].notna().all(), (
            f"Found {df['DOI'].isna().sum()} null DOI values"
        )
        assert df["ICSD Collection Code"].notna().all(), (
            f"Found {df['ICSD Collection Code'].isna().sum()} null ICSD Collection Code values"
        )

    def test_mchaffie_conductivity_range(self, mchaffie_data):
        """All conductivity values are positive."""
        df = mchaffie_data.dataframe
        conductivities = df["Ionic conductivity (S cm-1)"]
        assert (conductivities > 0).all(), (
            f"Found {(conductivities <= 0).sum()} non-positive conductivity values"
        )

    def test_mchaffie_remove_obelix_returns_dataset(self, mchaffie_data, obelix_data):
        """McHaffie.remove_obelix returns a Dataset instance.

        Since OBELiX has no dedicated ID column for McHaffie, this method
        should internally use remove_matching_entries for composition+DOI based
        deduplication."""
        result = mchaffie_data.remove_obelix(obelix_data)
        assert isinstance(result, Dataset), (
            f"Expected Dataset, got {type(result).__name__}"
        )

    def test_mchaffie_remove_obelix_reduces_count(self, mchaffie_data, obelix_data):
        """Removing OBELiX entries from McHaffie produces fewer rows."""
        result = mchaffie_data.remove_obelix(obelix_data)
        assert len(result) < len(mchaffie_data), (
            f"Expected fewer rows after removal: got {len(result)} vs original {len(mchaffie_data)}"
        )

    def test_mchaffie_icsd_stubs_raise(self, mchaffie_data):
        """All three ICSD stub methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            mchaffie_data.get_space_groups()
        with pytest.raises(NotImplementedError):
            mchaffie_data.get_lattice_parameters()
        with pytest.raises(NotImplementedError):
            mchaffie_data.get_cifs()


# ===================================================================
# Cross-dataset tests
# ===================================================================

class TestCrossDataset:
    """Tests for interactions between multiple dataset objects."""

    def test_dataset_addition(self, obelix_data, liion_data):
        """The + operator on two Datasets returns a new Dataset whose length
        equals the sum of the two input lengths."""
        combined = obelix_data + liion_data
        assert isinstance(combined, Dataset), (
            f"Expected Dataset from addition, got {type(combined).__name__}"
        )
        expected_len = len(obelix_data) + len(liion_data)
        assert len(combined) == expected_len, (
            f"Expected combined length {expected_len}, got {len(combined)}"
        )

    def test_dataset_addition_columns(self, obelix_data, liion_data):
        """When adding two datasets, the resulting columns should be only the
        columns that are common to both inputs."""
        combined = obelix_data + liion_data
        obelix_cols = set(obelix_data.dataframe.columns)
        liion_cols = set(liion_data.dataframe.columns)
        common_cols = obelix_cols & liion_cols
        result_cols = set(combined.dataframe.columns)
        assert result_cols == common_cols, (
            f"Expected only common columns {common_cols}, got {result_cols}"
        )

    def test_no_self_duplicates_after_remove_obelix(self, liion_data, obelix_data):
        """After removing OBELiX entries from LiIon, no remaining LiIon entry
        should share both Reduced Composition and DOI with any OBELiX entry.

        This is a stronger check than the ID-based correctness test: it
        verifies that composition+DOI deduplication is also effective."""
        result = liion_data.remove_obelix(obelix_data)
        result_df = result.dataframe
        ob_df = obelix_data.dataframe

        if "DOI" not in result_df.columns or "DOI" not in ob_df.columns:
            pytest.skip("DOI column not present in one or both datasets")

        # Build a set of (composition, doi) tuples from OBELiX for fast lookup
        ob_pairs = set()
        for _, row in ob_df.dropna(subset=["Reduced Composition", "DOI"]).iterrows():
            ob_pairs.add((row["Reduced Composition"], row["DOI"]))

        # Check that no remaining LiIon entry has an exact match
        matches = 0
        for _, row in result_df.dropna(subset=["Reduced Composition", "DOI"]).iterrows():
            if (row["Reduced Composition"], row["DOI"]) in ob_pairs:
                matches += 1

        assert matches == 0, (
            f"Found {matches} entries in LiIon (after remove_obelix) that share "
            f"both composition and DOI with an OBELiX entry"
        )

    def test_merge_all_datasets(self, obelix_data, liion_data, laskowski_data, shonandmin_data, mchaffie_data):
        """Merging all five datasets with duplicate removal should produce a
        result whose length is strictly less than the naive sum of individual
        dataset lengths, since overlapping entries exist."""
        naive_total = (
            len(obelix_data) + len(liion_data)
            + len(laskowski_data) + len(shonandmin_data)
            + len(mchaffie_data)
        )
        merged = Dataset.merge_datasets(
            obelix_data, liion_data, laskowski_data, shonandmin_data,
            mchaffie_data,
            remove_duplicates=True,
        )
        assert len(merged) > 0, "Merged dataset is empty"
        assert len(merged) < naive_total, (
            f"Expected merged length ({len(merged)}) to be less than naive sum ({naive_total})"
        )

    def test_merge_removes_duplicates(self, obelix_data, liion_data, laskowski_data, shonandmin_data, mchaffie_data):
        """Merging with remove_duplicates=True should produce strictly fewer
        rows than merging with remove_duplicates=False, confirming that
        cross-dataset duplicates exist and are removed."""
        merged_with = Dataset.merge_datasets(
            obelix_data, liion_data, laskowski_data, shonandmin_data,
            mchaffie_data,
            remove_duplicates=True,
        )
        merged_without = Dataset.merge_datasets(
            obelix_data, liion_data, laskowski_data, shonandmin_data,
            mchaffie_data,
            remove_duplicates=False,
        )
        assert len(merged_with) < len(merged_without), (
            f"Expected deduplicated ({len(merged_with)}) < raw merge ({len(merged_without)})"
        )
