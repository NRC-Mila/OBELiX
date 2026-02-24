"""
Performance tests for the OBELiX deduplication and merging operations.

These tests download the real datasets once per session (via session-scoped
fixtures) and verify that key operations complete within generous wall-clock
time limits.  They exist to catch algorithmic regressions -- for example,
an O(n^3) dedup loop that used to complete in seconds ballooning to minutes
after a dataset grows.

All tests in this module are marked ``@pytest.mark.slow`` so they can be
excluded from fast CI runs with::

    pytest -m "not slow"

Note: Some tests exercise methods that have not yet been implemented
(``Laskowski.remove_obelix`` and ``Dataset.union``).  These tests will fail
until the corresponding methods are added.  They are included now so that
the expected performance contract is documented from the start.
"""

import time

import pytest

from obelix import OBELiX, LiIon, Laskowski, ShonAndMin
from obelix.dataset import Dataset


# ---------------------------------------------------------------------------
# Session-scoped fixtures -- download data once, reuse across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def obelix_data():
    """Load the OBELiX dataset without CIF files (faster download)."""
    return OBELiX(no_cifs=True)


@pytest.fixture(scope="session")
def liion_data():
    """Load the LiIon dataset with default settings (room-temp only)."""
    return LiIon()


@pytest.fixture(scope="session")
def laskowski_data():
    """Load the Laskowski dataset with default settings."""
    return Laskowski()


@pytest.fixture(scope="session")
def shonandmin_data():
    """Load the ShonAndMin dataset with default settings."""
    return ShonAndMin()


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_remove_matching_entries_performance(obelix_data):
    """Verify that remove_matching_entries completes in under 30 seconds.

    The original O(n^3) implementation could take several minutes on the
    full LiIon-vs-OBELiX comparison.  This test ensures we stay within a
    reasonable envelope after optimisation.

    A fresh ``LiIon`` instance is created because ``remove_matching_entries``
    returns a new Dataset but we want to be explicit about not reusing
    shared session state.
    """
    liion = LiIon()
    timeout = 30  # seconds

    start = time.time()
    result = liion.remove_matching_entries(obelix_data)
    elapsed = time.time() - start

    assert elapsed < timeout, (
        f"remove_matching_entries(obelix) took {elapsed:.1f}s, "
        f"exceeding the {timeout}s limit"
    )
    # Sanity check: the result should be a Dataset with fewer rows than the input.
    assert len(result) < len(liion), (
        "Expected some entries to be removed, but output length equals input length"
    )


@pytest.mark.slow
def test_remove_obelix_liion_performance(obelix_data):
    """Verify that LiIon.remove_obelix completes in under 5 seconds.

    This operation is a simple index-based drop, so it should be nearly
    instantaneous on any reasonable hardware.
    """
    liion = LiIon()
    timeout = 5  # seconds

    start = time.time()
    result = liion.remove_obelix(obelix_data)
    elapsed = time.time() - start

    assert elapsed < timeout, (
        f"LiIon.remove_obelix(obelix) took {elapsed:.1f}s, "
        f"exceeding the {timeout}s limit"
    )
    assert len(result) < len(liion.dataframe), (
        "Expected some entries to be removed via index drop"
    )


@pytest.mark.slow
def test_remove_obelix_laskowski_performance(obelix_data):
    """Verify that Laskowski.remove_obelix completes in under 5 seconds.

    NOTE: ``Laskowski.remove_obelix`` has not been implemented yet.  This
    test documents the expected performance contract and will begin passing
    once the method is added.
    """
    laskowski = Laskowski()
    timeout = 5  # seconds

    start = time.time()
    result = laskowski.remove_obelix(obelix_data)
    elapsed = time.time() - start

    assert elapsed < timeout, (
        f"Laskowski.remove_obelix(obelix) took {elapsed:.1f}s, "
        f"exceeding the {timeout}s limit"
    )


@pytest.mark.slow
def test_merge_datasets_performance(obelix_data, liion_data, laskowski_data, shonandmin_data):
    """Verify that merging all four datasets completes in under 30 seconds.

    Uses the session-scoped fixtures directly since ``merge_datasets`` does
    not mutate any of its inputs.
    """
    timeout = 30  # seconds

    start = time.time()
    result = Dataset.merge_datasets(
        obelix_data, liion_data, laskowski_data, shonandmin_data
    )
    elapsed = time.time() - start

    assert elapsed < timeout, (
        f"merge_datasets(obelix, liion, laskowski, shonandmin) took {elapsed:.1f}s, "
        f"exceeding the {timeout}s limit"
    )
    assert len(result) > 0, "Merged dataset should not be empty"


@pytest.mark.slow
def test_union_performance():
    """Verify that Dataset.union completes in under 15 seconds.

    NOTE: ``Dataset.union`` has not been implemented yet.  This test
    documents the expected performance contract and will begin passing
    once the method is added.
    """
    liion = LiIon()
    laskowski = Laskowski()
    timeout = 15  # seconds

    start = time.time()
    result = liion.union(laskowski)
    elapsed = time.time() - start

    assert elapsed < timeout, (
        f"LiIon.union(laskowski) took {elapsed:.1f}s, "
        f"exceeding the {timeout}s limit"
    )
    # The union should contain at least as many rows as the larger input.
    assert len(result) >= max(len(liion), len(laskowski)), (
        "Union result should have at least as many entries as the larger input dataset"
    )
