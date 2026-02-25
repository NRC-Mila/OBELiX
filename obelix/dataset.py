import pandas as pd
from pymatgen.core import Composition

from .utils import round_partial_occ

# Columns that merge_datasets will keep when present in ALL input datasets.
_RELEVANT_COLUMNS = [
    "Reduced Composition",
    "Ionic conductivity (S cm-1)",
    "Space group #",
    "DOI",
]


class Dataset:
    """
    Dataset class. This is a wrapper around a pandas DataFrame (which cannot be inhertided).

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
        entries (list): List of entry IDs (3 lower-case alphanumeric symbols).
        labels (list): List of labels (columns).

    Methods:
        to_numpy(): Returns the dataset as a numpy array.
        to_dict(): Returns the dataset as a dictionary.
        with_cifs(): Returns a new Dataset object with only the entries that have a CIF.
        round_partial(): Returns a new Datset where the partial occupancies of the sites in the structures are rounded to the nearest integer.

    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.entries = list(self.dataframe.index)
        self.labels = list(self.dataframe.keys())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if type(idx) == int:
            entry = self.dataframe.iloc[idx]
        else:
            entry = self.dataframe.loc[idx]

        if type(entry) == pd.Series:
            entry_dict = entry.to_dict()
            entry_dict["ID"] = entry.name
        else:
            entry_dict = entry.to_dict()

        return entry_dict

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_numpy(self):
        return self.dataframe.to_numpy()

    def to_dict(self):
        return self.dataframe.to_dict()

    def with_cifs(self):
        return Dataset(self.dataframe.dropna(subset=["structure"]))

    def round_partial(self):
        structures = []
        for i, row in self.dataframe.iterrows():
            if row["structure"] is not None:
                structures.append(round_partial_occ(row["structure"]))
            else:
                structures.append(None)
        return Dataset(self.dataframe.assign(structure=structures))

    def __add__(self, other):
        """Concatenate two datasets, keeping only common columns.

        No deduplication is performed — use :meth:`union` for that.

        Returns:
            A new :class:`Dataset` with rows from both operands.
        """
        common_cols = [
            c for c in self.dataframe.columns if c in other.dataframe.columns
        ]
        combined = pd.concat(
            [self.dataframe[common_cols], other.dataframe[common_cols]],
            ignore_index=True,
        )
        return Dataset(combined)

    def union(self, other):
        """Concatenate two datasets and deduplicate by reduced composition.

        Like :meth:`__add__` but removes rows whose canonical reduced
        formula (via pymatgen) duplicates an earlier row.  The first
        occurrence is kept.

        Returns:
            A new :class:`Dataset` with combined, deduplicated rows.
        """
        combined = self + other

        def _canonical(f):
            if pd.isna(f):
                return f
            try:
                return Composition(f).reduced_formula
            except Exception:
                return f

        canonical = combined.dataframe["Reduced Composition"].apply(_canonical)
        deduped = combined.dataframe[~canonical.duplicated(keep="first")]
        return Dataset(deduped)

    @staticmethod
    def merge_datasets(*datasets, remove_duplicates=True):
        """Merge multiple Dataset objects into a single Dataset.

        Keeps only columns from _RELEVANT_COLUMNS that are present in ALL
        input datasets. If remove_duplicates is True, deduplicates by
        canonical reduced composition (using pymatgen).

        Args:
            *datasets: Dataset objects to merge.
            remove_duplicates: If True, drop rows with duplicate reduced
                compositions (keeping the first occurrence). Default: True.

        Returns:
            A new Dataset containing the merged rows.
        """
        # Find columns present in ALL datasets, restricted to the relevant set
        common_cols = None
        for ds in datasets:
            ds_cols = set(ds.dataframe.columns)
            relevant = {c for c in _RELEVANT_COLUMNS if c in ds_cols}
            if common_cols is None:
                common_cols = relevant
            else:
                common_cols = common_cols & relevant

        if common_cols is None:
            common_cols = set()

        # Always keep at least composition and conductivity
        col_order = [c for c in _RELEVANT_COLUMNS if c in common_cols]

        dfs = []
        for ds in datasets:
            dfs.append(ds.dataframe[col_order])

        combined = pd.concat(dfs, ignore_index=True)

        if remove_duplicates:
            # Use canonical reduced_formula for dedup
            def _canonical(f):
                if pd.isna(f):
                    return f
                try:
                    return Composition(f).reduced_formula
                except Exception:
                    return f

            canonical = combined["Reduced Composition"].apply(_canonical)
            combined = combined[~canonical.duplicated(keep="first")]

        return Dataset(combined)

    def remove_matching_entries(self, other):
        """Remove entries that are duplicated in another dataset.

        Compares by reduced composition (using pymatgen for chemical
        equivalence) and DOI.  A row in ``self`` is removed when:

        * Its reduced composition matches a composition in *other*, **and**
        * Its DOI matches a DOI in *other* for that composition (both
          non-NaN), **or** its DOI is NaN while *other* has at least one
          non-NaN DOI for that composition.

        Rows where the composition matches but DOIs differ (both non-NaN)
        are kept — they represent distinct measurements from different
        sources.

        Parameters:
            other: A :class:`Dataset` or :class:`~pandas.DataFrame` with
                ``'Reduced Composition'`` and ``'DOI'`` columns.

        Returns:
            A new :class:`Dataset` with matching entries removed.  The
            original dataset is **not** mutated.
        """
        if isinstance(other, pd.DataFrame):
            other_df = other
        else:
            other_df = other.dataframe

        # Build lookup: canonical reduced_formula → set of non-NaN DOIs
        other_lookup: dict[str, set[str]] = {}
        for _, row in other_df.iterrows():
            comp = row.get("Reduced Composition")
            doi = row.get("DOI")
            try:
                key = Composition(comp).reduced_formula
            except Exception:
                continue
            if key not in other_lookup:
                other_lookup[key] = set()
            if pd.notna(doi):
                other_lookup[key].add(doi)

        # Determine which rows in self to remove
        indices_to_remove = set()
        for i, self_row in self.dataframe.iterrows():
            self_comp = self_row.get("Reduced Composition")
            self_doi = self_row.get("DOI")
            try:
                self_key = Composition(self_comp).reduced_formula
            except Exception:
                continue

            if self_key not in other_lookup:
                continue

            other_dois = other_lookup[self_key]
            if pd.notna(self_doi) and self_doi in other_dois:
                # Exact DOI match — confirmed duplicate
                indices_to_remove.add(i)
            elif pd.isna(self_doi) and len(other_dois) > 0:
                # Self has no DOI but other has a confirmed source —
                # cannot verify this is a distinct measurement
                indices_to_remove.add(i)

        new_dataframe = self.dataframe.drop(index=indices_to_remove)
        return Dataset(new_dataframe)
