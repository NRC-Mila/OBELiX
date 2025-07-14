import pandas as pd
import urllib.request
import tarfile
from pathlib import Path
from pymatgen.core import Structure
import warnings
from tqdm import tqdm
import importlib 
import re
import numpy as np

from obelix.utils import round_partial_occ, replace_text_IC, is_same_formula

class Dataset():
    '''
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

    '''
    
    def __init__(self, dataframe, *datasets):
        self.dataframe = dataframe
        self.entries = list(self.dataframe.index)
        self.labels = list(self.dataframe.keys())
        self.datasets = datasets
        
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

    def merge_datasets(*datasets, remove_duplicates=True):
        dfs = []
        for ds in datasets:
        # Keep only columns with no missing values in that dataset
            df_clean = ds.dataframe.dropna(axis=1)
            dfs.append(df_clean)

        combined = pd.concat(dfs, ignore_index=True)

        if remove_duplicates:
            combined = combined.drop_duplicates(subset = ['Reduced Composition', 'Ionic conductivity (S cm-1)'])

        return combined

    def remove_matching_entries(self, other):
        """
        Removes entries from the current dataset that are present in another dataset,
        comparing by 'Reduced Composition' (chemical equivalence) and DOI.
        If a composition matches and at least one of the matching rows in either dataset has the same DOI,
        all such rows in the current dataset are removed — even if the DOI is missing on some duplicates.

        Parameters:
        - other: An object with a 'dataframe' attribute or a pandas DataFrame containing 'Reduced Composition' and 'DOI' columns.

        Returns:
        - The updated DataFrame with matching entries removed.
        """
        
        if isinstance(other, pd.DataFrame):
            other_df = other
        else:
            other_df = other.dataframe

        # Get list of index positions to remove
        indices_to_remove = set()

        for i, self_row in self.dataframe.iterrows():
            self_comp = self_row['Reduced Composition']
            self_doi = self_row.get('DOI')

            for _, other_row in other_df.iterrows():
                other_comp = other_row['Reduced Composition']
                other_doi = other_row.get('DOI')

                if is_same_formula(self_comp, other_comp):
                    if (self_doi == other_doi) and pd.notna(self_doi) and pd.notna(other_doi):
                        for j, test_row in self.dataframe.iterrows():
                            if is_same_formula(test_row['Reduced Composition'], self_comp):
                                test_doi = test_row.get('DOI')
                                if (test_doi == self_doi) or pd.isna(test_doi):
                                    indices_to_remove.add(j)
                        break

        self.dataframe.loc[list(indices_to_remove)].to_csv('matching_entries.csv', index=False)

        new_dataframe = self.dataframe.drop(index=indices_to_remove)
        return Dataset(new_dataframe)
