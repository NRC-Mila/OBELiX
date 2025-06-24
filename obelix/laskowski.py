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

from obelix import Dataset, OBELiX

class Laskowski(Dataset):
    '''
    Laskowski dataset class.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    '''

    def __init__(self, data_path="./laskowski_rawdata", no_cifs=False, commit_id=None, rename_columns=True):
        '''
        Loads the Laskowski dataset.
        
        '''
        
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "laskowski_with_dois.csv"
        
        # Download data if it does not exist

        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id)

        df = self.read_data(self.data_path, no_cifs)

        if rename_columns:
            df = df.rename(columns={
                'σ(RT)(S cm-1)': 'Ionic conductivity (S cm-1)',
                'Structure': 'Reduced Composition'
            })

        super().__init__(df)
    
    def download_data(self, output_path, commit_id=None):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        dataset_url = "https://raw.githubusercontent.com/leah-mungai/leah---OBELiX/new_taset_classes/data/misc/laskowski_with_dois.csv"
        df = pd.read_csv(dataset_url)
        df.to_csv(output_path / "laskowski_with_dois.csv", index=False)
    
    def read_data(self, data_path, no_cifs=False):
        '''Reads the Laskowski dataset.'''
        data = pd.read_csv(self.data_path / "laskowski_with_dois.csv", index_col=0)
        
        return data

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
                    # Check if DOI matches or if either DOI is missing (to catch all duplicates)
                    if (self_doi == other_doi) or (pd.notna(self_doi) and pd.notna(other_doi) and self_doi == other_doi):
                        # Mark ALL rows in current_df with same formula for removal
                        for j, test_row in self.dataframe.iterrows():
                            if is_same_formula(test_row['Reduced Composition'], self_comp):
                                indices_to_remove.add(j)
                        break  # Once a match is found, stop checking other_df rows

        self.dataframe.loc[list(indices_to_remove)].to_csv('matching_entries.csv', index=False)

        # Build new DataFrame without the matched entries
        self.dataframe = self.dataframe.drop(index=indices_to_remove)
        return self.dataframe

    def print_composition_matches_with_missing_doi(self, obelix_object):
        """
        Prints compositions where OBELiX and the current dataset match by 'Reduced Composition',
        and none of the matching entries (in both datasets) have a DOI.
        """

        ob_df = obelix_object.dataframe.dropna(subset=['Reduced Composition'])
        curr_df = self.dataframe.dropna(subset=['Reduced Composition'])

        # Combine and reset index for easier reference
        combined_df = pd.concat([
            ob_df[['Reduced Composition', 'DOI']],
            curr_df[['Reduced Composition', 'DOI']]
        ], ignore_index=True)

        checked = set()
        matched_comps = []

        for i, row_i in combined_df.iterrows():
            comp_i = row_i['Reduced Composition']
            if comp_i in checked:
                continue

            matches = []
            for j, row_j in combined_df.iterrows():
                comp_j = row_j['Reduced Composition']
                if is_same_formula(comp_i, comp_j):
                    matches.append(row_j)

            if len(matches) < 2:
                continue  # skip: no matching pair found

            # Check if all DOIs are missing in these matched rows
            if all(pd.isna(row['DOI']) for row in matches):
                matched_comps.append(comp_i)

            # Mark all matched compositions as checked
            for row in matches:
                checked.add(row['Reduced Composition'])

        if matched_comps:
            print("Compositions that match across datasets and have missing DOI in all entries:")
            for comp in matched_comps:
                print("  -", comp)
            print("Total:", len(matched_comps))
        else:
            print("No matching compositions found with all DOIs missing.")

        return matched_comps

