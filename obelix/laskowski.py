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
from obelix.dataset import Dataset
#from obelix import OBELiX

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
                'Structure': 'Reduced Composition', 'space group': 'Space group'
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

