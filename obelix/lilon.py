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

class LiIon(Dataset):
    '''
    LiIon dataset class.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    '''

    def __init__(self, data_path="./lilon_rawdata", no_cifs=False, commit_id=None, rename_columns=True, room_temp_only=True):
        '''
        Loads the LiIon dataset.
        
        '''
        
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "LiIonDatabase.csv"
        
        # Download data if it does not exist
        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id)

        df = self.read_data(self.data_path, no_cifs)

        if rename_columns:
            df = df.rename(columns={'target': 'Ionic conductivity (S cm-1)', 'composition' : 'Reduced Composition', 'source' : 'DOI'})
        
        if room_temp_only:
            # Filter for temperatures within room temperature range
            room_temp = 25
            tolerance = 7
            temp_min = room_temp - tolerance
            temp_max = room_temp + tolerance

            # Keep rows where 'temperature' is within [temp_min, temp_max]
            if "temperature" in df.columns:
                df = df[(df["temperature"] >= temp_min) & (df["temperature"] <= temp_max)]

        super().__init__(df)
    
    def download_data(self, output_path, commit_id=None, local=True):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if local:
            dataset_url = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/misc/LiIonDatabase.csv"
        else:
            dataset_url = "https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.csv"

        df = pd.read_csv(dataset_url)
        df.to_csv(output_path / "LiIonDatabase.csv", header=0)
        
    
    def  read_data(self, data_path, no_cifs=False):
         '''Reads the LiIon dataset.'''
         df = pd.read_csv(self.data_path / "LiIonDatabase.csv", header=0)

         return df


    def  remove_obelix(self, obelix_object):
         """
         Removes entries from the LiIon dataset that are present in OBELiX.
         """

         ob_df = obelix_object.dataframe
         liion_ids = ob_df["Liion ID"].dropna().astype(int)

         self.dataframe.loc[self.dataframe.index.intersection(liion_ids)].to_csv('liIon_obelix_matching_entries.csv', index=False)

         return self.dataframe.drop(index=liion_ids, errors="ignore")


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

