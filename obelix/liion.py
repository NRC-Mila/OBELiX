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

class LiIon(Dataset):
    '''
    LiIon dataset class.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    '''

    def __init__(self, data_path="./liion_rawdata", no_cifs=False, commit_id=None, rename_columns=True, room_temp_only=True):
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
            df = df.rename(columns={'target': 'Ionic conductivity (S cm-1)', 'composition' : 'Reduced Composition', 'source' : 'DOI', 
            'family' : 'Family'})
        
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
        df.to_csv(output_path / "LiIonDatabase.csv", index=False)
        
    
    def read_data(self, data_path, no_cifs=False):
        '''Reads the LiIon dataset.'''
        df = pd.read_csv(self.data_path / "LiIonDatabase.csv")
        return df

    def remove_obelix(self, obelix_object):
        """
        Removes entries from the LiIon dataset that are present in OBELiX.
        """
        ob_df = obelix_object.dataframe
        liion_ids = ob_df["Liion ID"].dropna().astype(int)

        self.dataframe.loc[self.dataframe.index.intersection(liion_ids)].to_csv('liion_obelix_matching_entries.csv', index=False)
        return self.dataframe.drop(index=liion_ids, errors="ignore")


