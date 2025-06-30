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
from obelix.shon_min import clean_shon_min


class ShonAndMin(Dataset):
    def __init__(self, data_path="./shonandmin_rawdata", no_cifs=False, clean_data=True, commit_id=None, keep_min_conductivity=True,
            rename_columns=True):
        '''
        Loads and cleans the ShonAndMin dataset.
        '''
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "sheet2.csv"  

        # Download data if it does not exist
        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id)

        df = self.read_data(self.data_path, no_cifs)
        
        if clean_data:
            df = clean_shon_min(df)

        if rename_columns:
            df = df.rename(columns={'Name': 'Reduced Composition', 'Ionic Conductivity':'Ionic conductivity (S cm-1)'
            })

        if keep_min_conductivity:
           """
           Keeps only one entry per unique (Reduced Composition, DOI) pair,selecting the row with the minimum Ionic Conductivity.
           Returns the filtered DataFrame.
           """
           df = df.loc[df.groupby(['Reduced Composition', 'DOI'])['Ionic conductivity (S cm-1)'].idxmin()]

        super().__init__(df)

    def download_data(self, output_path, commit_id=None):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        dataset_url = "https://raw.githubusercontent.com/leah-mungai/leah---OBELiX/new_taset_classes/data/ao3c01424_si_001.xlsx"
        df = pd.read_excel(dataset_url, sheet_name="Sheet2")  
        df.to_csv(output_path / "sheet2.csv", index=False)

    def read_data(self, data_path, no_cifs=False):
        '''Reads the ShonAndMin dataset.'''
        return pd.read_csv(data_path / "sheet2.csv", index_col=0) 

