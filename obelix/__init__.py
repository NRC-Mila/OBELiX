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

from .utils import round_partial_occ, replace_text_IC, is_same_formula

from .shon_min import clean_shon_min

__version__ = importlib.metadata.version("obelix-data")

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

class OBELiX(Dataset):
    '''
    OBELiX dataset class.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
        train_dataset (Dataset): Dataset containing the training entries.
        test_dataset (Dataset): Dataset containing the test entries.
        entries (list): List of entries.
    '''

    def __init__(self, data_path="./rawdata", no_cifs=False, commit_id=f"v{__version__}-data", dev=False, unspecified_low_value=1e-15):
        '''
        Loads the OBELiX dataset.
        
        Args:
            data_path (str): Path to the data directory. If the directory does not exist, the data will be downloaded.
            no_cifs (bool): If True, the CIFs will not be read. Default: False
            commit_id (str): Commit ID. By default the data corresponding to the version of the package (`obelix.__version__`) will be downloaded. To use the latest realease, set `commit_id="main"`.
            dev (bool): If True, the data will be downloaded from the private repository. Default: False
            unspecified_low_value (float): Value to replace "<1E-10" and "<1E-8" in the "Ionic conductivity (S cm-1)" column. If None, the values will not be replaced. Default: 1e-15
        '''
        
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.download_data(self.data_path, commit_id=commit_id, dev=dev)

        df = self.read_data(self.data_path, no_cifs)

        if unspecified_low_value is not None:
            df["Ionic conductivity (S cm-1)"] = df["Ionic conductivity (S cm-1)"].apply(replace_text_IC, args=(unspecified_low_value,))
            
        super().__init__(df)

        if (self.data_path / "test.csv").exists():
            test = pd.read_csv(self.data_path / "test.csv", index_col="ID")
        else:
            test = pd.read_csv(self.data_path / "test_idx.csv", index_col="ID")

        self.train_dataset = Dataset(self.dataframe[~self.dataframe.index.isin(test.index)])
        
        self.test_dataset = Dataset(self.dataframe[self.dataframe.index.isin(test.index)])
        
    def download_data(self, output_path, commit_id=None, dev=False):
        output_path = Path(output_path)
        if dev:
            from git import Repo
            print("Development mode: cloning the private repository...")
            Repo.clone_from("git@github.com:NRC-Mila/private-OBELiX.git", output_path)

            xlsx_url = "https://github.com/NRC-Mila/OBELiX/raw/refs/heads/main/data/raw.xlsx"
            df = pd.read_excel(xlsx_url, index_col="ID")
            df.to_csv(output_path / "all.csv")
            
            test_csv_url = "https://github.com/NRC-Mila/OBELiX/raw/refs/heads/main/data/test_idx.csv"
            df = pd.read_csv(test_csv_url, index_col="ID")
            df.to_csv(output_path / "test.csv")
            
        else:
            print("Downloading data...", end="")
            output_path.mkdir(exist_ok=True)
            
            if commit_id is None:
                commit_id = "main"

            tar_url = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{commit_id}/data/downloads/all_cifs.tar.gz"
            
            fileobj = urllib.request.urlopen(tar_url)
            tar = tarfile.open(fileobj=fileobj, mode="r|gz")
            tar.extractall(output_path, filter="data")
            
            csv_url = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{commit_id}/data/downloads/all.csv"
            df = pd.read_csv(csv_url, index_col="ID")
            df.to_csv(output_path / "all.csv")
            
            test_csv_url = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{commit_id}/data/downloads/test.csv"
            df = pd.read_csv(test_csv_url, index_col="ID")
            df.to_csv(output_path / "test.csv")
            
            print("Done.")
        
    def read_data(self, data_path, no_cifs=False):

        try:
            data = pd.read_csv(self.data_path / "all.csv", index_col="ID")
        except FileNotFoundError:
            data = pd.read_excel(self.data_path / "raw.xlsx", index_col="ID")
            
        if no_cifs:
            return data

        if (Path(data_path) / "anon_cifs").exists():
            cif_path = Path(data_path) / "anon_cifs"
            print("Reading original CIFs...")
        else:
            cif_path = Path(data_path) / "all_randomized_cifs"
            print("Reading randomized CIFs...")
            
        struc_dict = {}
            
        for i, row in tqdm(data.iterrows(), total=len(data)):

            filename = (cif_path / i).with_suffix(".cif")
        
            if row["Cif ID"] == "done":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="We strongly encourage explicit .*")
                    warnings.filterwarnings("ignore", message="Issues encountered .*")
                    structure = Structure.from_file(filename)
            else:
                structure = None
                    
            struc_dict[i] = structure

        data["structure"] = pd.Series(struc_dict)
        
        return data    

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
    
    def download_data(self, output_path, commit_id=None, local=False):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if local:
            dataset_url = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/misc/LiIonDatabase.csv"
        else:
            dataset_url = "https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.csv"

        df = pd.read_csv(dataset_url)
        df.to_csv(output_path / "LiIonDatabase.csv", header=2)
        
    
    def  read_data(self, data_path, no_cifs=False):
         '''Reads the LiIon dataset.'''
         df = pd.read_csv(self.data_path / "LiIonDatabase.csv", header=2)

         return df


    def  remove_obelix(self, obelix_object):
         """
         Removes entries from the LiIon dataset that are present in OBELiX.
         """

         ob_df = obelix_object.dataframe
         liion_ids = ob_df["Liion ID"].dropna().astype(int)
         return self.dataframe.drop(index=liion_ids, errors="ignore")



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

    def remove_obelix(self, obelix_object):
        """
        Removes entries from the Laskowski dataset that are present in OBELiX,
        comparing by 'Reduced Composition' (chemical equivalence) and DOI.
        If a composition matches and at least one of the matching rows in either dataset has the same DOI,
        all such rows in Laskowski are removed — even if the DOI is missing on some duplicates.
        """

        ob_df = obelix_object.dataframe[['Reduced Composition', 'DOI']].dropna(subset=['Reduced Composition'])
        current_df = self.dataframe.dropna(subset=['Reduced Composition'])

        # Get list of index positions to remove
        indices_to_remove = set()

        for i, curr_row in current_df.iterrows():
            curr_comp = curr_row['Reduced Composition']
            curr_doi = curr_row.get('DOI')

            for _, ob_row in ob_df.iterrows():
                ob_comp = ob_row['Reduced Composition']
                ob_doi = ob_row.get('DOI')

                if is_same_formula(curr_comp, ob_comp):
                    # Check if DOI matches or if either DOI is missing (to catch all duplicates)
                    if (curr_doi == ob_doi) or (pd.notna(curr_doi) and pd.notna(ob_doi) and curr_doi == ob_doi):
                        # Mark ALL rows in current_df with same formula for removal
                        for j, test_row in current_df.iterrows():
                            if is_same_formula(test_row['Reduced Composition'], curr_comp):
                                indices_to_remove.add(j)
                        break  # Once a match is found, stop checking OBELiX rows

        # Build new dataframe without the matched entries
        self.dataframe = current_df.drop(index=indices_to_remove)
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

class ShonAndMin(Dataset):
    def __init__(self, data_path="./SM_rawdata", no_cifs=False, clean_data=True, commit_id=None, rename_columns=True):
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




