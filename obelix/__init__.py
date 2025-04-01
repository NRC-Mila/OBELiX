import pandas as pd
import urllib.request
import tarfile
from pathlib import Path
from pymatgen.core import Structure
import warnings
from tqdm import tqdm
import importlib 
import re


from .utils import round_partial_occ, replace_text_IC

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

    def __init__(self, data_path="./lilon_data", no_cifs=False, commit_id=None, dev=False):
        '''
        Loads the LiIon dataset.
        
        '''
        
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "LiIonDatabase.csv"
        
        # Download data if it does not exist
        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id, dev=dev)

        df = self.read_data(self.data_path, no_cifs)
        super().__init__(df)
    
    def download_data(self, output_path, commit_id=None, dev=False):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if dev:
            from git import Repo
            print("Development mode: cloning the private repository...")
            Repo.clone_from("git@github.com:NRC-Mila/private-OBELiX.git", output_path)
        
        dataset_url = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/misc/LiIonDatabase.csv"
        df = pd.read_csv(dataset_url, index_col="ID")
        df.to_csv(output_path / "LiIonDatabase.csv")
    
    def read_data(self, data_path, no_cifs=False):
        '''Reads the LiIon dataset.'''
        data_file = data_path / "LiIonDatabase.csv"
        try:
            data = pd.read_csv(data_file, index_col="ID")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {data_file}. Please check the download path.")
        
        return data

    def is_same_formula(formula_string1, formula_string2):

        """
        Compares two formulas to determine if they represent the same composition
        """

        f1 = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string1)
        f2 = re.findall(r"([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string2)

        if len(f1) != len(f2):
            return False

        for elem, count in f1:
            if (elem, count) not in f2:
                return False

        return True

class Laskowski(Dataset):
    '''
    Laskowski dataset class.
    
    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    '''

    def __init__(self, data_path="./laskowski_data", no_cifs=False, commit_id=None, dev=False):
        '''
        Loads the Laskowski dataset.
        
        '''
        
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "digitized_data_for_SSEs.csv"
        
        # Download data if it does not exist
        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id, dev=dev)

        df = self.read_data(self.data_path, no_cifs)
        super().__init__(df)
    
    def download_data(self, output_path, commit_id=None, dev=False):
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        if dev:
            from git import Repo
            print("Development mode: cloning the private repository...")
            Repo.clone_from("git@github.com:NRC-Mila/private-OBELiX.git", output_path)
        
        dataset_url =  "https://raw.githubusercontent.com/FALL-ML/materials-discovery/blob/main/data/digitized_data_for_SSEs.csv"
        df = pd.read_csv(dataset_url)
        df.to_csv(output_path / "digitized_data_for_SSEs.csv")
    
    def read_data(self, data_path, no_cifs=False):
        '''Reads the Laskowski dataset.'''
        data_file = data_path / "digitized_data_for_SSEs.csv"
        try:
            data = pd.read_csv(data_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {data_file}. Please check the download path.")
        
        return data
    
