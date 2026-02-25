import pandas as pd
from pathlib import Path

from .dataset import Dataset


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

        # laskowski_with_dois.csv is a curated version of the Laskowski
        # dataset with DOIs manually added.  It is hosted in the OBELiX
        # repo under data/misc/.
        # TODO: get Felix's feedback on long-term hosting of this file
        dataset_url = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/misc/laskowski_with_dois.csv"
        df = pd.read_csv(dataset_url)
        df.to_csv(output_path / "laskowski_with_dois.csv", index=False)

    def read_data(self, data_path, no_cifs=False):
        '''Reads the Laskowski dataset.'''
        data = pd.read_csv(self.data_path / "laskowski_with_dois.csv")
        return data

    def remove_obelix(self, obelix_object):
        """Remove entries from the Laskowski dataset that are present in OBELiX.

        Uses the ``'Laskowski ID'`` column in the OBELiX dataset to identify
        which Laskowski rows to drop (by index).

        Parameters:
            obelix_object: An OBELiX :class:`Dataset` whose dataframe
                contains a ``'Laskowski ID'`` column.

        Returns:
            A new :class:`Dataset` with the matching entries removed.
            The original dataset is **not** mutated.
        """
        ob_df = obelix_object.dataframe
        lask_ids = ob_df["Laskowski ID"].dropna().astype(int)
        new_df = self.dataframe.drop(index=lask_ids, errors="ignore")
        return Dataset(new_df)
