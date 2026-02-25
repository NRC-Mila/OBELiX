from pathlib import Path

import pandas as pd

from .dataset import Dataset
from .shon_min import clean_shon_min


class ShonAndMin(Dataset):
    """
    ShonAndMin dataset class.

    Loads ionic conductivity data from the supplementary information of
    Shon and Min (ACS Omega, 2023).  See
    https://doi.org/10.1021/acsomega.3c01424

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    """

    def __init__(
        self,
        data_path="./obelixdata/shonandmin",
        no_cifs=False,
        clean_data=True,
        commit_id=None,
        keep_min_conductivity=True,
        rename_columns=True,
        room_temp_only=True,
        local=False,
    ):
        """
        Loads and cleans the ShonAndMin dataset.

        Parameters:
            data_path: Directory to cache downloaded data.
            no_cifs: Unused, kept for API compatibility.
            clean_data: If True, parse and validate ionic conductivity
                values using :func:`~obelix.shon_min.clean_shon_min`.
            commit_id: Unused, kept for API compatibility.
            keep_min_conductivity: If True, keep only the entry with the
                minimum conductivity per (Reduced Composition, DOI) pair.
            rename_columns: If True, rename columns to the standard
                OBELiX schema.
            room_temp_only: If True, filter for rows within 25 +/- 7 C.
            local: If True, read from the repo's bundled data directory
                instead of downloading from ACS.
        """
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "sheet2.csv"

        # Download data if it does not exist
        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id, local=local)

        df = self.read_data(self.data_path, no_cifs)

        if clean_data:
            df = clean_shon_min(df)

        if rename_columns:
            df = df.rename(
                columns={
                    "Name": "Reduced Composition",
                    "Ionic Conductivity": "Ionic conductivity (S cm-1)",
                }
            )

        if keep_min_conductivity:
            # Keep only one entry per unique (Reduced Composition, DOI) pair,
            # selecting the row with the minimum Ionic Conductivity.
            df = df.loc[
                df.groupby(["Reduced Composition", "DOI"])[
                    "Ionic conductivity (S cm-1)"
                ].idxmin()
            ]

        if room_temp_only:
            room_temp = 25
            tolerance = 7
            temp_min = room_temp - tolerance
            temp_max = room_temp + tolerance
            if "temperature" in df.columns:
                df = df[
                    (df["temperature"] >= temp_min) & (df["temperature"] <= temp_max)
                ]

        super().__init__(df)

    def download_data(self, output_path, commit_id=None, local=False):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if local:
            # Read from the repo's bundled data directory
            repo_file = (
                Path(__file__).parent.parent / "data" / "misc" / "ao3c01424_si_001.xlsx"
            )
            df = pd.read_excel(repo_file, sheet_name="Sheet2")
        else:
            # Download directly from ACS supplementary information
            dataset_url = (
                "https://pubs.acs.org/doi/suppl/10.1021/acsomega.3c01424"
                "/suppl_file/ao3c01424_si_001.xlsx"
            )
            df = pd.read_excel(dataset_url, sheet_name="Sheet2")

        df.to_csv(output_path / "sheet2.csv", index=False)

    def read_data(self, data_path, no_cifs=False):
        """Reads the ShonAndMin dataset."""
        return pd.read_csv(data_path / "sheet2.csv")

    def remove_obelix(self, obelix_object):
        """Remove entries that overlap with the OBELiX dataset.

        Since OBELiX has no dedicated ID column for ShonAndMin entries,
        this uses composition + DOI matching via
        :meth:`~obelix.dataset.Dataset.remove_matching_entries`.

        Parameters:
            obelix_object: An OBELiX :class:`Dataset`.

        Returns:
            A new :class:`Dataset` with matching entries removed.
            The original dataset is **not** mutated.
        """
        return self.remove_matching_entries(obelix_object)
