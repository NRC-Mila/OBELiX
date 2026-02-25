from pathlib import Path

import pandas as pd

from .dataset import Dataset

# Caltech Data Repository download URL (CC0 license)
_CALTECH_URL = (
    "https://data.caltech.edu/records/23mvv-6gk43"
    "/files/ionic_conductivity_database.csv?download=1"
)

_ICSD_NOT_IMPLEMENTED_MSG = (
    "ICSD lookup not yet implemented. Space group, lattice parameter, "
    "and CIF retrieval from ICSD collection codes is planned for a "
    "future release. See https://github.com/NRC-Mila/OBELiX/issues"
)


class McHaffie(Dataset):
    """McHaffie dataset class.

    Loads ionic conductivity data from McHaffie et al. (Digital Discovery,
    2025).  See https://doi.org/10.1039/D5DD00052A

    The dataset contains 571 room-temperature Li-ion conductivity
    measurements linked to ICSD crystal structure identifiers.  All
    conductivity values are room-temperature: 436 were measured directly
    at RT and 135 were extrapolated from Arrhenius plots.

    Space group, lattice parameters, and CIF data are not included in the
    distributed CSV but can be obtained from ICSD using the
    ``ICSD Collection Code`` column.  Stub methods
    (:meth:`get_space_groups`, :meth:`get_lattice_parameters`,
    :meth:`get_cifs`) are provided and will raise
    :exc:`NotImplementedError` until ICSD integration is added.

    Data source: https://data.caltech.edu/records/23mvv-6gk43

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    """

    def __init__(
        self,
        data_path="./obelixdata/mchaffie",
        no_cifs=False,
        commit_id=None,
        rename_columns=True,
        local=False,
    ):
        """Load the McHaffie dataset.

        Parameters:
            data_path: Directory to cache downloaded data.
            no_cifs: Unused, kept for API compatibility.
            commit_id: Unused, kept for API compatibility.
            rename_columns: If True, rename columns to the standard
                OBELiX schema.
            local: If True, copy from the repo's ``data/misc/mchaffie/``
                directory instead of downloading from Caltech Data.
        """
        self.data_path = Path(data_path)
        self.data_file = self.data_path / "ionic_conductivity_database.csv"

        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id, local=local)

        df = self.read_data(self.data_path, no_cifs)

        if rename_columns:
            df = df.rename(
                columns={
                    "compound": "Reduced Composition",
                    "conductivity_doi": "DOI",
                    "conductivity_siemens_per_cm": "Ionic conductivity (S cm-1)",
                    "icsd_collectioncode": "ICSD Collection Code",
                    "lowest_extrapolation_temperature_K": "Extrapolation Temperature (K)",
                }
            )

        df.index = [f"MCH_{i:04d}" for i in df.index]
        df.index.name = "ID"

        super().__init__(df)

    def download_data(self, output_path, commit_id=None, local=False):
        """Download and cache the McHaffie dataset.

        Parameters:
            output_path: Directory to write the cached CSV into.
            commit_id: Unused, kept for API compatibility.
            local: If True, copy from the repo's bundled data directory.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if local:
            import shutil

            repo_file = (
                Path(__file__).parent.parent
                / "data"
                / "misc"
                / "mchaffie"
                / "ionic_conductivity_database.csv"
            )
            shutil.copy2(repo_file, output_path / "ionic_conductivity_database.csv")
        else:
            df = pd.read_csv(_CALTECH_URL)
            df.to_csv(output_path / "ionic_conductivity_database.csv", index=False)

    def read_data(self, data_path, no_cifs=False):
        """Read the cached McHaffie CSV file.

        Parameters:
            data_path: Directory containing the cached CSV.
            no_cifs: Unused, kept for API compatibility.

        Returns:
            pd.DataFrame with the raw dataset.
        """
        return pd.read_csv(self.data_path / "ionic_conductivity_database.csv")

    def remove_obelix(self, obelix_object):
        """Remove entries that overlap with the OBELiX dataset.

        Since OBELiX has no dedicated ID column for McHaffie entries,
        this uses composition + DOI matching via
        :meth:`~obelix.dataset.Dataset.remove_matching_entries`.

        Parameters:
            obelix_object: An OBELiX :class:`Dataset`.

        Returns:
            A new :class:`Dataset` with matching entries removed.
            The original dataset is **not** mutated.
        """
        return self.remove_matching_entries(obelix_object)

    # ------------------------------------------------------------------
    # ICSD integration stubs (to be implemented)
    # ------------------------------------------------------------------

    def get_space_groups(self):
        """Retrieve space groups for all entries from ICSD.

        Requires ICSD access via the ``ICSD Collection Code`` column.
        Not yet implemented.

        Raises:
            NotImplementedError: Always, until ICSD integration is added.
        """
        raise NotImplementedError(_ICSD_NOT_IMPLEMENTED_MSG)

    def get_lattice_parameters(self):
        """Retrieve lattice parameters (a, b, c, alpha, beta, gamma) from ICSD.

        Requires ICSD access via the ``ICSD Collection Code`` column.
        Not yet implemented.

        Raises:
            NotImplementedError: Always, until ICSD integration is added.
        """
        raise NotImplementedError(_ICSD_NOT_IMPLEMENTED_MSG)

    def get_cifs(self):
        """Retrieve CIF files for all entries from ICSD.

        Requires ICSD access via the ``ICSD Collection Code`` column.
        Not yet implemented.

        Raises:
            NotImplementedError: Always, until ICSD integration is added.
        """
        raise NotImplementedError(_ICSD_NOT_IMPLEMENTED_MSG)
