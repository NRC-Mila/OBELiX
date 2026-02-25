import io
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

from .dataset import Dataset

# Primary source: OBELiX GitHub mirror (reliable, clean CSV)
_GITHUB_URL = (
    "https://raw.githubusercontent.com/NRC-Mila/OBELiX"
    "/main/data/misc/LiIonDatabase.csv"
)
# Original source: University of Liverpool personal page (may go offline)
# Reference: Hargreaves et al., npj Computational Materials, 2022
# https://doi.org/10.1038/s41524-022-00951-z
_LIVERPOOL_URL = "https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.csv"


class LiIon(Dataset):
    """
    LiIon dataset class.

    Data originally published by Hargreaves et al. (npj Computational
    Materials, 2022).  The default download source is the OBELiX GitHub
    mirror.  The original University of Liverpool URL is used as a
    fallback; if both fail a warning is printed so the user can
    download the file manually.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset.
    """

    def __init__(
        self,
        data_path="./obelixdata/liion",
        no_cifs=False,
        commit_id=None,
        rename_columns=True,
        room_temp_only=True,
        local=False,
    ):
        """
        Loads the LiIon dataset.

        Parameters:
            data_path: Directory to cache downloaded data.
            no_cifs: Unused, kept for API compatibility.
            commit_id: Unused, kept for API compatibility.
            rename_columns: If True, rename columns to the standard OBELiX schema.
            room_temp_only: If True, filter for rows within 25 +/- 7 C.
            local: If True, copy from the repo's bundled data directory
                instead of downloading.
        """

        self.data_path = Path(data_path)
        self.data_file = self.data_path / "LiIonDatabase.csv"

        # Download data if it does not exist
        if not self.data_file.exists():
            self.download_data(self.data_path, commit_id=commit_id, local=local)

        df = self.read_data(self.data_path, no_cifs)

        if rename_columns:
            df = df.rename(
                columns={
                    "target": "Ionic conductivity (S cm-1)",
                    "composition": "Reduced Composition",
                    "source": "DOI",
                    "family": "Family",
                }
            )

        if room_temp_only:
            # Filter for temperatures within room temperature range
            room_temp = 25
            tolerance = 7
            temp_min = room_temp - tolerance
            temp_max = room_temp + tolerance

            # Keep rows where 'temperature' is within [temp_min, temp_max]
            if "temperature" in df.columns:
                df = df[
                    (df["temperature"] >= temp_min) & (df["temperature"] <= temp_max)
                ]

        super().__init__(df)

    def download_data(self, output_path, commit_id=None, local=False):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        dest = output_path / "LiIonDatabase.csv"

        if local:
            import shutil

            repo_file = (
                Path(__file__).parent.parent / "data" / "misc" / "LiIonDatabase.csv"
            )
            shutil.copy2(repo_file, dest)
            return

        # Try the GitHub mirror first, fall back to University of Liverpool
        for label, url in [
            ("GitHub mirror", _GITHUB_URL),
            ("University of Liverpool", _LIVERPOOL_URL),
        ]:
            try:
                df = self._read_csv_from_url(url)
                df.to_csv(dest, index=False)
                return
            except Exception as exc:
                warnings.warn(
                    f"Failed to download LiIon data from {label} " f"({url}): {exc}"
                )

        raise RuntimeError(
            "Could not download LiIonDatabase.csv from any source. "
            "You can download it manually from:\n"
            f"  {_GITHUB_URL}\n"
            f"  {_LIVERPOOL_URL}\n"
            f"and place it at: {dest}"
        )

    @staticmethod
    def _read_csv_from_url(url):
        """Read a CSV from *url*, skipping any non-CSV preamble lines.

        The University of Liverpool version of LiIonDatabase.csv starts
        with two lines of licensing/citation text before the actual CSV
        header.  This method detects that and skips the preamble so
        ``pd.read_csv`` sees a clean header row.
        """
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8-sig")

        lines = raw.splitlines(keepends=True)
        # Find the header line by looking for the expected first column
        skip = 0
        for i, line in enumerate(lines):
            if line.startswith("ID,"):
                skip = i
                break

        return pd.read_csv(io.StringIO("".join(lines[skip:])))

    def read_data(self, data_path, no_cifs=False):
        """Reads the LiIon dataset."""
        df = pd.read_csv(self.data_path / "LiIonDatabase.csv")
        return df

    def remove_obelix(self, obelix_object):
        """Remove entries from the LiIon dataset that are present in OBELiX.

        First drops rows by ``'Liion ID'`` index matching, then removes
        any remaining duplicates by composition + DOI matching via
        :meth:`~obelix.dataset.Dataset.remove_matching_entries`.

        Parameters:
            obelix_object: An OBELiX :class:`Dataset` whose dataframe
                contains ``'Liion ID'``, ``'Reduced Composition'``, and
                ``'DOI'`` columns.

        Returns:
            A new :class:`Dataset` with the matching entries removed.
            The original dataset is **not** mutated.
        """
        ob_df = obelix_object.dataframe
        liion_ids = ob_df["Liion ID"].dropna().astype(int)
        after_id = Dataset(self.dataframe.drop(index=liion_ids, errors="ignore"))
        return after_id.remove_matching_entries(obelix_object)
