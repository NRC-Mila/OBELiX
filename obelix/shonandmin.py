"""The Shon and Min dataset contains many particulars that we manually normalize here to be cross-compatible with OBELiX
"""

from pathlib import Path

import pandas as pd

from .dataset import Dataset


import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition


# Unit conversion
UNIT_CONVERSION = {"S/cm": 1, "Scm-1": 1, "mScm−1": 1e-3, "mScm-1": 1e-3, "mS/cm": 1e-3}


# String normalization & scientific‐notation parser
def normalize_string(s: str) -> str:
    replacements = {
        "âˆ’": "-",
        "â€“": "-",
        "—": "-",
        "–": "-",
        "−": "-",  # proper minus sign
        "Ã": "×",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s


#  Convert a wide variety of human‐written scientific notations and simple ranges into a float (or np.nan if unconvertible).
def convert_scientific_string(s) -> float:
    s = str(s).strip()
    if not s:
        return np.nan

    s = normalize_string(s)
    # Strip leading "to"
    if s.lower().startswith("to"):
        s = s[2:].strip()

    # Handle "to" ranges → average
    if "to" in s:
        parts = [p.strip() for p in s.split("to") if p.strip()]
        vals = [convert_scientific_string(p) for p in parts]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else np.nan

    # Handle "and" lists → first valid
    if "and" in s:
        for part in [p.strip() for p in s.split("and")]:
            v = convert_scientific_string(part)
            if not np.isnan(v):
                return v
        return np.nan

    # Handle commas → first valid
    if "," in s:
        for part in [p.strip() for p in s.split(",")]:
            v = convert_scientific_string(part)
            if not np.isnan(v):
                return v
        return np.nan

    # Explicit "10 - exp" → 1e‑exp
    m_10 = re.match(r"^\s*10\s*-\s*(\d+(?:\.\d+)?)", s)
    if m_10:
        try:
            exp = float(m_10.group(1))
            return 10 ** (-exp)
        except ValueError:
            return np.nan

    # Simple numeric range "a - b" → average or scientific if a==10
    m_range = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$", s)
    if m_range:
        a, b = float(m_range.group(1)), float(m_range.group(2))
        if a == 10:
            return 10 ** (-b)
        return (a + b) / 2.0

    # Standard sci notation "coef × 10 - exp"
    m_sci = re.match(
        r"^\s*(?P<coef>-?\d+(?:\.\d+)?)?\s*[×x*]?\s*10\s*-\s*(?P<exp>\d+)\s*$", s
    )
    if m_sci:
        coef = (
            float(m_sci.group("coef")) if m_sci.group("coef") not in (None, "") else 1.0
        )
        exp = int(m_sci.group("exp"))
        return coef * 10 ** (-exp)

    # Search anywhere for sci pattern
    m_any = re.search(r"(?P<coef>-?\d+(?:\.\d+)?)?\s*[×x*]?\s*10\s*-\s*(?P<exp>\d+)", s)
    if m_any:
        coef = (
            float(m_any.group("coef")) if m_any.group("coef") not in (None, "") else 1.0
        )
        exp = int(m_any.group("exp"))
        return coef * 10 ** (-exp)

    # Fallback to float
    try:
        return float(s)
    except ValueError:
        return np.nan


def convert_to_S_cm(unit: str, value: float) -> float:
    """Convert given value into S/cm based on its raw unit."""
    factor = UNIT_CONVERSION.get(unit, None)
    if factor is None:
        return value
    return value * factor


# Formula validation
def is_valid_formula(formula: str) -> bool:
    """Return True if pymatgen can parse the formula into a non-empty composition."""
    try:
        comp = Composition(formula)
        return len(comp.get_el_amt_dict()) > 0
    except Exception:
        return False


def clean_shon_min(df):
    # 1) Parse ionic conductivity strings
    df["Ionic Conductivity Numeric"] = df["Ionic Conductivity"].apply(
        convert_scientific_string
    )

    # 2) Convert to standard units (S/cm)
    df["Ionic Conductivity Numeric (S/cm)"] = df.apply(
        lambda r: convert_to_S_cm(
            r.get("Raw_unit", ""), r["Ionic Conductivity Numeric"]
        ),
        axis=1,
    )

    # 3) Validate & filter rows
    to_drop = []
    for idx, row in df.iterrows():
        cond = row["Ionic Conductivity Numeric (S/cm)"]
        if pd.isna(cond) or cond <= 0 or not (-18 <= np.log10(cond) <= 0):
            to_drop.append(idx)
            continue
        name = row.get("Name", "")
        if pd.isna(name) or not is_valid_formula(name):
            to_drop.append(idx)
            continue

    df_clean = df.drop(index=to_drop)

    # Replace original string column with cleaned numeric values (S/cm)
    # and drop intermediate columns
    df_clean = df_clean.copy()
    df_clean["Ionic Conductivity"] = df_clean["Ionic Conductivity Numeric (S/cm)"]
    df_clean = df_clean.drop(
        columns=["Ionic Conductivity Numeric", "Ionic Conductivity Numeric (S/cm)"],
        errors="ignore",
    )

    return df_clean


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

        df.index = [f"SM_{i:04d}" for i in df.index]
        df.index.name = "ID"

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
        df = pd.read_csv(data_path / "sheet2.csv")
        df = df.drop(
            columns=[c for c in df.columns if c.startswith("Unnamed:")],
            errors="ignore",
        )
        return df

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
