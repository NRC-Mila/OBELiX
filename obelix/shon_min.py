import re
import numpy as np
import pandas as pd
from pymatgen.core import Composition

# String normalization & scientific‐notation parser
def normalize_string(s: str) -> str:
    replacements = {
        "âˆ’": "-",
        "â€“": "-",
        "—": "-",
        "–": "-",
        "−": "-",    # proper minus sign
        "Ã": "×"
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
    m_sci = re.match(r"^\s*(?P<coef>-?\d+(?:\.\d+)?)?\s*[×x*]?\s*10\s*-\s*(?P<exp>\d+)\s*$", s)
    if m_sci:
        coef = float(m_sci.group("coef")) if m_sci.group("coef") not in (None, "") else 1.0
        exp = int(m_sci.group("exp"))
        return coef * 10 ** (-exp)

    # Search anywhere for sci pattern
    m_any = re.search(r"(?P<coef>-?\d+(?:\.\d+)?)?\s*[×x*]?\s*10\s*-\s*(?P<exp>\d+)", s)
    if m_any:
        coef = float(m_any.group("coef")) if m_any.group("coef") not in (None, "") else 1.0
        exp = int(m_any.group("exp"))
        return coef * 10 ** (-exp)

    # Fallback to float
    try:
        return float(s)
    except ValueError:
        return np.nan

# Unit conversion
UNIT_CONVERSION = {
    "S/cm":    1,
    "Scm-1":   1,
    "mScm−1":  1e-3,
    "mScm-1":  1e-3,
    "mS/cm":   1e-3
}

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
    df["Ionic Conductivity Numeric"] = df["Ionic Conductivity"].apply(convert_scientific_string)

    # 2) Convert to standard units (S/cm)
    df["Ionic Conductivity Numeric (S/cm)"] = df.apply(
        lambda r: convert_to_S_cm(r.get("Raw_unit", ""), r["Ionic Conductivity Numeric"]),
        axis=1
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
    return df_clean

