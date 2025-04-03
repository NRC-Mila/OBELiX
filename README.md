# OBELiX: An Experimental Ionic Conductivity Dataset

OBELiX (<ins>O</ins>pen solid <ins>B</ins>attery <ins>E</ins>lectrolytes with <ins>Li</ins>: an e<ins>X</ins>perimental dataset) is a dataset of 599 synthesized solid electrolyte materials and their **experimentally measured room temperature ionic conductivity** along with descriptors of their space group, lattice parameters, and chemical composition. It contains full crystallographic description in the form of CIF files for 321 entries. 

A full description an analysis can be found in [our paper](https://arxiv.org/abs/2502.14234)

<h1 align="center">
<img src="https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/paper/figures/gathered.svg">

## Files Download
| File               | Links   |
| --------           | ------- |
| Train Dataset       | [xlsx](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.xlsx), [csv](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.csv)|
| Train CIF files      | [zip](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train_cifs.zip), [tar.gz](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train_cifs.tar.gz)    |
| Test Dataset       | [xlsx](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.xlsx), [csv](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.csv)|
| Test CIF files      | [zip](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test_cifs.zip), [tar.gz](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test_cifs.tar.gz)    |
| Full Dataset       | [xlsx](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.xlsx), [csv](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.csv)|
| All CIF files      | [zip](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all_cifs.zip), [tar.gz](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all_cifs.tar.gz)    |

## Python API

### Installation

```
pip install obelix-data
```

### Examples

Print the composition and ionic conductivity of the first entry

```
from obelix import OBELiX

ob = OBELiX()

print(f"The ionic conductivity of {ob[0]['Reduced Composition']} is {ob[0]['Ionic conductivity (S cm-1)']}")
```

Print all the labels and IDs:

```
print("Column labels:", ob.labels)
print("Entry IDs:", ob.entries)
```

Use the entry ID to get information about it
```
print(f"The ID of the first entry is {ob.entries[0]}")

print(f"The ionic conductivity of {ob['jqc']['Reduced Composition']} is {ob['jqc']['Ionic conductivity (S cm-1)']}")
```

Plot the distribution of ionic conductivities in the train set:

```
import numpy as np
import matplotlib.pyplot as plt

np.log10(ob.train_dataset.dataframe["Ionic conductivity (S cm-1)"]).plot.hist()
plt.show()
```

Print the entries in the test dataset that have a CIF file:
```
print(ob.test_dataset.with_cifs().entries)
```

Round partial occupancies of all CIF files and save them in a folder named `np_cifs`

```
from pathlib import Path

output_path = "np_cifs"
output_path.mkdir(exist_ok=True)

for entry in ob.round_partial().with_cifs():
    entry["structure"].to(output_path+entry["ID"]+".cif")

'''
class LiIon examples
'''

import re
import pandas as pd
from obelix import LiIon, OBELiX, Dataset

ob = OBELiX()
li = LiIon()

is_same_formula = LiIon.is_same_formula

'''
Find matching compositions
'''
matched_rows = []

for i, li_comp in enumerate(li.dataframe["composition"]):
    for j, ob_comp in enumerate(ob.dataframe["Reduced Composition"]):
        if is_same_formula(li_comp, ob_comp):
            matched_rows.append((li_comp, ob_comp))

matched_comp_df = pd.DataFrame(matched_rows, columns=["LiIon Composition", "OBELiX Reduced Composition"])

'''
Find matching compositions and DOIs
'''

matched_rows = []

for i, li_comp in enumerate(li.dataframe["composition"]):
    li_doi = li.dataframe.iloc[i, li.dataframe.columns.get_loc("source")]  # Get DOI from LiIon dataframe

    for j, ob_comp in enumerate(ob.dataframe["Reduced Composition"]):
        ob_doi = ob.dataframe.iloc[j, ob.dataframe.columns.get_loc("DOI")]  # Get DOI from OBELiX dataframe

        if is_same_formula(li_comp, ob_comp):  # Check if formulas match
            if li_doi == ob_doi:  # Check if DOI matches
                matched_rows.append((li_comp, ob_comp, li_doi, ob_doi))
            else:
                print(f"Formula matches but DOI is different: {li_comp} (DOI: {li_doi}) vs {ob_comp} (DOI: {ob_doi})")

matched_comp_doi_df = pd.DataFrame(matched_rows, columns=["LiIon Composition", "OBELiX Reduced Composition", "source", "DOI"])

'''
Print results
'''
print("Number of fully matching compositions:", len(matched_comp_df))
print(matched_comp_df)
print("Number of fully matching compositions (including DOI):", len(matched_comp_doi_df))
print(matched_comp_doi_df)

'''
Remove matched entries from LiIon
'''
li_comp = li.dataframe[~li.dataframe["composition"].isin(matched_comp_df["LiIon Composition"])]
li_comp_doi = li.dataframe[~li.dataframe["composition"].isin(matched_comp_doi_df["LiIon Composition"])]

'''
Save the filtered dataset
'''
li_comp.to_csv("li_comp.csv", index=False)
li_comp_doi.to_csv("li_comp_doi.csv", index=False)

```
## Labels and Features

| Name | Definition |
|------|------------|
| `ID`  | Entry identifier, a three symbols alphanumeric string|
| `Reduced Composition` | Chemical formula of the material |
| `Z` | Number of formula units in the unit cell |
| `True Composition` | Composition of the unit cell |
| `Ionic conductivity (S cm-1)` | Experimental ionic conductivity in Siemens per centimeter. It is the total ionic conductivity when both are reported |
| `IC (Total)` | Total ionic conductivity in S/cm| 
| `IC (Bulk)` | Bulk Ionic conductivity in S/cm, if both total and bulk are blank, it is assumed that the reported conductivity is the total IC|
| `Space group` | Space group Hermann–Mauguin notation short name|
| `Space group #` | Space group number |
| `a`, `b`, `c`, `alpha`, `beta`, `gamma` | Lattice parameters |
| `Family` | Crystal family | 
| `DOI` | Digital object identifier of the original experimental publication from which this IC measurement was taken|
| `Checked` | (*OUTDATED*) Weather entries was manually checked |
| `Ref` | (*OUTDATED*, see `Laskowski ID` and `Liion ID` instead)) Reference from which data was taken from D1=Liion, D2=Laskowski |
| `Cif ID` | Whether the entry has a cif or not. This field will read "done" when it does and will be empty otherwise|
| `Cif ref_1`, `Cif ref_2` | (*OUTDATED*, see `ICSD ID`) Notes of where to find the CIF information if available|
| `note` | Notes and comments about the entry |
| `close match` | Whether the cif file comes from a closely matching structure or from the actual publication (DOI). If a close match this field will read "Yes"
| `close match DOI` | Digital object identifier of the publication from which the CIF was taken|
| `ICSD ID` | Inorganic Crystal Structure Database ID of the structure if it can be found in that database |
| `Laskowski ID` | Entry number (in order of appearance) in the supplementary information of [Forrest A. L. Laskowski et al., Energy Environ. Sci., 16, 1264 (2023)](https://pubs.rsc.org/en/content/articlelanding/2023/ee/d2ee03499a#!) if the entry is also in that database|
| `Liion ID` | Entry number in the [The Liverpool Ionics Dataset](http://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html) if the entry is also in that database|

## Citation

If you use OBELiX, please cite [our paper](https://arxiv.org/abs/2502.14234)

```
@article{therrien2025obelix,
  title   = {OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes},
  author  = {Félix Therrien and Jamal Abou Haibeh and Divya Sharma and Rhiannon Hendley and Alex Hernández-García and Sun Sun and Alain Tchagang and Jiang Su and Samuel Huberman and Yoshua Bengio and Hongyu Guo and Homin Shin},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2502.14234}
}

```


