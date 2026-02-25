import shutil
import tarfile
import zipfile
from pathlib import Path
from zipfile import ZIP_DEFLATED

import pandas as pd

DATAPATH = Path(__file__).parents[1]

data = pd.read_excel(DATAPATH / "raw.xlsx", index_col="ID")
test = pd.read_csv(DATAPATH / "test_idx.csv", index_col="ID")
train = pd.read_csv(DATAPATH / "train_idx.csv", index_col="ID")

assert set(test.index).union(set(train.index)) == set(data.index)

train_data = data[data.index.isin(train.index)]
test_data = data[data.index.isin(test.index)]

download_path = DATAPATH / "downloads"
download_path.mkdir(exist_ok=True)

# Data files in different formats
shutil.copy(DATAPATH / "raw.xlsx", download_path / "all.xlsx")
data.to_csv(download_path / "all.csv")

train_data.to_csv(download_path / "train.csv")
test_data.to_csv(download_path / "test.csv")

train_data.to_excel(download_path / "train.xlsx")
test_data.to_excel(download_path / "test.xlsx")

# Copy and compress CIF files

with tarfile.open(download_path / "all_cifs.tar.gz", "w:gz") as tar:
    tar.add(DATAPATH / "randomized_cifs", arcname="all_randomized_cifs")

all_zip = zipfile.ZipFile(download_path / "all_cifs.zip", "w", compression=ZIP_DEFLATED)

train_tar = tarfile.open(download_path / "train_cifs.tar.gz", "w:gz")
test_tar = tarfile.open(download_path / "test_cifs.tar.gz", "w:gz")

train_zip = zipfile.ZipFile(
    download_path / "train_cifs.zip", "w", compression=ZIP_DEFLATED
)
test_zip = zipfile.ZipFile(
    download_path / "test_cifs.zip", "w", compression=ZIP_DEFLATED
)

for idx in data.index:
    name = (DATAPATH / "randomized_cifs" / idx).with_suffix(".cif")
    if name.exists():
        all_zip.write(name, arcname="all_randomized_cifs/" + name.name)
        if idx in train.index:
            train_tar.add(name, arcname="train_randomized_cifs/" + name.name)
            train_zip.write(name, arcname="train_randomized_cifs/" + name.name)
        elif idx in test.index:
            test_tar.add(name, arcname="test_randomized_cifs/" + name.name)
            test_zip.write(name, arcname="test_randomized_cifs/" + name.name)

all_zip.close()
train_tar.close()
test_tar.close()
train_zip.close()
test_zip.close()
