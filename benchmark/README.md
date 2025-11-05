# Benchmarking

This directory contains all code and configuration files necessary to reproduce the experiments presented in the paper.

To obtain the `../data/np_cifs` folder used in these scripts use

```
from pathlib import Path
from obelix import OBELiX

ob = OBELiX("../data")

output_path = "../data/np_cifs"
output_path.mkdir(exist_ok=True)

for entry in ob.round_partial().with_cifs():
    entry["structure"].to(output_path+entry["ID"]+".cif")

```

### RF and MLP
Read the comments at the end of `tuning.py` and run `python tuning.py`

### CGCNN
Go to `./CGCNN` and run `python main.py`

### M3GNet and SO3Net
Go to `./MatGL/[experiment]` and run `python ../main.py`

### PaiNN and SchNet
Go to `./SchNetPack/[experiment]` and run `python [config]`

`./SchNetPack/painn` and `./SchNetPack/schnet` sweep through hyperparameters whereas all other experiments use the same hyperparamters from those sweeps.

`./SchNetPack/[model]_rand/best_config_normal_cifs.yaml` trains and tests on original cifs whereas `./SchNetPack/[model]_rand/best_config.yaml` trains and tests on randomized cifs. 