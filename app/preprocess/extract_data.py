import yaml
import torch
import data
import numpy as np
import os

SHOTS = 10
APP_DATA_DIR = "/data/meteor-paper/app"

config = yaml.safe_load(open("data/config.yaml"))

datasets = ["eurosat", "anthroprotect", "dfc2020", "denethor", "floatingobjects", "nwpuresisc45"]
for name in datasets:
    ## Load Data
    dataconfig = config["data"][name]
    support_input, support_target, query_input, query_target, classes, bands = data.__dict__[name](dataconfig["datapath"],
                                                                                                   shots=SHOTS)

    os.makedirs(APP_DATA_DIR, exist_ok=True)
    filename = os.path.join(APP_DATA_DIR, name + ".npy")
    print(f"writing {filename}")
    np.save(filename, support_input.numpy())
