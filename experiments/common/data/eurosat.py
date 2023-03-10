import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from glob import glob
import pandas as pd
import os
import torchvision
import numpy as np
import rasterio as rio
import argparse
import json

bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
               "S2B12"]

class EuroSat(Dataset):
    def __init__(self, root):
        self.root = root
        files = glob(os.path.join(root, "ds/images/remote_sensing/otherDatasets/sentinel_2/tif", "*", "*.tif"))
        paths = [f.replace(root + "/", "") for f in files]
        images = [os.path.basename(f) for f in files]
        classname = [i.split("_")[0] for i in images]
        tileid = [i.split("_")[1].replace(".tif", "") for i in images]
        self.index = pd.DataFrame([paths, images, classname, tileid],
                                  index=["paths", "images", "classname", "tileid"]).T
        self.labels = list(self.index.classname.unique())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]
        with rio.open(os.path.join(self.root, row.paths)) as src:
            arr = src.read()
        image = arr * 1e-4
        return torch.from_numpy(image), self.labels.index(row.classname)


def split_support_query(ds, shots, at_least_n_queries=0, random_state=0):
    classes, counts = np.unique(ds.index.classname, return_counts=True)
    classes = classes[counts > (
                shots + at_least_n_queries)]  # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index.classname == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    return supports, queries

def get_data(datapath, shots=5, random_state=0):
    ds = EuroSat(datapath)
    classnames = ds.labels

    support, query = split_support_query(ds, shots, random_state)

    # dataset is very large subsample to make it runnable
    query = query.sample(1000, random_state=0)

    def load_samples(ds, dataframe):
        data_input, data_target = [], []
        for idx in dataframe["index"]:
            im, label = ds[idx]
            data_input.append(im)
            data_target.append(label)

        data_input = torch.stack(data_input)
        data_target = torch.tensor(data_target)
        return data_input, data_target

    support_input, support_target = load_samples(ds, support)
    query_input, query_target = load_samples(ds, query)

    return support_input, support_target, query_input, query_target, classnames, bands
