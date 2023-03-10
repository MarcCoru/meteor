import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
import rasterio as rio

s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11", "S2B12"]
bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def add_b1b9b10_zerobands(arr):
    _, H, W = arr.shape
    img = np.zeros((13, H, W))
    img[1:9] = arr[:8]
    img[-2:] = arr[8:]
    return img


class Anthroprotect(Dataset):
    def __init__(self, root):
        self.root = root
        self.index = pd.read_csv(os.path.join(root, "infos.csv")).sample(1000, random_state=0)
        self.index = self.index.reset_index(drop=True).reset_index()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index.iloc[item]

        with rio.open(os.path.join(self.root, "tiles", "s2", row.file)) as src:
            arr = src.read()

        #arr = add_b1b9b10_zerobands(arr)
        arr = arr * 1e-4
        arr = torch.from_numpy(arr)

        return arr, row.label


def split_support_query(ds, shots, at_least_n_queries=0, random_state=0):
    classes, counts = np.unique(ds.index.label, return_counts=True)
    classes = classes[counts > (
                shots + at_least_n_queries)]  # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index.label == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    return supports, queries


def load_samples(ds, dataframe):
    data_input, data_target = [], []
    for idx in dataframe["index"]:
        im, label = ds[idx]
        data_input.append(im)
        data_target.append(label)

    data_input = torch.stack(data_input)
    data_target = torch.tensor(data_target)
    return data_input, data_target


def get_data(datapath, shots=5):
    ds = Anthroprotect(datapath)

    support, query = split_support_query(ds, shots=shots)

    support_input, support_target = load_samples(ds, support)
    query_input, query_target = load_samples(ds, query)
    return support_input, support_target, query_input, query_target, ["not protected", "protected"], bands
