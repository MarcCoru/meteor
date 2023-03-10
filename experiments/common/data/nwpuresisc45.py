from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np
import torch

CLASSES_NAMES = ["commercial_area", "sparse_residential", "dense_residential", "medium_residential", "industrial_area"]

bands = ["S2B2", "S2B3", "S2B4"]

class NWPURESISC45(Dataset):
    def __init__(self, root):
        super(NWPURESISC45, self).__init__()
        self.root = root
        files = os.listdir(root)
        self.index = pd.DataFrame(
            [{"file": f, "classname": "_".join(f.split("_")[:-1]), "tile": f.split("_")[-1].replace(".jpg", "")} for f
             in files])
        self.classes = list(self.index.classname.unique())
        self.index["maxclass"] = self.index.classname.apply(lambda x: self.classes.index(x))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index.iloc[item]
        im = np.array(Image.open(os.path.join(self.root, row.file)))
        im = im / 255
        im = torch.from_numpy(im)
        label = row.classname
        return im, torch.tensor(CLASSES_NAMES.index(label))


def split_support_query(ds, shots, at_least_n_queries=0, random_state=0):
    classes, counts = np.unique(ds.index.maxclass, return_counts=True)
    classes = classes[counts > (
                shots + at_least_n_queries)]  # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index.maxclass == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    return supports, queries


def get_data(datapath, shots=5, random_state=0):
    ds = NWPURESISC45(datapath)

    selected_classes = [ds.classes.index(c) for c in
                        CLASSES_NAMES]

    support, query = split_support_query(ds, shots, random_state=random_state)

    support = support.loc[support.maxclass.isin(selected_classes)]
    query = query.loc[query.maxclass.isin(selected_classes)]

    def load_samples(ds, dataframe):
        data_input, data_target = [], []
        for idx in dataframe["index"]:
            im, label = ds[idx]
            data_input.append(im)
            data_target.append(label)

        data_input = torch.stack(data_input).permute(0, 3, 1, 2)
        data_target = torch.tensor(data_target)
        return data_input, data_target

    support_input, support_target = load_samples(ds, support)
    query_input, query_target = load_samples(ds, query)
    return support_input, support_target, query_input, query_target, CLASSES_NAMES, bands
