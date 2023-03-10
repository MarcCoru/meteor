from glob import glob
import rasterio
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
import torch
np.random.seed(0)
torch.manual_seed(0)
import pandas as pd
import os
IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

s1bands = ["S1VV", "S1VH"]
s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11", "S2B12"]
bands = s1bands + s2bands

classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban", "snow", "barren", "water"])

regions = [('KippaRing', 'winter'),
           ('MexicoCity', 'winter'),
           ('CapeTown', 'autumn'),
           ('BandarAnzali', 'autumn'),
           ('Mumbai', 'autumn'),
           ('BlackForest', 'spring'),
           ('Chabarovsk', 'summer')]


IGBP_simplified_class_mapping = [
    0,  # Evergreen Needleleaf Forests
    0,  # Evergreen Broadleaf Forests
    0,  # Deciduous Needleleaf Forests
    0,  # Deciduous Broadleaf Forests
    0,  # Mixed Forests
    1,  # Closed (Dense) Shrublands
    1,  # Open (Sparse) Shrublands
    2,  # Woody Savannas
    2,  # Savannas
    3,  # Grasslands
    4,  # Permanent Wetlands
    5,  # Croplands
    6,  # Urban and Built-Up Lands
    5,  # Cropland Natural Vegetation Mosaics
    7,  # Permanent Snow and Ice
    8,  # Barren
    9,  # Water Bodies
]

class DFCDataset(Dataset):
    def __init__(self, dfcpath, region, transform, verbose=False):
        super(DFCDataset, self).__init__()
        self.dfcpath = dfcpath
        indexfile = os.path.join(dfcpath, "index.csv")
        self.transform = transform

        if os.path.exists(indexfile):
            if verbose:
                print(f"loading {indexfile}")
            index = pd.read_csv(indexfile)
        else:

            tifs = glob(os.path.join(dfcpath, "*/dfc_*/*.tif"))
            assert len(tifs) > 1

            index_dict = []
            for t in tqdm(tifs):
                basename = os.path.basename(t)
                path = t.replace(dfcpath, "")

                # remove leading slash if exists
                path = path[1:] if path.startswith("/") else path

                seed, season, type, region, tile = basename.split("_")

                with rasterio.open(os.path.join(dfcpath, path), "r") as src:
                    arr = src.read()

                classes, counts = np.unique(arr, return_counts=True)

                maxclass = classes[counts.argmax()]

                N_pix = len(arr.reshape(-1))
                counts_ratio = counts / N_pix

                # multiclass labelled with at least 10% of occurance following Schmitt and Wu. 2021
                multi_class = classes[counts_ratio > 0.1]
                multi_class_fractions = counts_ratio[counts_ratio > 0.1]

                s1path = os.path.join(f"{seed}_{season}", f"s1_{region}", basename.replace("dfc", "s1"))
                assert os.path.exists(os.path.join(dfcpath, s1path))

                s2path = os.path.join(f"{seed}_{season}", f"s2_{region}", basename.replace("dfc", "s2"))
                assert os.path.exists(os.path.join(dfcpath, s2path))

                lcpath = os.path.join(f"{seed}_{season}", f"lc_{region}", basename.replace("dfc", "lc"))
                assert os.path.exists(os.path.join(dfcpath, lcpath))

                index_dict.append(
                    dict(
                        basename=basename,
                        dfcpath=path,
                        seed=seed,
                        season=season,
                        region=region,
                        tile=tile,
                        maxclass=maxclass,
                        multi_class=multi_class,
                        multi_class_fractions=multi_class_fractions,
                        s1path=s1path,
                        s2path=s2path,
                        lcpath=lcpath
                    )
                )
            index = pd.DataFrame(index_dict)
            print(f"saving {indexfile}")
            index.to_csv(indexfile)

        index = index.reset_index()
        self.index = index.set_index(["region", "season"])
        self.index = self.index.sort_index()
        self.index = self.index.loc[region]
        self.region_seasons = self.index.index.unique().tolist()



    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index.loc[self.index["index"] == item].iloc[0]

        with rasterio.open(os.path.join(self.dfcpath, row.s1path), "r") as src:
            s1 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.s2path), "r") as src:
            s2 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.lcpath), "r") as src:
            lc = src.read(1)

        input, target = self.transform(s1, s2, lc)

        return input.float(), target



def transform(s1, s2, label):
    s2 = s2 * 1e-4

    s1 = s1 * 1e-2
    input = s2 # np.vstack([s1, s2])

    igbp_label = np.bincount(label.reshape(-1)).argmax() - 1
    target = IGBP_simplified_class_mapping[igbp_label]

    if np.isnan(input).any():
        input = np.nan_to_num(input)
    assert not np.isnan(target).any()

    return torch.from_numpy(input), target

def split_support_query(ds, shots, random_state=0, at_least_n_queries=0):

    classes, counts = np.unique(ds.index.maxclass, return_counts=True)
    classes = classes[counts > (shots + at_least_n_queries)] # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index.maxclass == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)

    supports = pd.concat(supports)
    support_data = [ds[idx] for idx in supports["index"].to_list()]
    support_input, _ = list(zip(*support_data))
    support_dfc_labels = supports.maxclass.values

    support_input = torch.stack(support_input)
    support_target = torch.from_numpy(support_dfc_labels)

    # query
    queries = pd.concat(queries)
    query_data = [ds[idx] for idx in queries["index"].to_list()]
    query_input, _ = list(zip(*query_data))
    query_input = torch.stack(query_input)
    query_target = torch.from_numpy(queries.maxclass.values)

    return support_input, support_target, query_input, query_target, supports, queries

def get_data(datapath, shots=5, region=regions[0], random_state=0, return_info=False):
    ds = DFCDataset(datapath, region, transform)

    support_input, support_target, query_input, query_target, support, query = split_support_query(ds, shots=shots, random_state=random_state)
    support_target = support_target - 1
    query_target = query_target - 1

    classids = np.unique(support_target)
    present_classes = classnames[classids]
    support_target = torch.tensor([list(present_classes).index(classnames[t]) for t in support_target])
    query_target = torch.tensor([list(present_classes).index(classnames[t]) for t in query_target])

    ids, counts = np.unique(ds.index.maxclass, return_counts=True)
    names = [classnames[i-1] for i in ids]
    dataset_stats = dict(
        support=len(support_target),
        query=len(query_target),
        classes=len(np.unique(query_target)),
        total=len(ds),
        composition=", ".join([f"{n} ({c})" for n,c in zip(names, counts)])
    )

    if return_info:
        return support_input, support_target, query_input, query_target, present_classes, s2bands, dataset_stats
    else:
        return support_input, support_target, query_input, query_target, present_classes, s2bands
