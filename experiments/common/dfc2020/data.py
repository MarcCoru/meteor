import os
import tqdm
import glob
import pandas as pd
from torch.utils.data import Dataset
import rasterio
import numpy as np
import torch

IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban/built-up", "snow/ice", "barren", "water"])

bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
         "S2B12"]

regions = [('KippaRing', 'winter'),
           ('MexicoCity', 'winter'),
           ('CapeTown', 'autumn'),
           ('BandarAnzali', 'autumn'),
           ('Mumbai', 'autumn'),
           ('BlackForest', 'spring'),
           ('Chabarovsk', 'summer')]


def split_support_query(ds, shots, at_least_n_queries=1, random_state=0, classcolumn="maxclass"):
    classes, counts = np.unique(ds.index[classcolumn], return_counts=True)

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index[classcolumn] == c].reset_index()

        N_samples = shots if len(samples) > shots else len(samples) - at_least_n_queries  # keep at least one sample for query
        support = samples.sample(N_samples, random_state=random_state, replace=False)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    return supports, queries

def load_samples(ds, dataframe):
    data_input, data_target = [], []
    for idx in dataframe["index"]:
        #i = (ds.index["index"] == idx).argmax()
        im, label = ds[idx]
        data_input.append(im)
        data_target.append(label)

    data_input = torch.stack(data_input)
    data_target = torch.tensor(data_target)
    return data_input, data_target

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

        return input.float(), row.maxclass
