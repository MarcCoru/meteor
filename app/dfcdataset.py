from glob import glob
import rasterio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pandas as pd
import os

np.random.seed(0)
torch.manual_seed(0)

IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban/built-up", "snow/ice", "barren", "water"])

regions = [('KippaRing', 'winter'),
           ('MexicoCity', 'winter'),
           ('CapeTown', 'autumn'),
           ('BandarAnzali', 'autumn'),
           ('Mumbai', 'autumn'),
           ('BlackForest', 'spring'),
           ('Chabarovsk', 'summer')]

import numpy as np
from skimage import exposure

def get_rgb(s2):

    s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
               "S2B12"]

    rgb_idx = [s2bands.index(b) for b in np.array(['S2B4', 'S2B3', 'S2B2'])]

    X = np.clip(s2, a_min=0, a_max=1)

    rgb = np.swapaxes(X[rgb_idx, :, :], 0, 2)
    # rgb = exposure.rescale_intensity(rgb)
    rgb = exposure.equalize_hist(rgb)
    #rgb = exposure.equalize_adapthist(rgb, clip_limit=0.1)
    # rgb = exposure.adjust_gamma(rgb, gamma=0.8, gain=1)

    rgb *= 255

    return rgb

class DFCDataset(Dataset):
    def __init__(self, dfcpath, region, transform):
        super(DFCDataset, self).__init__()
        self.dfcpath = dfcpath
        indexfile = os.path.join(dfcpath, "index.csv")
        self.transform = transform

        if os.path.exists(indexfile):
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
