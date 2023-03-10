"""
from https://github.com/zhu-xlab/SSL4EO-S12/blob/1dc0c8310ca8c2bf4c6f8d67f168eab568a10a9d/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py
"""
import torch
import numpy as np

ALL_BANDS_S2_L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
ALL_BANDS_S2_L1C = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
ALL_BANDS_S1_GRD = ['VV','VH']


### band statistics: mean & std
# calculated from 50k data
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]

S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]

S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

MEAN = S2C_MEAN
STD = S2C_STD

# normalize: standardize + percentile
def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def get_norm(bands, select_bands):

    # e.g. remove S2 from "S2B1" to make band names match across different datasets
    s = [s.replace("S2","") for s in select_bands]
    all_bands = [s.replace("S2", "") for s in bands]
    band_idxs = np.array([all_bands.index(b) for b in s])

    def norm_s2maxs(arr):
        b = []
        for x in arr:
            normalized_x = np.stack([normalize(ch.numpy(), MEAN[b_idx], STD[b_idx]) for ch,b_idx in zip(x, band_idxs)])
            b.append(torch.from_numpy(normalized_x))
        return torch.stack(b).float() / 255.
    return norm_s2maxs
