import torch
import rasterio as rio
import numpy as np
import geopandas as gpd
import os

from skimage.exposure import equalize_hist
import torchvision
import pandas as pd

CLASSES = ["Wheat", "Rye", "Barley", "Oats", "Corn", "Oil Seeds", "Root Crops", "Meadows", "Forage Crops"]
CROP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

labelgeojson = "dlr_fusion_competition_germany_train_labels.geojson"
tif = "sr_train_20180508.tif"

select_classes = ['Wheat', 'Meadows', 'Corn'] # 'Rye', 'Barley',

bands = ["S2B4", "S2B3", "S2B2", "S2B8"]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tif, labelgeojson, transform=None, min_area=1000):
        self.tif = tif

        self.labels = gpd.read_file(labelgeojson)

        self.data_transform = transform

        # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
        with rio.open(self.tif) as image:
            self.crs = image.crs
            self.transform = image.transform

        mask = self.labels.geometry.area > min_area
        print(f"ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area}m2")
        self.labels = self.labels.loc[mask]

        self.labels = self.labels.loc[self.labels.crop_name.isin(select_classes)]

        self.labels = self.labels.to_crs(self.crs)
        self.labels = self.labels.reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.labels.iloc[item]

        left, bottom, right, top = feature.geometry.bounds

        window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

        # reads each tif in tifs on the bounds of the feature. shape D x H x W
        tif = self.tif
        image = rio.open(tif).read(window=window)

        # get meta data from first image to get the windowed transform
        with rio.open(self.tif) as src:
            win_transform = src.window_transform(window)

        out_shape = image[0].shape
        assert out_shape[0] > 0 and out_shape[
            1] > 0, f"fid:{feature.fid} image stack shape {image.shape} is zero in one dimension"

        if self.data_transform is not None:
            image = self.data_transform(image)

        return image, CROP_IDS.index(feature.crop_id), feature.fid


resize = torchvision.transforms.Resize((128, 128))
def data_transform(image):
    rgb_idxs = np.array([2, 1, 0])
    false_color_idxs = np.array([2, 1, 0])

    #image = equalize_hist(image)
    image = torch.from_numpy(image)
    image = resize(image)
    return image


def split_support_query(ds, shots, at_least_n_queries=0, random_state=0):
    classes, counts = np.unique(ds.labels.crop_id, return_counts=True)
    classes = classes[counts > (
                shots + at_least_n_queries)]  # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.labels.loc[ds.labels.crop_id == c].reset_index()
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
        im, label, id = ds[idx]
        data_input.append(im)
        data_target.append(label)

    data_input = torch.stack(data_input)
    data_target = torch.tensor([select_classes.index(CLASSES[c]) for c in data_target])

    return data_input, data_target


def get_data(datapath, shots=5):
    root = datapath
    ds = Dataset(os.path.join(root, tif), os.path.join(root, labelgeojson), min_area=30000, transform=data_transform)

    support, query = split_support_query(ds, shots=shots)

    support_input, support_target = load_samples(ds, support)
    query_input, query_target = load_samples(ds, query)

    return support_input, support_target, query_input, query_target, select_classes, bands
