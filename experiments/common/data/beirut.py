import os
import rasterio as rio

import numpy as np
import datetime
import torch
import urllib

bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL", "TCI_R",
         "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA10", "QA20", "QA60"]


def get_data_raw():
    folder = "/data/meteor-paper/data/beirut"
    files = [f for f in os.listdir(folder) if f.endswith(".tif")]

    idxs = np.array([bands.index(b) for b in ["TCI_R", "TCI_G", "TCI_B"]])
    b_idx = np.array([bands.index(b) for b in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]])

    cld_prob = []
    for file in files:
        with rio.open(os.path.join(folder, file)) as src:
            arr = src.read()

        cld_prob.append(arr[bands.index("MSK_CLDPRB")].mean() / 100)
    cld_prob = np.array(cld_prob)

    arrays, dates = [], []
    for file in np.array(files)[cld_prob < 0.1]:
        with rio.open(os.path.join(folder, file)) as src:
            arrays.append(src.read()[b_idx])
            geotransform = src.transform
            crs = src.crs
        dates.append(file[:8])
    timeseries = np.array(arrays)


    def add_zeroband_b10(timeseries):
        b1b9 = timeseries[:, :10]
        b11b12 = timeseries[:, -2:]

        T, D, H, W = timeseries.shape
        s2images = np.zeros((T, 13, H, W))
        s2images[:, :10] = b1b9
        s2images[:, -2:] = b11b12
        return s2images


    timeseries = add_zeroband_b10(timeseries)
    timeseries = np.nan_to_num(timeseries)
    timeseries *= 1e-4

    timeseries = torch.from_numpy(timeseries).float()

    timeseries = timeseries[0:70]
    dates = dates[0:70]

    dates_dt = [datetime.datetime.strptime(d, "%Y%m%d") for d in dates]

    np.savez("/data/bagofmaml/examples/beirut/data.npz",
             timeseries=timeseries.numpy(),
             dates=np.array(dates),
             ids=np.array(files)[:70],
             crs=str(crs),
             geotransform=np.array(geotransform)
             )

def get_data():

    url = "https://bagofmaml.s3.eu-central-1.amazonaws.com/examples/beirut/data.npz"

    def load_datafile(filename):
        f = np.load(filename)
        timeseries = torch.tensor(f["timeseries"])
        dates = f["dates"]
        crs = f["crs"]
        ids = f["ids"]
        geotransform = f["geotransform"]
        return timeseries, dates, crs, ids, geotransform


    if not os.path.exists("data.npz"):
        urllib.request.urlretrieve(url, "data.npz")

    timeseries, dates, crs, ids, geotransform = load_datafile("data.npz")

    dates_dt = [datetime.datetime.strptime(d, "%Y%m%d") for d in dates]
    return timeseries, dates_dt
