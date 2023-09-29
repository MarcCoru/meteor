import urllib.request
import os
import numpy as np
import datetime
import torch
import sys
sys.path.append("..")

import pandas as pd
import matplotlib.pyplot as plt


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


def plot(y_score, dates_dt, savedir=None):
    plt.rc("axes.spines", top=False, right=False)

    fig, ax = plt.subplots(figsize=(8, 2))
    df = pd.DataFrame([dates_dt, y_score.T[:, 1].numpy()], index=["date", "score"]).T.set_index("date")
    # df.plot(ax=ax, legend=False)
    ax.fill_between(dates_dt, y_score.T[:, 1].numpy())
    ax.axvline(datetime.datetime(2020, 8, 4), ymin=0.35, ymax=0.75, linestyle="dotted")
    ax.text(datetime.datetime(2020, 8, 4), 1.1, "Beirut Explosion 2020-08-24", ha="center", va="center")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("score \"after\"")

    if savedir is not None:
        fig.savefig(os.path.join(savedir, "plot.pdf"), transparent=True, bbox_inches="tight", pad_inches=0)

    plt.show()

    return fig

def example():
    import torch
    from meteor import METEOR
    from meteor import models

    # download data
    timeseries, dates_dt = get_data()

    # select support images from time series (first and last <shot> images)
    shot = 3

    start = timeseries[:shot]
    end = timeseries[-shot:]
    X_support = torch.vstack([start, end])
    y_support = torch.hstack([torch.zeros(shot), torch.ones(shot)]).long()

    # get model
    # get model
    s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
               "S2B12"]
    model = models.get_model("maml_resnet12", subset_bands=s2bands)
    taskmodel = METEOR(model, verbose=True, inner_step_size=0.4, gradient_steps=20)

    # fit and predict
    taskmodel.fit(X_support, y_support)
    y_pred, y_score = taskmodel.predict(timeseries)

    # plot score
    plot(y_score, dates_dt)

if __name__ == '__main__':
    example()
    plt.show()