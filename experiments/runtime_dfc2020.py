import numpy as np
import torch
import yaml
import os
from sklearn.metrics import classification_report

from common.dfc2020.data import regions, bands
from common.model import get_model
from common.aggregate_results import save_results
from common.data.dfc2020 import get_data
import time
import pandas as pd


CONFIG_YAML = "/home/marc/projects/meteor/experiments/config.yaml"

compare_models = ["meteor", "MOSAIKS-localfeatures", "SSLTransformerRS-resnet50", "swav", "dino", "seco",
                  "imagenet", "proto", "scratch", "ssl4eo-mocorn50", "baseline-resnet18"]


def aggregate_results():
    from glob import glob

    config = yaml.safe_load(open(CONFIG_YAML))
    outputfolder = config['results']['runtimeresultsfolder']

    times_files = glob(f"{outputfolder}/*/*/*/*/times.csv")

    stats = []
    for times_file in times_files:
        df = pd.read_csv(times_file)

        stats.append(dict(
                model = times_file.split("/")[-2],
                seed = times_file.split("/")[-3],
                region = times_file.split("/")[-4],
                shots = times_file.split("/")[-5],
                fit_timedelta = df.fit_timedelta.values[0],
                predict_timedelta = df.predict_timedelta.values[0],
                total_timedelta = df.total_timedelta.values[0]
            )
        )

    aggregated_df = pd.DataFrame(stats)
    csvfile = os.path.join(outputfolder, "runtimes.csv")
    print(f"writing {csvfile}")
    aggregated_df.to_csv(csvfile)

def fit():
    config = yaml.safe_load(open(CONFIG_YAML))
    dfc_path = config['data']['dfc2020']['datapath']
    outputfolder = config['results']['runtimeresultsfolder']
    os.makedirs(outputfolder, exist_ok=True)

    random_states = [0]
    num_shots = [1, 2, 5, 10, 15]

    for shots in num_shots:
        for region in regions:
            for random_state in random_states:

                outfolder = os.path.join(outputfolder, str(shots), "-".join(region), str(random_state))
                os.makedirs(outfolder, exist_ok=True)
                print(f'writing results in {outfolder}')

                support_input, support_target, query_input, query_target, present_classes, s2bands, dataset_stats \
                    = get_data(dfc_path, shots, region, random_state=random_state, return_info=True)
                test_models(support_input, support_target, query_input, query_target, outfolder, present_classes)


def test_models(support_input, support_target, query_input, query_target, outfolder, classes):
    config = yaml.safe_load(open(CONFIG_YAML))

    device = "cuda"

    model = get_model("meteor", snapshot_path=None, inplanes=13, select_bands=bands, device=device)

    start_time = time.time()
    model.fit(support_input.to(device), support_target)
    fit_time = time.time()
    y_pred, y_score = model.predict(query_input.to(device))
    end_time = time.time()

    times = dict(
        fit_timedelta=fit_time - start_time,
        predict_timedelta=end_time - fit_time,
        total_timedelta=end_time - start_time
    )

    os.makedirs(os.path.join(outfolder, "meteor"), exist_ok=True)
    save_results(os.path.join(outfolder, "meteor"), y_pred, query_target, y_score, classes)
    pd.DataFrame([times]).to_csv(os.path.join(outfolder, "meteor", "times.csv"))
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"METEOR-CPU")
    device = "cpu"

    model = get_model("meteor", snapshot_path=None, inplanes=13, select_bands=bands, device=device)
    support_input = support_input.to(device)
    query_input = query_input.to(device)

    start_time = time.time()
    model.fit(support_input, support_target)
    fit_time = time.time()
    y_pred, y_score = model.predict(query_input)
    end_time = time.time()

    times = dict(
        fit_timedelta = fit_time-start_time,
        predict_timedelta = end_time-fit_time,
        total_timedelta = end_time-start_time
    )

    os.makedirs(os.path.join(outfolder, "meteor-cpu"), exist_ok=True)
    save_results(os.path.join(outfolder, "meteor-cpu"), y_pred, query_target, y_score, classes)
    pd.DataFrame([times]).to_csv(os.path.join(outfolder, "meteor-cpu", "times.csv"))

    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))


def reset_indices(targets, class_ids):
    """
    resets absolute class indices (1,7,5,3) with relative ones (0,1,2,3)
    """
    row = torch.clone(targets)
    for idx, id in enumerate(class_ids):
        row[row == id] = idx
    return row

if __name__ == '__main__':
    #fit()
    aggregate_results()
