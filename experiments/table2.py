import numpy as np
import torch
import yaml
import os
from sklearn.metrics import classification_report

from common.dfc2020.data import regions, bands
from common.model import get_model, PrototypicalWrapper
from common.aggregate_results import save_results
from common.models.mosaiks import fit_predict_mosaiks
import common.models.ssltransformerrs as ssltransformerrs
from common.data.dfc2020 import get_data

compare_models = ["meteor", "MOSAIKS-localfeatures", "SSLTransformerRS-resnet50", "swav", "dino", "seco",
                  "imagenet", "proto", "scratch", "ssl4eo-mocorn50", "baseline-resnet18"]


def main():
    config = yaml.safe_load(open("config.yaml"))
    dfc_path = config['data']['dfc2020']['datapath']
    outputfolder = config['results']['table2resultsfolder']
    os.makedirs(outputfolder, exist_ok=True)

    random_states = [0, 1, 2]
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
    config = yaml.safe_load(open("config.yaml"))

    print(f"SSL4EO - ResNet50")
    model = get_model(modelname="ssl4eo-mocorn50",
                      snapshot_path=config["models"]["ssl4eo"]["mocorn50"]["path"],
                      inplanes=13,
                      select_bands=bands)

    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "ssl4eo-mocorn50"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"SSLTransformerRS - ResNet50")
    model = get_model(modelname="ssltransformerrs-resnet50",
                      snapshot_path=config["models"]["ssltransformerrs"]["resnet50"]["path"],
                      inplanes=13,
                      select_bands=bands)

    protomodel = PrototypicalWrapper(model)
    norm = ssltransformerrs.get_norm(bands, bands)
    protomodel.fit(norm(support_input), support_target)
    y_pred, dist = protomodel.predict(norm(query_input))
    save_results(os.path.join(outfolder, "SSLTransformerRS-resnet50"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes))

    model = get_model("meteor", snapshot_path=None, inplanes=13, select_bands=bands)
    model.fit(support_input, support_target)
    y_pred, y_score = model.predict(query_input)
    save_results(os.path.join(outfolder, "meteor"), y_pred, query_target, y_score, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"MOSAIKS - local features")
    y_test, y_pred, y_score = fit_predict_mosaiks(support_input.float(), support_target, query_input.float(),
                                                  query_target)
    save_results(os.path.join(outfolder, "MOSAIKS-localfeatures"), y_pred, query_target,
                 torch.from_numpy(y_score),
                 classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"baseline - ResNet18")
    model = get_model(modelname="baseline-resnet18",
                      snapshot_path=config["models"]["baseline"]["resnet18"]["path"],
                      inplanes=13,
                      select_bands=bands)

    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "baseline-resnet18"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    # RGB only beyond this point
    if support_input.shape[1] != 3:
        support_input = support_input[:, np.array([3, 2, 1])]
        query_input = query_input[:, np.array([3, 2, 1])]

    ## Seasonal Contrast
    print(f"SeCo")
    model = get_model(modelname="seco_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=3)
    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "seco"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"Swav")
    ## DINO
    model = get_model(modelname="swav_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=3)
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "swav"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"DINO")
    ## DINO
    model = get_model(modelname="dino_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=3)
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "dino"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))


    print(f"Imagenet")
    model = get_model(modelname="imagenet_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=3)
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "imagenet"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"random")
    model = get_model(modelname="scratch_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=3)
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outfolder, "scratch"), y_pred, query_target, dist, classes)
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
    main()
