import numpy as np
from sklearn.metrics import classification_report
import os
import yaml
import torch

import common.data as data
from common.model import get_model, PrototypicalWrapper
from common.models.mosaiks import fit_predict_mosaiks
import common.models.ssl4eo as ssl4eo
import common.models.ssltransformerrs as ssltransformerrs
from common.aggregate_results import write_and_aggregate_results, save_support_images, save_results

def main():
    shots = 5

    config = yaml.safe_load(open("config.yaml"))
    datadir = config['results']['table3resultsfolder']
    summarydir = os.path.join(datadir, "summary")
    datasets = ["eurosat", "anthroprotect", "dfc2020", "denethor", "floatingobjects", "nwpuresisc45"]

    for taskname in datasets:
        run_task(taskname, datadir, shots=shots)

    # one summary folder having all comparisons at one place (not used in the paper)
    write_and_aggregate_results(datadir, os.path.join(summarydir, "all"), datasets)

    # Table 2
    compare_models = ["meteor", "MOSAIKS-localfeatures", "SSLTransformerRS-resnet50", "swav", "dino", "seco", "imagenet",
                      "proto", "scratch", "ssl4eo-mocorn50", "baseline-resnet18"]
    write_and_aggregate_results(datadir, os.path.join(summarydir, "final"), datasets, compare_models=compare_models)

    # Appendix tables comparing METEOR to different variants
    compare_models = ["meteor", "MOSAIKS-localfeatures", "MOSAIKS-global features", "MOSAIKS-semisupervised",
                      "MOSAIKS-gaussian", "MOSAIKS-laplace"]
    write_and_aggregate_results(datadir, os.path.join(summarydir, "MOSAIKS"), datasets, compare_models=compare_models)

    compare_models = ["meteor", "SSLTransformerRS-resnet18", "SSLTransformerRS-resnet50", "SSLTransformerRS-swin"]
    write_and_aggregate_results(datadir, os.path.join(summarydir, "SSLTransformerRS"), datasets, compare_models=compare_models)

    compare_models = ["meteor", "baseline-resnet18", "baseline-resnet50", "baseline-resnet12"]
    write_and_aggregate_results(datadir, os.path.join(summarydir, "baseline"), datasets, compare_models=compare_models)

    compare_models = ["meteor", "ssl4eo-dinorn50", "ssl4eo-mocorn50"]
    write_and_aggregate_results(datadir, os.path.join(summarydir, "ssl4eo"), datasets, compare_models=compare_models)


def run_task(name, resultsdir, shots=5):
    """
    fine-tuning and prediction of every model on one task
    """

    config = yaml.safe_load(open("config.yaml"))
    outputfolder = os.path.join(resultsdir, name)

    ## Load Data
    dataconfig = config["data"][name]
    support_input, support_target, query_input, query_target, classes, bands = data.__dict__[name](dataconfig["datapath"], shots=shots)

    print(bands)
    band_idxs = np.array([bands.index(b) for b in dataconfig["rgb_bands"]])
    save_support_images(outputfolder,
                        support_input[:, band_idxs], # rgb images
                        [classes[t] for t in support_target]) #  classnames

    ## MOSAIKS
    print(f"{name} MOSAIKS - local features")
    y_test, y_pred, y_score = fit_predict_mosaiks(support_input.float(), support_target, query_input.float(), query_target)
    save_results(os.path.join(outputfolder, "MOSAIKS-localfeatures"), y_pred, query_target, torch.from_numpy(y_score),
                 classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    ## METEOR (same as Bag of MAML but newer implementation)
    model = get_model(modelname="meteor",
                      snapshot_path=None,
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])
    model.fit(support_input, support_target)
    y_pred, y_score = model.predict(query_input)
    save_results(os.path.join(outputfolder, "meteor"), y_pred, query_target, y_score, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"{name} baseline - ResNet50")
    model = get_model(modelname="baseline-resnet50",
                      snapshot_path=config["models"]["baseline"]["resnet50"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "baseline-resnet50"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes))


    print(f"{name} baseline - ResNet18")
    model = get_model(modelname="baseline-resnet18",
                      snapshot_path=config["models"]["baseline"]["resnet18"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "baseline-resnet18"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes))

    print(f"{name} baseline - ResNet12")
    model = get_model(modelname="baseline-resnet12",
                      snapshot_path=config["models"]["baseline"]["resnet12"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "baseline-resnet12"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"{name} SSL4EO - dinorn50")
    model = get_model(modelname="ssl4eo-dinorn50",
                      snapshot_path=config["models"]["ssl4eo"]["dinorn50"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])
    protomodel = PrototypicalWrapper(model)
    norm = ssl4eo.get_norm(bands, dataconfig["select_bands"])
    protomodel.fit(norm(support_input), support_target)
    y_pred, dist = protomodel.predict(norm(query_input))
    save_results(os.path.join(outputfolder, "ssl4eo-dinorn50"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))


    print(f"{name} SSL4EO - ResNet50")
    model = get_model(modelname="ssl4eo-mocorn50",
                      snapshot_path=config["models"]["ssl4eo"]["mocorn50"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    norm = ssl4eo.get_norm(bands, dataconfig["select_bands"])
    protomodel.fit(norm(support_input), support_target)
    y_pred, dist = protomodel.predict(norm(query_input))
    save_results(os.path.join(outputfolder, "ssl4eo-mocorn50"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    ## MOSAIKS
    print(f"{name} MOSAIKS - laplace")
    y_test, y_pred, y_score = fit_predict_mosaiks(support_input.float(), support_target, query_input.float(), query_target, patch_distribution="laplace")
    save_results(os.path.join(outputfolder, "MOSAIKS-laplace"), y_pred, query_target, torch.from_numpy(y_score),
                 classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    ## MOSAIKS
    print(f"{name} MOSAIKS - gaussian")
    y_test, y_pred, y_score = fit_predict_mosaiks(support_input.float(), support_target, query_input.float(), query_target, patch_distribution="gaussian")
    save_results(os.path.join(outputfolder, "MOSAIKS-gaussian"), y_pred, query_target, torch.from_numpy(y_score),
                 classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    ## MOSAIKS
    print(f"{name} MOSAIKS - global features")
    y_test, y_pred, y_score = fit_predict_mosaiks(support_input.float(), support_target, query_input.float(), query_target,
                                                  semi_supervised=False, local_features=False, select_bands=dataconfig["select_bands"])
    save_results(os.path.join(outputfolder, "MOSAIKS-global features"), y_pred, query_target, torch.from_numpy(y_score),
                 classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    ## MOSAIKS
    print(f"{name} MOSAIKS - semi supervised")
    y_test, y_pred, y_score = fit_predict_mosaiks(support_input.float(), support_target, query_input.float(), query_target, semi_supervised=True)
    save_results(os.path.join(outputfolder, "MOSAIKS-semisupervised"), y_pred, query_target, torch.from_numpy(y_score),
                 classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    ## SSLTransformerRS - Swin-T
    print(f"{name} SSLTransformerRS - Swin-T")
    model = get_model(modelname="ssltransformerrs-swin",
                      snapshot_path=config["models"]["ssltransformerrs"]["swin"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    norm = ssltransformerrs.get_norm(bands, dataconfig["select_bands"])
    protomodel.fit(norm(support_input), support_target)
    y_pred, dist = protomodel.predict(norm(query_input))
    save_results(os.path.join(outputfolder, "SSLTransformerRS-swin"), y_pred.cpu(), query_target.cpu(), dist.cpu(),
                 classes)
    print(classification_report(y_pred=y_pred.cpu(), y_true=query_target.cpu(), target_names=classes))

    print(f"{name} SSLTransformerRS - ResNet50")
    model = get_model(modelname="ssltransformerrs-resnet50",
                      snapshot_path=config["models"]["ssltransformerrs"]["resnet50"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    protomodel.fit(norm(support_input), support_target)
    norm = ssltransformerrs.get_norm(bands, dataconfig["select_bands"])
    y_pred, dist = protomodel.predict(norm(query_input))
    save_results(os.path.join(outputfolder, "SSLTransformerRS-resnet50"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes))

    print(f"{name} SSLTransformerRS - ResNet18")
    model = get_model(modelname="ssltransformerrs-resnet18",
                      snapshot_path=config["models"]["ssltransformerrs"]["resnet18"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])

    protomodel = PrototypicalWrapper(model)
    norm = ssltransformerrs.get_norm(bands, dataconfig["select_bands"])
    protomodel.fit(norm(support_input), support_target)
    y_pred, dist = protomodel.predict(norm(query_input))
    save_results(os.path.join(outputfolder, "SSLTransformerRS-resnet18"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes))

    ## Prototypical Network
    print(f"{name} Proto")
    model = get_model(modelname="proto",
                      snapshot_path=config["models"]["proto"]["model"]["path"],
                      inplanes=dataconfig["inplanes"],
                      select_bands=dataconfig["select_bands"])
    protomodel = PrototypicalWrapper(model)
    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "proto"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred = y_pred, y_true=query_target, target_names=classes))

    # RGB only beyond this point
    if support_input.shape[1] != 3:
        support_input = support_input[:, band_idxs]
        query_input = query_input[:, band_idxs]

    ## Seasonal Contrast
    model = get_model(modelname="seco_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=dataconfig["inplanes"])
    protomodel = PrototypicalWrapper(model)
    print(f"{name} SeCo")

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "seco"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"{name} DINO")
    ## DINO
    model = get_model(modelname="dino_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=dataconfig["inplanes"])
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "dino"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"{name} Swav")
    ## DINO
    model = get_model(modelname="swav_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=dataconfig["inplanes"])
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "swav"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))


    print(f"{name} Imagenet")
    model = get_model(modelname="imagenet_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=dataconfig["inplanes"])
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "imagenet"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

    print(f"{name} random")
    model = get_model(modelname="scratch_resnet50",
                      snapshot_path=config["models"]["seco_resnet50"]["rgbonly"]["path"],
                      inplanes=dataconfig["inplanes"])
    protomodel = PrototypicalWrapper(model)

    protomodel.fit(support_input, support_target)
    y_pred, dist = protomodel.predict(query_input)
    save_results(os.path.join(outputfolder, "scratch"), y_pred, query_target, dist, classes)
    print(classification_report(y_pred=y_pred, y_true=query_target, target_names=classes))

if __name__ == '__main__':
    main()
