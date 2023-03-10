import yaml
import urllib.request
import os
import zipfile

"""
downloads the required datasets and models to reproduce the experiments
"""

config = yaml.safe_load(open("config.yaml"))


for dataset in config["data"].keys():
    print(f"downloading {dataset}")

    dataset_config = config["data"][dataset]

    url = dataset_config["url"]
    filename = os.path.basename(url)
    data_root = os.path.split(dataset_config["datapath"])[0]
    zipfilename = os.path.join(data_root, filename)

    if not os.path.exists(dataset_config["datapath"]):
        urllib.request.urlretrieve(url, zipfilename)

        with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
            zip_ref.extractall(data_root)
    else:
        print(f"{dataset_config['datapath']} exists. skipping")

for model in config["models"].keys():
    print(f"downloading {model}")

    model_config = config["models"][model]

    for variant in model_config.keys():
        model_variant_config = model_config[variant]

        if not os.path.exists(model_variant_config["path"]):
            urllib.request.urlretrieve(model_variant_config["url"], model_variant_config["path"])
        print(f"{model_variant_config['path']} exists. skipping")
