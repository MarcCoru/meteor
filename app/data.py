import os

import numpy as np
import requests

MARINEDEBRIS_ACCRA_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/app/marinedebris_accra.npy"
BANDAR_ANZALI_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/app/BandarAnzali.npy"

root_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(root_dir, "data"), exist_ok=True)

def get_marinedebris_accra():
    npyfile = os.path.join(root_dir, "data", "marinedebris_accra.npy")
    if not os.path.exists(npyfile):
        response = requests.get(MARINEDEBRIS_ACCRA_URL)
        with open(npyfile, "wb") as dst:
            dst.write(response.content)
    return np.load(npyfile)


def get_bandaranzali():
    npyfile = os.path.join(root_dir, "data", "bandaranzali.npy")
    if not os.path.exists(npyfile):
        response = requests.get(BANDAR_ANZALI_URL)
        with open(npyfile, "wb") as dst:
            dst.write(response.content)
    return np.load(npyfile)

if __name__ == '__main__':
    get_bandaranzali()
    get_marinedebris_accra()
