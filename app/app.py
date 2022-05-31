import io
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, jsonify, request, send_file

from data import get_marinedebris_accra, get_bandaranzali
from utils import get_rgb
from gpu_module import generate_links

# load the data
num_images = 18
marinedebris_accra = torch.from_numpy(get_marinedebris_accra())
bandar_anzali = torch.from_numpy(get_bandaranzali())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', name="name")

def get_dataset_idxs(dataset):
    if dataset == "landcover":
        return np.random.choice(bandar_anzali.shape[0], size=num_images, replace=False)
    if dataset == "marinedebris":
        return np.random.choice(marinedebris_accra.shape[0], size=num_images, replace=False)


@app.route("/get_nodes")
def get_nodes():
    dataset = request.args.get('dataset')
    idxs = get_dataset_idxs(dataset)

    nodes = []
    for i, idx in zip(range(num_images), idxs):
        node = {
            "id": str(idx),
            "href": "/get_rgb_image?dataset=" + dataset + "&idx=" + str(idx),
            "dataset": dataset
        }
        nodes.append(node)

    data = {
        "nodes": nodes
    }

    return jsonify(data)


@app.route("/get_rgb_image")
def get_rgb_image():
    # e.g. http://127.0.0.1:5000/get_rgb_image?path=fall/1/Grassland/p176bl
    idx = request.args.get('idx')
    dataset = request.args.get('dataset')

    if dataset == "landcover":
        # x, y = ds[int(idx)]
        x = bandar_anzali[int(idx)]
    elif dataset == "marinedebris":
        x = marinedebris_accra[int(idx)]
    else:
        raise ValueError("invalid dataset argument!")

    rgb = get_rgb(x.numpy())

    img = Image.fromarray(rgb.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')

    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')


@app.route("/request_links", methods=['POST'])
def request_links():
    data = request.json
    nodes = data["nodes"]
    associations = data["associations"]
    data["links"] = generate_links(nodes, associations)
    return jsonify(data)

if __name__ == '__main__':
    app.run()
