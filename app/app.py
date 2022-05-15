import io

import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, jsonify, request, send_file

from meteor import METEOR
from data import get_marinedebris_accra, get_bandaranzali
from model import get_model
from utils import get_rgb

app = Flask(__name__)

inner_step_size = 0.32
gradient_steps = 3
batch_size = 8
num_images = 18

device = "cuda" if torch.cuda.is_available() else "cpu"

marinedebris_accra = torch.from_numpy(get_marinedebris_accra())
bandar_anzali = torch.from_numpy(get_bandaranzali())

model = get_model()
bag = METEOR(model, first_order=True, gradient_steps=gradient_steps, inner_step_size=inner_step_size,
             verbose=True, device=device, batch_size=batch_size)


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
    data["links"] = generate_links(data)
    return jsonify(data)


def generate_links(data):
    nodes = data["nodes"]
    associations = data["associations"]

    classvalues = [int(v) for v in associations.values()]

    X_query = []
    X_support, y_support = [], []
    for node in nodes["nodes"]:
        idx = int(node["id"])
        if node["dataset"] == "marinedebris":
            x = marinedebris_accra[idx]
        elif node["dataset"] == "landcover":
            x = bandar_anzali[idx]

        X_query.append(x)
        if idx in classvalues:
            X_support.append(x)
            y_support.append(classvalues.index(idx))

    X_query = torch.stack(X_query)
    X_support = torch.stack(X_support)
    y_support = torch.tensor(y_support)

    bag.fit(X_support.to(device), y_support)
    predictions, y_score = bag.predict(X_query.to(device))

    query_idxs = [int(n["id"]) for n in nodes["nodes"]]

    links = []
    for i, support_id in enumerate(classvalues):
        for j, (query_id, pred) in enumerate(zip(query_idxs, predictions)):
            if query_id not in classvalues:
                if pred == i:
                    links.append(
                        {
                            "source": str(support_id),
                            "target": str(query_id),
                            "value": 1
                        }
                    )
    return links


if __name__ == '__main__':
    app.run()
