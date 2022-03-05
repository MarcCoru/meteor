from flask import Flask, render_template, jsonify, request, send_file
import io
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import cross_entropy

from dfcdataset import DFCDataset, get_rgb
from transforms import get_classification_transform
from resnet12 import prepare_classification_model
from bagofmaml import BagOfMAML

regions = [('KippaRing', 'winter'),
           ('MexicoCity', 'winter'),
           ('CapeTown', 'autumn'),
           ('BandarAnzali', 'autumn'),
           ('Mumbai', 'autumn'),
           ('BlackForest', 'spring'),
           ('Chabarovsk', 'summer')]

all_classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban/built-up", "snow/ice", "barren", "water"])

app = Flask(__name__)

#model = prepare_classification_model(nclasses=4)
#model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
inner_step_size = 0.32

transform = get_classification_transform(s2only=True)
ds = DFCDataset(dfcpath="/data/sen12ms/DFC_Public_Dataset", region=regions[3], transform=transform)
device = "cuda"
N = 18

model = prepare_classification_model(1, inplanes=13, resnet=True, norm="layernorm")
model.load_state_dict(torch.load("model/model_best.pth"))

bag = BagOfMAML(model, 1, first_order=True, verbose=True, device=device, batch_size=2)

@app.route('/')
def index():
    return render_template('index.html', name="name")

@app.route("/get_nodes")
def get_nodes():

    idxs = ds.index["index"].sample(N).values

    nodes = []
    for i, idx in zip(range(N), idxs):
        row = ds.index.loc[ds.index["index"] == idx]
        node = {
            "id": str(idx),
            "classname": str(all_classnames[row.maxclass-1]),
            "href":"/get_rgb_image?idx="+str(idx)
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

    x, y = ds[int(idx)]

    rgb = get_rgb(x.numpy())

    img = Image.fromarray(rgb.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')

    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')

@app.route("/request_links", methods = ['POST'])
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
        x, y = ds[idx]
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
            if support_id != query_id:
                if pred == i:
                    v = float(y_score.cpu()[i,j])
                    links.append(
                        {
                            "source": support_id,
                            "target": query_id,
                            "value": v
                        }
                    )

    """
    links = []
    for prediction, target in zip(test_prediction, query_ids):
        source = support_ids[prediction]
        links.append(
            {
                "source": source,
                "target": target,
                "value": 1
            }
        )
    """
    return links

if __name__ == '__main__':
    app.run()
