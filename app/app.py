from flask import Flask, render_template, jsonify, request, send_file
import io
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import cross_entropy

from dfcdataset import DFCDataset, get_rgb
from transforms import get_classification_transform

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

N = 18

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
            "id": str(i),
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

    def href_to_h5path(href):
        """parse /get_rgb_image?path=spring/58/Savanna/p607tr to spring/58/Savanna/p607tr"""
        return href.split("?")[-1].split("=")[-1]

    h5paths = [href_to_h5path(n["href"]) for n in nodes["nodes"]]
    ids = [n["id"] for n in nodes["nodes"]]

    batch = get_batch(h5paths)

    # split batch into support and query based on associations
    support_ids = list(associations.values())
    support_idxs =  [ids.index(support_id) for support_id in support_ids]

    query_ids = [id for id in ids if id not in support_ids]
    query_idxs = [ids.index(query_id) for query_id in query_ids]

    support_batch = batch[support_idxs]
    query_batch = batch[query_idxs]

    train_logit = model(support_batch.float())
    inner_loss = cross_entropy(train_logit, torch.arange(4))

    model.zero_grad()
    params = update_parameters(model, inner_loss,
                               inner_step_size=inner_step_size, first_order=False)

    test_logit = model(query_batch.float(), params=params)

    test_prediction = test_logit.argmax(1)

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

    return links

if __name__ == '__main__':
    app.run()
