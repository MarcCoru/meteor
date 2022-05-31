import torch

# make relative imports possible by adding this folder to system path (may be a bit too hacky for confusing imports)
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import get_marinedebris_accra, get_bandaranzali
from meteor import METEOR
from model import get_model

inner_step_size = 0.32
gradient_steps = 3
batch_size = 2

marinedebris_accra = torch.from_numpy(get_marinedebris_accra())
bandar_anzali = torch.from_numpy(get_bandaranzali())

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model()
bag = METEOR(model, first_order=True, gradient_steps=gradient_steps, inner_step_size=inner_step_size,
             verbose=True, device=device, batch_size=batch_size)

def generate_links(nodes, associations):

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

def test():

    # demo data
    nodes = {'nodes': [
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=18', 'id': '18', 'index': 0,
         'x': 603.2546083152496, 'y': 250.7825262396727, 'vy': -8.836018935584013e-55, 'vx': 1.9440098162418472e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=1', 'id': '1', 'index': 1,
         'x': 433.1682177704992, 'y': 345.3225221513641, 'vy': 4.304779249636019e-55, 'vx': -4.534490042767877e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=19', 'id': '19', 'index': 2,
         'x': 500.3922732282932, 'y': 232.55103984561845, 'vy': -5.406860897433229e-55, 'vx': -2.9728617550117645e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=8', 'id': '8', 'index': 3,
         'x': 505.69633701946356, 'y': 420.8099772152952, 'vy': 1.6407685270088875e-54, 'vx': 1.673808585730683e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=10', 'id': '10', 'index': 4,
         'x': 359.6467367255987, 'y': 274.58038676486325, 'vy': -4.699371058777796e-55, 'vx': -1.3506559162168613e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=17', 'id': '17', 'index': 5,
         'x': 542.9223482362737, 'y': 127.23430409966366, 'vy': -1.572391696085254e-54, 'vx': 4.6394717351640384e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=6', 'id': '6', 'index': 6,
         'x': 460.59843660309417, 'y': 512.7102678559222, 'vy': 2.6617891207826115e-54, 'vx': -4.488936566614337e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=13', 'id': '13', 'index': 7,
         'x': 408.70066438509934, 'y': 183.63491650856915, 'vy': -1.3360584686153183e-54, 'vx': -9.661974565112163e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=4', 'id': '4', 'index': 8,
         'x': 534.8695119673354, 'y': 325.7948278948853, 'vy': 4.716174749690925e-55, 'vx': 6.304619591368713e-56},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=2', 'id': '2', 'index': 9,
         'x': 322.28100893205567, 'y': 375.3199118821627, 'vy': 6.735450367277005e-55, 'vx': -1.4440369102627462e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=5', 'id': '5', 'index': 10,
         'x': 653.215693794271, 'y': 165.78058808109068, 'vy': -2.1667972127580277e-54, 'vx': 2.4720791943617543e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=14', 'id': '14', 'index': 11,
         'x': 584.4076470467384, 'y': 487.9754323328646, 'vy': 1.5885011811467512e-54, 'vx': 1.1860126415421888e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=9', 'id': '9', 'index': 12,
         'x': 315.2063465544728, 'y': 155.3398687927675, 'vy': -1.1736672920081729e-54, 'vx': -1.4563147939500332e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=7', 'id': '7', 'index': 13,
         'x': 689.3340844421804, 'y': 297.6532730347007, 'vy': 5.5476831813965874e-55, 'vx': 2.3328357692311404e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=16', 'id': '16', 'index': 14,
         'x': 364.6635846712284, 'y': 469.213115075268, 'vy': 1.2877948132375074e-54, 'vx': -1.054596714330881e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=11', 'id': '11', 'index': 15,
         'x': 441.06410983662266, 'y': 87.18858890258916, 'vy': -1.638440050121566e-54, 'vx': -5.142152672856521e-55},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=3', 'id': '3', 'index': 16,
         'x': 659.3355796764907, 'y': 402.441950887008, 'vy': 6.39289333009416e-55, 'vx': 1.2193319331678221e-54},
        {'dataset': 'landcover', 'href': '/get_rgb_image?dataset=landcover&idx=0', 'id': '0', 'index': 17,
         'x': 261.24281079503453, 'y': 285.66650243569575, 'vy': -1.669719212173808e-55,
         'vx': -1.8629976875511286e-54}]}
    associations = {'class3': '13', 'class2': '18', 'class1': '5'}

    return generate_links(nodes, associations)

if __name__ == '__main__':
    print(test())

