from glob import glob
import rasterio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

np.random.seed(0)
torch.manual_seed(0)

import argparse

from meteor import Meteor
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import geopandas as gpd
import rasterio as rio
from train.utils.utils import prepare_transform_and_model
import sklearn.metrics

IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

classnames = np.array(["forest", "shrubland", "savanna", "grassland", "wetland", "cropland", "urban/built-up", "snow/ice", "barren", "water"])

regions = [('KippaRing', 'winter'),
           ('MexicoCity', 'winter'),
           ('CapeTown', 'autumn'),
           ('BandarAnzali', 'autumn'),
           ('Mumbai', 'autumn'),
           ('BlackForest', 'spring'),
           ('Chabarovsk', 'summer')]

def reset_indices(targets):
    """
    resets absolute class indices (1,7,5,3) with relative ones (0,1,2,3)
    """
    class_ids = targets.unique()

    row = torch.clone(targets)
    for idx, id in enumerate(class_ids):
        row[row == id] = idx
    return row

def main(args):
    folder = f"dfc2020_{args.num_shots}shots"
    if args.ensemble:
        folder += "_ensemble"
    results_dir = os.path.join(args.output_folder, folder)
    os.makedirs(results_dir, exist_ok=True)

    args.one_vs_all = True
    args.resnet = True
    transform, model, mask, mask_optimizer = prepare_transform_and_model(args)

    summary_writer = SummaryWriter(
        log_dir=os.path.join(results_dir, "tensorboard"))

    model.load_state_dict(torch.load(os.path.join(args.output_folder, "model_best.pth")))
    model.to(args.device)
    accuracies = []
    confusion_matrices = []
    for region in regions:

        print(f"preparing data of region {region}")
        ds = DFCDataset(dfcpath=args.dfc_path, region=region, transform=transform)
        accuracy, query_target, query_predictions, support, query = evaluate(model, ds, args.num_shots, args.device, mask=mask, verbose=True,
                                                             first_order=args.first_order, gradient_steps=args.gradient_steps,
                                                                             inner_step_size=args.inner_step_size,
                                                                             ensemble=args.ensemble, batch_size=args.batch_size)
        support.to_csv(os.path.join(results_dir, f"{region[0]}_{region[1]}_support.csv"))
        query.to_csv(os.path.join(results_dir, f"{region[0]}_{region[1]}_query.csv"))
        support.to_file(os.path.join(results_dir, f"{region[0]}_{region[1]}_support.geojson"), driver="GeoJSON")
        query.to_file(os.path.join(results_dir, f"{region[0]}_{region[1]}_query.geojson"), driver="GeoJSON")
        accuracies.append(accuracy)
        np.savez(os.path.join(results_dir, f"predictions_{region[0]}_{region[1]}.npz"),
                 targets=query_target,
                 predictions=query_predictions)

        labels = classnames[query_target.unique() - 1]

        print("report")
        print(sklearn.metrics.classification_report(query_target, query_predictions, labels=query_target.unique(), target_names=labels))
        report = sklearn.metrics.classification_report(query_target, query_predictions, labels=query_target.unique(), target_names=labels, output_dict=True)
        pd.DataFrame(report).T.to_csv(os.path.join(results_dir, f"report_{region[0]}_{region[1]}.csv"))

        print("ACCURACY")
        print(f"{region}: accuracy {accuracy}")

        cm = sklearn.metrics.confusion_matrix(query_target, query_predictions)

        df_cm = pd.DataFrame(cm, index=labels,
                             columns=labels)

        confusion_matrices.append(df_cm)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(accuracies)), height=accuracies)
    ax.set_xticks(np.arange(len(accuracies)))
    ax.set_xticklabels([r for r, _ in regions], rotation=45)
    ax.set_ylabel("accuracy")
    plt.tight_layout()
    summary_writer.add_figure("DFC 2020", figure=fig)

    fig.savefig(os.path.join(results_dir, "dfc2020_accuracies.png"))


    for (region, season), df_cm in zip(regions, confusion_matrices):

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, ax=ax)
        cmfilename = os.path.join(results_dir, f"dfc2020_confusion_{region}-{season}.png")
        summary_writer.add_figure(figure=fig, tag=f"{region}-{season}")
        fig.savefig(cmfilename)
        print(f"writing {cmfilename}")

def evaluate(model, ds, shots, device, verbose=False, first_order=False, gradient_steps=1, inner_step_size=0.32, ensemble=False, batch_size=4, mask=None):

    meteor = Meteor(model, inner_step_size=inner_step_size, gradient_steps=gradient_steps, device=device, verbose=verbose,
                                first_order=first_order, mask=mask)

    support_input, support_target, query_input, query_target, support, query = split_support_query(ds, shots)

    print(f"{shots}-shots specified. Label fraction {len(support_target) / (len(query_target) + len(support_target))*100:.2f}%")

    support_input = support_input#[:, :, 64:-64, 64:-64]
    query_input = query_input#[:, :, 64:-64, 64:-64]

    meteor.fit(support_input, support_target)

    query_predictions, query_probas = meteor.predict(query_input, batch_size=batch_size)
    accuracy = (query_target == torch.tensor(query_predictions)).float().mean()

    query["predictions"] = query_predictions
    query["targets"] = query_target

    # ensemble
    if len(query_probas.shape) == 3:
        num_classes, num_members, samples = query_probas.shape
        for c, name in zip(range(num_classes), classnames[query_target.unique() - 1]):
           for member in range(num_members):
               query[f"proba_{name}_{member}"] = query_probas[c, member]
    else:
        num_classes, samples = query_probas.shape
        for c, name in zip(range(num_classes), classnames[query_target.unique() - 1]):
            query[f"proba_{name}"] = query_probas[c]

    # add coordinates
    support = add_geometry(support, ds.dfcpath)
    query = add_geometry(query, ds.dfcpath)

    return accuracy, query_target, query_predictions, support, query

def add_geometry(df, root_dir):
    x, y = [], []
    for f in df.s2path:
        with rio.open(os.path.join(root_dir, f), "r") as src:
            meta = src.meta
            x.append(meta["transform"].c + 10 * 128)
            y.append(meta["transform"].f + 10 * 128)

    df["x"] = x
    df["y"] = y

    geometry = gpd.points_from_xy(df['x'], df['y']).buffer(1280, cap_style=3)
    gdf = gpd.GeoDataFrame(
        df, geometry=geometry, crs=meta["crs"])
    return gdf

def split_support_query(ds, shots, random_state=0, at_least_n_queries=0):

    classes, counts = np.unique(ds.index.maxclass, return_counts=True)
    classes = classes[counts > (shots + at_least_n_queries)] # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index.maxclass == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)

    supports = pd.concat(supports)
    support_data = [ds[idx] for idx in supports["index"].to_list()]
    support_input, _ = list(zip(*support_data))
    support_dfc_labels = supports.maxclass.values

    support_input = torch.stack(support_input)
    support_target = torch.from_numpy(support_dfc_labels)

    # query
    queries = pd.concat(queries)
    query_data = [ds[idx] for idx in queries["index"].to_list()]
    query_input, _ = list(zip(*query_data))
    query_input = torch.stack(query_input)
    query_target = torch.from_numpy(queries.maxclass.values)

    return support_input, support_target, query_input, query_target, supports, queries

class DFCDataset(Dataset):
    def __init__(self, dfcpath, region, transform, verbose=False):
        super(DFCDataset, self).__init__()
        self.dfcpath = dfcpath
        indexfile = os.path.join(dfcpath, "index.csv")
        self.transform = transform

        if os.path.exists(indexfile):
            if verbose:
                print(f"loading {indexfile}")
            index = pd.read_csv(indexfile)
        else:

            tifs = glob(os.path.join(dfcpath, "*/dfc_*/*.tif"))
            assert len(tifs) > 1

            index_dict = []
            for t in tqdm(tifs):
                basename = os.path.basename(t)
                path = t.replace(dfcpath, "")

                # remove leading slash if exists
                path = path[1:] if path.startswith("/") else path

                seed, season, type, region, tile = basename.split("_")

                with rasterio.open(os.path.join(dfcpath, path), "r") as src:
                    arr = src.read()

                classes, counts = np.unique(arr, return_counts=True)

                maxclass = classes[counts.argmax()]

                N_pix = len(arr.reshape(-1))
                counts_ratio = counts / N_pix

                # multiclass labelled with at least 10% of occurance following Schmitt and Wu. 2021
                multi_class = classes[counts_ratio > 0.1]
                multi_class_fractions = counts_ratio[counts_ratio > 0.1]

                s1path = os.path.join(f"{seed}_{season}", f"s1_{region}", basename.replace("dfc", "s1"))
                assert os.path.exists(os.path.join(dfcpath, s1path))

                s2path = os.path.join(f"{seed}_{season}", f"s2_{region}", basename.replace("dfc", "s2"))
                assert os.path.exists(os.path.join(dfcpath, s2path))

                lcpath = os.path.join(f"{seed}_{season}", f"lc_{region}", basename.replace("dfc", "lc"))
                assert os.path.exists(os.path.join(dfcpath, lcpath))

                index_dict.append(
                    dict(
                        basename=basename,
                        dfcpath=path,
                        seed=seed,
                        season=season,
                        region=region,
                        tile=tile,
                        maxclass=maxclass,
                        multi_class=multi_class,
                        multi_class_fractions=multi_class_fractions,
                        s1path=s1path,
                        s2path=s2path,
                        lcpath=lcpath
                    )
                )
            index = pd.DataFrame(index_dict)
            print(f"saving {indexfile}")
            index.to_csv(indexfile)

        index = index.reset_index()
        self.index = index.set_index(["region", "season"])
        self.index = self.index.sort_index()
        self.index = self.index.loc[region]
        self.region_seasons = self.index.index.unique().tolist()



    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index.loc[self.index["index"] == item].iloc[0]

        with rasterio.open(os.path.join(self.dfcpath, row.s1path), "r") as src:
            s1 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.s2path), "r") as src:
            s2 = src.read()

        with rasterio.open(os.path.join(self.dfcpath, row.lcpath), "r") as src:
            lc = src.read(1)

        input, target = self.transform(s1, s2, lc)

        return input.float(), target

@torch.no_grad()
def evaluate_proto(model, support_input, query_input, support_target, query_target, batch_size=16):
    model.eval()

    support_features = model(support_input)
    #query_features = model(query_input)
    query_features = torch.vstack([model(inp.float()) for inp in torch.split(query_input, batch_size)])

    labels = torch.unique(support_target, sorted=True)

    y = reset_indices(support_target.clone().unsqueeze(0)).squeeze()
    y_test = reset_indices(query_target.clone().unsqueeze(0)).squeeze()

    prototypes = []
    for c in torch.unique(y,sorted=True):
        prototypes.append(support_features[y == c].mean(0))
    prototypes = torch.stack(prototypes)

    sq_distances = torch.sum((prototypes
        - query_features.unsqueeze(1)) ** 2, dim=-1)

    y_pred = sq_distances.argmin(-1)

    assert (labels[y_test] == query_target).all()
    return labels[y_pred], query_target

def main_proto(args):
    args.one_vs_all = False
    args.resnet = True

    transform, model, mask, mask_optimizer = prepare_transform_and_model(args)
    model.load_state_dict(torch.load(os.path.join(args.output_folder, "model_best.pth")))

    for region in regions:
        folder = f"dfc2020_{args.num_shots}shots"
        results_dir = os.path.join(args.output_folder, folder)
        os.makedirs(results_dir, exist_ok=True)

        ds = DFCDataset(dfcpath="/data/sen12ms/DFC_Public_Dataset", region=region, transform=transform)
        support_input, support_target, query_input, query_target, support, query = split_support_query(ds, args.num_shots)
        query_predictions, query_target = evaluate_proto(model, support_input, query_input, support_target,
                                                         query_target)

        query["predictions"] = query_predictions
        query["targets"] = query_target

        support = add_geometry(support, ds.dfcpath)
        query = add_geometry(query, ds.dfcpath)

        support.to_csv(os.path.join(results_dir, f"{region[0]}_{region[1]}_support.csv"))
        query.to_csv(os.path.join(results_dir, f"{region[0]}_{region[1]}_query.csv"))
        support.to_file(os.path.join(results_dir, f"{region[0]}_{region[1]}_support.geojson"), driver="GeoJSON")
        query.to_file(os.path.join(results_dir, f"{region[0]}_{region[1]}_query.geojson"), driver="GeoJSON")


def parse_args():
    parser = argparse.ArgumentParser('A central script to select learning scheme and mode')
    parser.add_argument('--dfc-path', type=str, default="/data/sen12ms/DFC_Public_Dataset")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--num-shots', type=int, nargs='+', default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--output-folder', type=str, default="/data/sen12ms/models/onevsall/maml_2step_layernorm")
    parser.add_argument('--gradient-steps', type=int, default=20)
    parser.add_argument('--inner-step-size', type=float, default=0.1)
    parser.add_argument('--s2only', action='store_true')
    parser.add_argument('--rgbonly', action='store_true')
    parser.add_argument('--prototypicalnetwork', action='store_true',
                    help='Use CUDA if available.')
    parser.add_argument('--ensemble', action='store_true', help='use bag of maml ensemble')
    parser.add_argument('--norm', type=str, default="instancenorm", #
                        help='normalization of the resnet model. naming following Bronskill et al., 2020.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    num_shots_list = args.num_shots
    for num_shots in tqdm(num_shots_list):
        args.num_shots = num_shots
        if args.prototypicalnetwork:
            main_proto(args)
        else:
            main(args)