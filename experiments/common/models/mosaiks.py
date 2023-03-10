from torch import nn
import numpy as np

import concurrent.futures as fs
from numba.typed import List
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from common.dfc2020.data import DFCDataset, split_support_query
from common.utils.transforms import get_classification_transform
from common.data.allsen12ms import AllSen12MSDataset

import torch
from tqdm import tqdm
import os
from datetime import datetime

MAX_THREADS = 1
TOT_PATCHES = int(1e5)

def build_local_featurizer(support_input, patch_distribution="empirical"):
    """
    build a featurizer specific to the downstream problem defined by the support_input
    """
    config = {"patch_size": 3,
              "seed": 0,
              "type": "random_features",
              "num_filters": 4096,
              "pool_size": 256,
              "pool_stride": 256,
              "bias": 0.0,
              "filter_scale": 1e-3,
              "patch_distribution": patch_distribution}

    patch_size = config["patch_size"]
    pool_size = config["pool_size"]
    pool_stride = config["pool_stride"]
    bias = config["bias"]
    patch_distribution = config["patch_distribution"]
    num_filters = config["num_filters"]
    filter_scale = config["filter_scale"]
    seed = 0
    num_channels = support_input.shape[1]

    #if num_channels == 3:
    #    rgb_idxs = np.array([5, 4, 3])
    #    X_train = support_input.permute(0, 2, 3, 1)[:, :, :, rgb_idxs].numpy()
    #    #X_all = torch.vstack([support_input, query_input])[:, rgb_idxs].permute(0, 2, 3, 1)
    #else:
    X_train = support_input.permute(0, 2, 3, 1).numpy()
        #X_all = torch.vstack([support_input, query_input]).permute(0, 2, 3, 1)

    featurizer = build_featurizer(
        patch_size,
        pool_size,
        pool_stride,
        bias,
        patch_distribution,
        num_filters,
        num_channels,
        seed,
        filter_scale,
        X_train=X_train,
        filter_batch_size=2048)

    return featurizer

def fit_predict_mosaiks(support_input, support_target, query_input, query_target, semi_supervised=False,
                        local_features=True, select_bands=None, patch_distribution="empirical"):
    X_train = support_input  # only training partition
    X = torch.vstack([support_input, query_input])  # all

    if local_features:
        if semi_supervised: # build featurizer based on all data (also test/query data)
            X_features = X
        else: # build featurizer only on support data
            X_features = support_input

        net = build_local_featurizer(X_features, patch_distribution=patch_distribution)
    else:
        net = build_sen12ms_featurizer(select_bands=select_bands)

    data_batchsize = 128
    num_filters = None
    filter_batch_size = None
    gpu = True
    rgb = True
    filterbatch_size = 16


    net.use_gpu = gpu
    if filter_batch_size is None:
        filter_batch_size = net.filter_batch_size
    if num_filters is None:
        num_filters = len(net.filters)
    X_lift_full = []

    starttime = datetime.now()
    for start, end in chunk_idxs_by_size(num_filters, filterbatch_size):
        batches = torch.split(X, data_batchsize)
        X_lift_batch = []
        print(f"generating features {start} to {end}")
        with tqdm(total=len(batches)) as pbar:
            for X_batch in batches:
                if gpu:
                    X_batch = X_batch.cuda()
                X_var = X_batch
                X_lift = net.forward_partial(X_var, start, end).cpu().data.numpy()
                X_lift_batch.append(X_lift)
                pbar.update(X_lift.shape[0])
        X_lift_full.append(np.concatenate(X_lift_batch, axis=0))
    conv_features = np.concatenate(X_lift_full, axis=1)
    net.deactivate()
    endtime = datetime.now()
    #print(f"{(endtime-starttime).total_seconds()}")

    train_features = conv_features[:X_train.shape[0]] # first N training samples
    test_features = conv_features[X_train.shape[0]:] # all other

    rf = RandomForestClassifier()

    classes = list(support_target.unique())
    y_train = [classes.index(y) for y in support_target]
    y_test = [classes.index(y) for y in query_target]

    rf = rf.fit(train_features.squeeze(), y_train)
    y_pred = rf.predict(test_features.squeeze())

    y_score = rf.predict_proba(test_features.squeeze())

    return y_test, y_pred, y_score

def build_sen12ms_featurizer(size=1000, select_bands=None):
    """
    build a global featurizer from the Sen12MS training set.
    """

    from tqdm import tqdm
    transform = get_classification_transform(s2only=True)
    ds = AllSen12MSDataset(root="/data/sen12ms/", fold="train", transform=transform)

    idxs = np.random.RandomState(0).randint(len(ds), size=size)
    X_train = []
    for idx in tqdm(idxs, total=len(idxs)):
        X,_,_ = ds[idx]
        X_train.append(X)
    X_train = np.stack(X_train)

    if select_bands is not None:
        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]
        band_idxs = np.array([bands.index(b) for b in select_bands])
        X_train = X_train[:,band_idxs]

    config = {"patch_size": 3,
              "seed": 0,
              "type": "random_features",
              "num_filters": 4096,
              "pool_size": 256,
              "pool_stride": 256,
              "bias": 0.0,
              "filter_scale": 1e-3,
              "patch_distribution": "empirical"}

    patch_size = config["patch_size"]
    pool_size = config["pool_size"]
    pool_stride = config["pool_stride"]
    bias = config["bias"]
    patch_distribution = config["patch_distribution"]
    num_filters = config["num_filters"]
    filter_scale = config["filter_scale"]
    seed = 0
    num_channels = len(select_bands)

    featurizer = build_featurizer(
        patch_size,
        pool_size,
        pool_stride,
        bias,
        patch_distribution,
        num_filters,
        num_channels,
        seed,
        filter_scale,
        X_train=X_train.transpose(0,2,3,1), # N x H x W x C
        filter_batch_size=2048)

    return featurizer

def main():


    transform = get_classification_transform(s2only=False)

    regions = [('KippaRing', 'winter'),
               ('MexicoCity', 'winter'),
               ('CapeTown', 'autumn'),
               ('BandarAnzali', 'autumn'),
               ('Mumbai', 'autumn'),
               ('BlackForest', 'spring'),
               ('Chabarovsk', 'summer')]
    num_channels = 15

    eval_shots = [1,2,5,10,15]
    dfcpath = "/data/sen12ms/DFC_Public_Dataset"

    rootdir = "/data/sen12ms/mosaic"
    for shots in eval_shots:
        for region in regions:
            r = region[0]
            resultsdir = os.path.join(rootdir, str(num_channels), str(shots), r)

            ds = DFCDataset(dfcpath=dfcpath, region=region, transform=transform)

            support_input, support_target, query_input, query_target, support, query = split_support_query(ds, shots)

            y_test, y_pred = fit_predict_mosaiks(support_input, support_target, query_input, query_target)


            os.makedirs(resultsdir, exist_ok=True)
            print(classification_report(y_test, y_pred), file=open(os.path.join(resultsdir, "report.txt"), "w"))
            print(accuracy_score(y_test, y_pred), file=open(os.path.join(resultsdir, "accuracy.txt"), "w"))
            print(classification_report(y_test, y_pred))


def build_featurizer(
        patch_size,
        pool_size,
        pool_stride,
        bias,
        patch_distribution,
        num_filters,
        num_channels,
        seed,
        filter_scale,
        X_train=None,
        filter_batch_size=2048,
):
    dtype = "float32"
    if patch_distribution == "empirical":
        assert (
                X_train is not None
        ), "X_train must be provided when patch distribution == empirical"
        all_patches, idxs = grab_patches(
            X_train,
            patch_size=patch_size,
            max_threads=MAX_THREADS,
            seed=seed,
            tot_patches=TOT_PATCHES,
        )
        all_patches = normalize_patches(all_patches, zca_bias=filter_scale)
        print(all_patches.shape[0], num_filters)
        idxs = np.random.choice(all_patches.shape[0], num_filters, replace=False)
        filters = all_patches[idxs].astype(dtype)
        print("filters shape", filters.shape)
    elif patch_distribution == "gaussian":
        filters = (
                np.random.randn(num_filters, num_channels, patch_size, patch_size).astype(
                    dtype
                )
                * filter_scale
        )
        print("filters shape", filters.shape)
    elif patch_distribution == "laplace":
        filters = np.random.laplace(
            loc=0.0,
            scale=filter_scale,
            size=(num_filters * num_channels * patch_size * patch_size),
        ).reshape(num_filters, num_channels, patch_size, patch_size)
        filters = filters.astype("float32")
        print("filters shape", filters.shape)
    else:
        raise Exception(f"Unsupported patch distribution : {patch_distribution}")
    net = BasicCoatesNgNet(
        filters,
        pool_size=pool_size,
        pool_stride=pool_stride,
        bias=bias,
        patch_size=patch_size,
        filter_batch_size=filter_batch_size,
    )
    return net


class BasicCoatesNgNet(nn.Module):
    """All image inputs in torch must be C, H, W"""

    def __init__(
            self,
            filters,
            patch_size=6,
            in_channels=3,
            pool_size=2,
            pool_stride=2,
            bias=1.0,
            filter_batch_size=1024,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.bias = bias
        self.filter_batch_size = filter_batch_size
        self.filters = filters.copy()
        self.active_filter_set = []
        self.start = None
        self.end = None
        self.gpu = False

    def _forward(self, x):
        # Max pooling over a (2, 2) window
        if "conv" not in self._modules:
            raise Exception("No filters active, conv does not exist")
        conv = self.conv(x)
        x_pos = F.avg_pool2d(
            F.relu(conv - self.bias),
            [self.pool_size, self.pool_size],
            stride=[self.pool_stride, self.pool_stride],
            ceil_mode=True,
        )
        x_neg = F.avg_pool2d(
            F.relu((-1 * conv) - self.bias),
            [self.pool_size, self.pool_size],
            stride=[self.pool_stride, self.pool_stride],
            ceil_mode=True,
        )
        return torch.cat((x_pos, x_neg), dim=1)

    def forward(self, x):
        num_filters = self.filters.shape[0]
        activations = []
        for start, end in chunk_idxs_by_size(num_filters, self.filter_batch_size):
            activations.append(self.forward_partial(x, start, end))
        z = torch.cat(activations, dim=1)
        return z

    def forward_partial(self, x, start, end):
        # We do this because gpus are horrible things
        self.activate(start, end)
        return self._forward(x)

    def activate(self, start, end):
        if self.start == start and self.end == end:
            return self
        self.start = start
        self.end = end
        filter_set = torch.from_numpy(self.filters[start:end])
        if self.use_gpu:
            filter_set = filter_set.cuda()
        conv = nn.Conv2d(self.in_channels, end - start, self.patch_size, bias=False)
        # print("rebounding nn.Parameter this shouldn't happen that often")
        conv.weight = nn.Parameter(filter_set)
        self.conv = conv
        self.active_filter_set = filter_set
        return self

    def deactivate(self):
        self.active_filter_set = None


def __grab_patches(images, random_idxs, patch_size=6, tot_patches=1e6, seed=0, scale=0):
    patches = np.zeros(
        (len(random_idxs), images.shape[1], patch_size, patch_size), dtype=images.dtype
    )
    for i, (im_idx, idx_x, idx_y) in enumerate(random_idxs):
        out_patch = patches[i, :, :, :]
        im = images[im_idx]
        grab_patch_from_idx(im, idx_x, idx_y, patch_size, out_patch)
    return patches


def grab_patch_from_idx(im, idx_x, idx_y, patch_size, outpatch):
    sidx_x = int(idx_x - patch_size / 2)
    eidx_x = int(idx_x + patch_size / 2)
    sidx_y = int(idx_y - patch_size / 2)
    eidx_y = int(idx_y + patch_size / 2)
    outpatch[:, :, :] = im[:, sidx_x:eidx_x, sidx_y:eidx_y]
    return outpatch


def grab_patches(
        images, patch_size=6, tot_patches=5e5, seed=0, max_threads=50, scale=0, rgb=True
):
    if rgb:
        images = images.transpose(0, 3, 1, 2)
    idxs = chunk_idxs(images.shape[0], max_threads)
    tot_patches = int(tot_patches)
    patches_per_thread = int(tot_patches / max_threads)
    np.random.seed(seed)
    seeds = np.random.choice(int(1e5), len(idxs), replace=False)

    tot_patches = int(tot_patches)

    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i, (sidx, eidx) in enumerate(idxs):
            images.shape[0]
            im_idxs = np.random.choice(
                images[sidx:eidx, :].shape[0], patches_per_thread
            )
            idxs_x = np.random.choice(
                int(images.shape[2]) - patch_size - 1, tot_patches
            )
            idxs_y = np.random.choice(
                int(images.shape[3]) - patch_size - 1, tot_patches
            )
            idxs_x += int(np.ceil(patch_size / 2))
            idxs_y += int(np.ceil(patch_size / 2))
            random_idxs = list(zip(im_idxs, idxs_x, idxs_y))

            # convert random_ixs to typed list for numba
            rix = List()
            [rix.append(i) for i in random_idxs]

            futures.append(
                executor.submit(
                    __grab_patches,
                    images[sidx:eidx, :],
                    patch_size=patch_size,
                    random_idxs=rix,
                    tot_patches=patches_per_thread,
                    seed=seeds[i],
                    scale=scale,
                )
            )
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    idxs = np.random.choice(results.shape[0], results.shape[0], replace=False)
    return results[idxs], idxs


def chunk_idxs(size, chunks):
    chunk_size = int(np.ceil(size / chunks))
    idxs = list(range(0, size + 1, chunk_size))
    if idxs[-1] != size:
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


def chunk_idxs_by_size(size, chunk_size):
    idxs = list(range(0, size + 1, chunk_size))
    if idxs[-1] != size:
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


def normalize_patches(
        patches, min_divisor=1e-8, zca_bias=0.001, mean_rgb=np.array([0, 0, 0])
):
    if patches.dtype == "uint8":
        patches = patches.astype("float64")
        patches /= 255.0
    print("zca bias", zca_bias)
    n_patches = patches.shape[0]
    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)
    # Zero mean every feature
    patches = patches - np.mean(patches, axis=1)[:, np.newaxis]

    # Normalize
    patch_norms = np.linalg.norm(patches, axis=1)

    # Get rid of really small norms
    patch_norms[np.where(patch_norms < min_divisor)] = 1

    # Make features unit norm
    patches = patches / patch_norms[:, np.newaxis]

    patchesCovMat = 1.0 / n_patches * patches.T.dot(patches)

    (E, V) = np.linalg.eig(patchesCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    patches_normalized = (patches).dot(global_ZCA).dot(global_ZCA.T)

    return patches_normalized.reshape(orig_shape).astype("float32")


def coatesng_featurize(
        net,
        dataset,
        data_batchsize=128,
        num_filters=None,
        filter_batch_size=None,
        gpu=False,
        rgb=True,
):
    net.use_gpu = gpu
    if filter_batch_size is None:
        filter_batch_size = net.filter_batch_size
    if num_filters is None:
        num_filters = len(net.filters)
    X_lift_full = []

    for start, end in chunk_idxs_by_size(num_filters, filterbatch_size):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=data_batchsize)
        X_lift_batch = []
        print(f"generating features {start} to {end}")
        names = []
        with tqdm(total=len(dataset)) as pbar:
            for X_batch_named in data_loader:
                X_batch = X_batch_named[1]
                if gpu:
                    X_batch = X_batch.cuda()
                X_var = X_batch
                names += [x for x in X_batch_named[0]]
                X_lift = net.forward_partial(X_var, start, end).cpu().data.numpy()
                X_lift_batch.append(X_lift)
                pbar.update(X_lift.shape[0])
        X_lift_full.append(np.concatenate(X_lift_batch, axis=0))
    conv_features = np.concatenate(X_lift_full, axis=1)
    net.deactivate()
    return conv_features.reshape(len(dataset), -1), names


if __name__ == '__main__':
    main()
