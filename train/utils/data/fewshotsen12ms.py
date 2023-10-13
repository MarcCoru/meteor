import os
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch

from torch.utils.data.sampler import RandomSampler
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
import random

from .data import CSVSIZE, CSVURL, H5SIZE, H5URL, trainregions, valregions, holdout_regions, IGBP_simplified_classes

def sen12ms(folder, shots, ways, shuffle=True, test_shots=None,
            seed=None, **kwargs):
    if test_shots is None:
        test_shots = shots

    dataset = Sen12MS(folder, num_classes_per_task=ways, min_samples_per_class=shots + test_shots,
                      min_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
                            num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset

def prepare_dataset(args, transform):
    dataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                      target_transform=None,
                      meta_split="train", shuffle=True)

    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     sampler=CombinationSubsetRandomSampler(dataset))

    valdataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                         target_transform=None,
                         meta_split="val", shuffle=True)

    valdataloader = BatchMetaDataLoader(valdataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        sampler=CombinationSubsetRandomSampler(valdataset))

    testdataset = sen12ms(args.dataset_path, shots=args.num_shots, ways=args.num_ways, transform=transform,
                         target_transform=None,
                         meta_split="test", shuffle=True)

    testdataloader = BatchMetaDataLoader(testdataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        sampler=CombinationSubsetRandomSampler(testdataset))

    return dataloader, valdataloader, testdataloader

class Sen12MS(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_split="train", transform=None,
                 target_transform=None, min_samples_per_class=None, min_classes_per_task=0, **kwargs):
        dataset = Sen12MSClassDataset(root, meta_split=meta_split, transform=transform,
                                      target_transform=target_transform, min_samples_per_class=min_samples_per_class,
                                      min_classes_per_task=min_classes_per_task, **kwargs)

        super(Sen12MS, self).__init__(dataset, num_classes_per_task, target_transform=target_transform)


class Sen12MSClassDataset(ClassDataset):

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split="train",
                 transform=None, target_transform=None, class_augmentations=None, min_samples_per_class=None,
                 min_classes_per_task=None, simplified_igbp_labels=True):
        super(Sen12MSClassDataset, self).__init__(meta_train=meta_train,
                                                  meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                                  class_augmentations=class_augmentations)
        print(f"Initializing {meta_split} meta-dataset")

        self.transform = transform
        self.target_transform = target_transform
        self.meta_test = meta_test
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.root = root
        self.h5file_path = os.path.join(self.root, "sen12ms.h5")
        self.paths_file = os.path.join(self.root, "sen12ms.csv")


        if self.meta_train or self.meta_split == "train":
            regions = trainregions
        elif self.meta_val or self.meta_split == "val":
            regions = valregions
        elif self.meta_test or self.meta_split == "test":
            regions = holdout_regions
        else:
            raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                 "meta_split must be in 'train','val','test'")

        self.regions = regions
        seasons = ["summer", "spring", "fall", "winter"]

        self.paths = pd.read_csv(self.paths_file, index_col=0)

        if simplified_igbp_labels:
            self.classes = IGBP_simplified_classes
        else:
            self.classes = IGBP_classes

        # list of all regions with classes
        counts = self.paths[["season", "region", "maxclass", "lcpath"]].groupby(
            by=["season", "region", "maxclass"]).count().reset_index()
        if min_samples_per_class is not None:
            mask = counts["lcpath"] > min_samples_per_class
            self._labels = counts.loc[mask][["season", "region", "maxclass"]]
            print(
                f"keeping {mask.sum()}/{len(counts)} region/class pairs with >{min_samples_per_class} samples per class")
        else:
            self._labels = counts[["season", "region", "maxclass"]]

        tasks_idxs = list()
        for region in regions:
            for season in seasons:
                mask = (self._labels["region"] == region) & (self._labels["season"] == season)
                task_idx = self._labels.reset_index(drop=True).index[mask].tolist()
                if len(task_idx) > min_classes_per_task:
                    tasks_idxs.append(task_idx)
        self.task_idxs = tasks_idxs
        print(
            f"keeping {len(tasks_idxs)}/{len(regions) * len(seasons)} regions/season pairs with >{min_classes_per_task} unique classes per region")

        """
        self.metadata = list()
        for idx in range(len(self.labels)):
            regionclass = self.labels.iloc[idx]
            selection_mask = (self.paths["region"] == regionclass.region) & \
                             (self.paths["maxclass"] == regionclass.maxclass)
            selected_paths = self.paths.loc[selection_mask]
            self.metadata.append(selected_paths[["lcpath", "s1path", "s2path", "npzpath","h5path","tile"]].values)
        """

        # speedup when querying from array in __get_item__
        self._labels = self._labels.values
        self._data = None

    @property
    def num_classes(self):
        return len(self.labels)

    @property
    def data(self):
        # if self._data is None:
        #    self._data = h5py.File(self.h5file_path, 'r')
        return None  # self._data

    @property
    def labels(self):
        return self._labels

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def __getitem__(self, idx):
        season, region, classname = self.labels[idx]
        subgroup = f"{season}/{region}/{classname.replace(' ', '_').replace('/', '_')}"
        return Sen12MSDataset(idx, self.h5file_path, subgroup, region, classname, self.transform, self.target_transform)


class Sen12MSDataset(Dataset):
    def __init__(self, index, h5file_path, group, region, classname, transform=None,
                 target_transform=None, debug=False):
        super(Sen12MSDataset, self).__init__(index)

        # IGBP [2], and LCCS Land Cover, Land Use, and Surface Hydrology [3].

        self.h5file_path = h5file_path
        with h5py.File(h5file_path, 'r') as data:
            self.tiles = list(data[group].keys())
        self.group = group
        self.transform = transform
        self.target_transform = target_transform
        self.region = region
        self.classname = classname
        self.counter = 0
        self.debug = debug

    def __len__(self):
        return len(self.tiles)

    def load_tiff(self, lcpath, s1path, s2path):
        with rasterio.open(os.path.join(self.root, s2path), "r") as src:
            s2 = src.read()

        with rasterio.open(os.path.join(self.root, s1path), "r") as src:
            s1 = src.read()

        with rasterio.open(os.path.join(self.root, lcpath), "r") as src:
            lc = src.read(1)

        image = np.vstack([s1, s2]).swapaxes(0, 2)
        target = lc
        return image, target

    def _tiff2npz(self, lcpath, s1path, s2path, npzpath):
        """takes 50ms per loop"""
        image, target = self.load_tiff(lcpath, s1path, s2path)

        npzpath = os.path.join(self.root, npzpath)
        os.makedirs(os.path.dirname(npzpath), exist_ok=True)
        np.savez(npzpath, image=image, target=target)
        return image, target

    def load_npz(self, npzpath):
        with np.load(os.path.join(self.root, npzpath), allow_pickle=True) as f:
            image = f["image"]
            target = f["target"]
        return image, target

    def __getitem__(self, index):
        tile = self.tiles[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[self.group + "/" + tile + "/s2"][()]
            s1 = data[self.group + "/" + tile + "/s1"][()]
            label = data[self.group + "/" + tile + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        # if self.target_transform is not None:
        #    target = self.target_transform((target, cropxy))

        if self.debug:
            self.counter += 1
            print(f"{self.region:<4}, {self.classname:<50}, {self.counter:<20}")

        return image, target, self.group + "/" + tile


class CombinationSubsetRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise ValueError()
        self.data_source = data_source

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        task_idxs = self.data_source.dataset.task_idxs
        num_tasks = len(task_idxs)

        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            idxs = task_idxs[random.choice(range(num_tasks))]
            if len(idxs) >= num_classes_per_task:
                yield tuple(random.sample(idxs, num_classes_per_task))
            else:
                raise ValueError(f"{num_classes_per_task} are not enough classes for task idxs {idxs}")
