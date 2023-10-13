import torch
import os
import pandas as pd
from .data import trainregions, valregions, holdout_regions
import h5py
import numpy as np

class RegionSen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, root, region, fold, transform, classes=None, seasons=None, train_test_ratio=0.75, random_seed=0):
        super(RegionSen12MSDataset, self).__init__()
        assert fold in ["train", "test"], "splitting tiles o region randomly. only train or tet folds are allowed"
        assert type(region) == int, "region must be specified as int according to the regions in data.py"

        self.transform = transform

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        self.paths = pd.read_csv(index_file, index_col=0)

        if region in trainregions:
            group = "training regions"
        if region in valregions:
            group = "validation regions"
        if region in holdout_regions:
            group = "hold-out regions"

        mask = self.paths.region == region
        print(f"region {region} specified ({group}). Keeping {mask.sum()} of {len(mask)} tiles")
        self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

        # fix random state with seed to ensure same mask if invoked with fold==train or fold==test
        mask = np.random.RandomState(random_seed).rand(len(self.paths)) < train_test_ratio
        if fold == "test":
            # invert mask
            mask = ~mask

        self.paths = self.paths[mask]
        print(f"fold {fold} specified. Keeping {mask.sum()} of {len(mask)} tiles")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths.iloc[index]

        with h5py.File(self.h5file_path, 'r') as data:
            s2 = data[path.h5path + "/s2"][()]
            s1 = data[path.h5path + "/s1"][()]
            label = data[path.h5path + "/lc"][()]

        image, target = self.transform(s1, s2, label)

        return image, target, path.h5path