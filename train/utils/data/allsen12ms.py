import torch
import os
import pandas as pd
from .data import trainregions, valregions, holdout_regions
import h5py

class AllSen12MSDataset(torch.utils.data.Dataset):
    def __init__(self, root, fold, transform, classes=None, seasons=None):
        super(AllSen12MSDataset, self).__init__()

        self.transform = transform

        self.h5file_path = os.path.join(root, "sen12ms.h5")
        index_file = os.path.join(root, "sen12ms.csv")
        self.paths = pd.read_csv(index_file, index_col=0)

        if fold == "train":
            regions = trainregions
        elif fold == "val":
            regions = valregions
        elif fold == "test":
            regions = holdout_regions
        elif fold == "all":
            regions = holdout_regions + valregions + trainregions
        else:
            raise AttributeError("one of meta_train, meta_val, meta_test must be true or "
                                 "fold must be in 'train','val','test'")

        mask = self.paths.region.isin(regions)
        print(f"fold {fold} specified. Keeping {mask.sum()} of {len(mask)} tiles")
        self.paths = self.paths.loc[mask]
        if classes is not None:
            mask = self.paths.maxclass.isin(classes)
            print(f"classes {classes} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]
        if seasons is not None:
            mask = self.paths.season.isin(seasons)
            print(f"seasons {seasons} specified. Keeping {mask.sum()} of {len(mask)} tiles")
            self.paths = self.paths.loc[mask]

        # shuffle the tiles once
        self.paths = self.paths.sample(frac=1)

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