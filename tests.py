import unittest

import pytest

from bagofmaml import BagOfMAML
from bagofmaml.models import ResNet, get_model
import torch

IMAGE_H, IMAGE_W = 32, 32
BATCH_SIZE = 10
N_CLASSES = 3
NUM_MEMBERS = 3

RGBBANDS = ["S2B4", "S2B3", "S2B2"]
S2BANDS = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
           "S2B12"]
S1BANDS = ["S1VV", "S1VH"]
ALLBANDS =  ["S1VV", "S1VH", "S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
           "S2B12"]

class Tests(unittest.TestCase):

    def test_resnet(self):
        model = ResNet(inplanes=3, out_features=1)
        logits = model(torch.ones(BATCH_SIZE, 3, IMAGE_H, IMAGE_W))
        self.assertFalse(logits.isnan().any(), "NANs in model predictions")  # add assertion here
        self.assertEqual(logits.shape, torch.Size([BATCH_SIZE,1]))

    def test_error_notsubsetbands(self):
        with pytest.raises(AssertionError):
            get_model("maml_resnet12", subset_bands=RGBBANDS + ["dfferentband"])

    def test_get_model(self):
        model = get_model("maml_resnet12", subset_bands=RGBBANDS)
        self.assertFalse(model(torch.ones(1, 3, IMAGE_H, IMAGE_W)).isnan().any(), "NANs in model predictions")

        model = get_model("maml_resnet12", subset_bands=ALLBANDS)
        self.assertFalse(model(torch.ones(1, 15, IMAGE_H, IMAGE_W)).isnan().any(), "NANs in model predictions")

        model = get_model("maml_resnet12", subset_bands=S2BANDS)
        self.assertFalse(model(torch.ones(1, 13, IMAGE_H, IMAGE_W)).isnan().any(), "NANs in model predictions")

    def test_bagofmaml(self):

        basemodel = get_model("maml_resnet12", subset_bands=RGBBANDS)
        bag = BagOfMAML(basemodel)

        bag.fit(torch.rand(BATCH_SIZE, 3, IMAGE_H, IMAGE_W), torch.randint(N_CLASSES, (BATCH_SIZE,)))
        y_pred, y_score = bag.predict(torch.rand(BATCH_SIZE, N_CLASSES, IMAGE_H, IMAGE_W))
        self.assertEqual(y_score.shape, torch.Size([N_CLASSES, BATCH_SIZE]))

    def test_bagofmaml_segmentation(self):
        basemodel = get_model("maml_resnet12", subset_bands=RGBBANDS, segmentation=True)
        bag = BagOfMAML(basemodel)
        bag.fit(torch.rand(BATCH_SIZE, 3, IMAGE_H, IMAGE_W), torch.randint(N_CLASSES, (BATCH_SIZE, IMAGE_H, IMAGE_W)))
        y_pred, y_score = bag.predict(torch.rand(BATCH_SIZE, N_CLASSES, IMAGE_H, IMAGE_W))
        self.assertEqual(y_score.shape, torch.Size([N_CLASSES, BATCH_SIZE, IMAGE_H, IMAGE_W]))

    def test_getting_started_examples_classification(self):
        # initialize an RGB model
        basemodel = get_model("maml_resnet12", subset_bands=["S2B4", "S2B3", "S2B2"])
        bag = BagOfMAML(basemodel)

        # fine-tune model to labelled data
        X_support, y_support = torch.rand(10, 3, 32, 32), torch.randint(3, (10,))
        bag.fit(X_support, y_support)

        # predict
        X_query = torch.rand(10, 3, 32, 32)
        y_pred, y_scores = bag.predict(X_query)

    def test_getting_started_examples_segmentation(self):
        # initialize an RGB model
        basemodel = get_model("maml_resnet12", subset_bands=["S2B4", "S2B3", "S2B2"], segmentation=True)
        bag = BagOfMAML(basemodel)

        # fine-tune model to labelled data
        X_support, y_support = torch.rand(10, 3, 32, 32), torch.randint(3, (10, 32, 32))
        bag.fit(X_support, y_support)

        # predict
        X_query = torch.rand(10, 3, 32, 32)
        y_pred, y_scores = bag.predict(X_query)


if __name__ == '__main__':
    unittest.main()
