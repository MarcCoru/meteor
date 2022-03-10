import unittest
from bagofmaml import BagOfMAML, BagOfMAMLEnsemble
from bagofmaml.models import ResNet, get_model
import torch

IMAGE_H, IMAGE_W = 128, 128
BATCH_SIZE = 10
N_CLASSES = 3
NUM_MEMBERS = 3

class Tests(unittest.TestCase):

    def test_resnet(self):
        model = ResNet(inplanes=3, out_features=1)
        logits = model(torch.ones(BATCH_SIZE, 3, IMAGE_H, IMAGE_W))
        self.assertFalse(logits.isnan().any(), "NANs in model predictions")  # add assertion here
        self.assertEqual(logits.shape, torch.Size([BATCH_SIZE,1]))

    def test_get_model(self):
        model = get_model("maml_rgb")
        self.assertFalse(model(torch.ones(1, 3, IMAGE_H, IMAGE_W)).isnan().any(), "NANs in model predictions")

        model = get_model("maml_s1s2")
        self.assertFalse(model(torch.ones(1, 15, IMAGE_H, IMAGE_W)).isnan().any(), "NANs in model predictions")

        model = get_model("maml_s2")
        self.assertFalse(model(torch.ones(1, 13, IMAGE_H, IMAGE_W)).isnan().any(), "NANs in model predictions")

    def test_bagofmaml(self):

        basemodel = get_model("maml_rgb")
        bag = BagOfMAML(basemodel)

        bag.fit(torch.rand(BATCH_SIZE, 3, IMAGE_H, IMAGE_W), torch.randint(N_CLASSES, (BATCH_SIZE,)))
        y_pred, y_score = bag.predict(torch.rand(BATCH_SIZE, N_CLASSES, IMAGE_H, IMAGE_W))
        self.assertEqual(y_score.shape, torch.Size([N_CLASSES, BATCH_SIZE]))

    def test_bagofmamlensemble(self):

        basemodel = get_model("maml_rgb")
        bag = BagOfMAMLEnsemble(basemodel, num_members=NUM_MEMBERS)

        bag.fit(torch.rand(BATCH_SIZE, 3, IMAGE_H, IMAGE_W), torch.randint(N_CLASSES, (BATCH_SIZE,)))
        y_pred, y_score = bag.predict(torch.rand(BATCH_SIZE, N_CLASSES, IMAGE_H, IMAGE_W))
        self.assertEqual(y_score.shape, torch.Size([N_CLASSES, NUM_MEMBERS, BATCH_SIZE]))

if __name__ == '__main__':
    unittest.main()
