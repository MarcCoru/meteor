import torch
import numpy as np

from .resnet import ResNet
from torchmeta.modules import MetaConv2d

MODEL_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/app/model.pth"

# band sequence of the pre-trained model
bands = ["S1VV", "S1VH", "S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
           "S2B12"]

def modify_weights_for_segmentation(state_dict):
    state_dict["classifier.weight"] = state_dict["classifier.weight"].unsqueeze(-1).unsqueeze(-1)
    return state_dict

def get_model(model="maml_resnet12", pretrained=True, segmentation=False, subset_bands=bands):
    assert set(subset_bands).intersection(bands) == set(subset_bands), f"the subset bands must be a subset of {bands}"

    avg_pool = not segmentation

    if model == "maml_resnet12":
        model = ResNet(inplanes=15, out_features=1, normtype="instancenorm", avg_pool=avg_pool)

        band_idxs = np.array([bands.index(b) for b in subset_bands])
        # modify model with less bands
        model.layer1[0].conv1 = MetaConv2d(len(subset_bands), 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                           bias=False)
        model.layer1[0].downsample[0] = MetaConv2d(len(subset_bands), 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")

            # remove names introduced by parallel training
            state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}

            # modify state dict
            state_dict["layer1.0.conv1.weight"] = state_dict["layer1.0.conv1.weight"][:, band_idxs]
            state_dict["layer1.0.downsample.0.weight"] = state_dict["layer1.0.downsample.0.weight"][:, band_idxs]

            if segmentation:
                state_dict = modify_weights_for_segmentation(state_dict)
            model.load_state_dict(state_dict)
    else:
        raise ValueError("invalid model")

    return model
