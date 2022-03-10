import torch

from .resnet import ResNet

MAML_RGB_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/models/maml_rgb.pth"
MAML_S1S2_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/models/maml_s1s2.pth"
MAML_S2_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/models/maml_s2.pth"


def get_model(model, pretrained=True):
    if model == "maml_resnet12_rgb":
        model = ResNet(inplanes=3, out_features=1, normtype="instancenorm", avg_pool=True)

        if pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url(MAML_RGB_URL, map_location="cpu"))

    elif model == "maml_resnet12_s1s2":
        model = ResNet(inplanes=15, out_features=1, normtype="instancenorm", avg_pool=True)

        if pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url(MAML_S1S2_URL, map_location="cpu"))

    elif model == "maml_resnet12_s2":
        model = ResNet(inplanes=13, out_features=1, normtype="instancenorm", avg_pool=True)

        if pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url(MAML_S2_URL, map_location="cpu"))

    else:
        raise ValueError("invalid model")

    return model
