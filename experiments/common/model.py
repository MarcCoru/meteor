from torchvision import models
from torch import nn
from collections import OrderedDict
from argparse import Namespace
import torch
from torchmeta.modules import (MetaConv2d)
import numpy as np

from common.models.ssltransformerrs import DoubleResNetSimCLRDownstream, DoubleSwinTransformerClassifier
from common.models.swin.swin_conf import swin_conf
from common.models.swin.build import build_model
from common.utils import prepare_transform_and_model
from common.utils.model import prepare_classification_model, prepare_regular_model

from meteor import METEOR
from meteor import models as meteormodels


def reset_indices(targets):
    """
    resets absolute class indices (1,7,5,3) with relative ones (0,1,2,3)
    """
    rows = []
    for row in targets:
        class_ids = row.unique()

        for idx, id in enumerate(class_ids):
            row[row == id] = idx
        rows.append(row)
    return torch.stack(rows)


def get_model(modelname, snapshot_path, inplanes, select_bands=None):

    if modelname == "meteor":
        basemodel = meteormodels.get_model("maml_resnet12", subset_bands=select_bands)
        model = METEOR(basemodel, verbose=False, device="cuda")

    elif modelname == "ssl4eo-dinorn50":
        ''' model from https://github.com/zhu-xlab/SSL4EO-S12 '''

        state_dict = torch.load(snapshot_path)["teacher"]

        state_dict = OrderedDict(
            {k.replace("module.backbone.", ""): v for k, v in state_dict.items() if "backbone" in k})

        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "head" not in k})

        model = models.resnet50(num_classes=10)
        model.fc = nn.Identity()

        inchannels = inplanes if select_bands is None else len(select_bands)
        model.conv1 = nn.Conv2d(inchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]
        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            state_dict["conv1.weight"] = state_dict["conv1.weight"][:, band_idxs]

        model.load_state_dict(state_dict)

    elif modelname == "ssl4eo-mocorn50":
        ''' model from https://github.com/zhu-xlab/SSL4EO-S12 '''

        state_dict = torch.load(snapshot_path)["state_dict"]

        state_dict = OrderedDict(
            {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items() if "encoder_q" in k})

        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "fc." not in k})

        model = models.resnet50(num_classes=10)
        model.fc = nn.Identity()

        inchannels = inplanes if select_bands is None else len(select_bands)
        model.conv1 = nn.Conv2d(inchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]
        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            state_dict["conv1.weight"] = state_dict["conv1.weight"][:, band_idxs]

        model.load_state_dict(state_dict)

    elif modelname == "baseline-resnet12":
        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]

        model = meteormodels.get_model("maml_resnet12", subset_bands=select_bands)

        # a custom identity class to accomodate for the torchmeta ResNet12 implementation
        class MetaIdentity(nn.Module):
            def __init__(self) -> None:
                super(MetaIdentity, self).__init__()

            def forward(self, input, params=None):
                return input

        model.classifier = MetaIdentity()

        state_dict = torch.load(snapshot_path)["model_state_dict"]

        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "classifier." not in k})

        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            state_dict["layer1.0.conv1.weight"] = state_dict["layer1.0.conv1.weight"][:, band_idxs]
            state_dict["layer1.0.downsample.0.weight"] = state_dict["layer1.0.downsample.0.weight"][:, band_idxs]

        model.load_state_dict(state_dict)

    elif modelname == "baseline-resnet18":
        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]

        state_dict = torch.load(snapshot_path)["model_state_dict"]
        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            state_dict["conv1.weight"] = state_dict["conv1.weight"][:, band_idxs]

        model = prepare_regular_model("resnet18", n_classes=10,
                                      n_channels=len(select_bands) if select_bands is not None else 13)

        model.load_state_dict(state_dict)
        model.fc = nn.Identity()

    elif modelname == "baseline-resnet50":
        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]

        state_dict = torch.load(snapshot_path)["model_state_dict"]
        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            state_dict["conv1.weight"] = state_dict["conv1.weight"][:, band_idxs]

        model = prepare_regular_model("resnet50", n_classes=10,
                                      n_channels=len(select_bands) if select_bands is not None else 13)

        model.load_state_dict(state_dict)
        model.fc = nn.Identity()

    elif modelname in ["ssltransformerrs-resnet50", "ssltransformerrs-resnet18"]:
        """
        from https://github.com/HSG-AIML/SSLTransformerRS
        """

        checkpoint = torch.load(snapshot_path)

        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]

        base_model = "resnet50" if "resnet50" in modelname else "resnet18"
        model = DoubleResNetSimCLRDownstream(base_model=base_model, out_dim=8)

        model.backbone2.conv1 = torch.nn.Conv2d(
            inplanes if select_bands is None else len(select_bands),
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        state_dict = checkpoint["state_dict"]

        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            state_dict["backbone2.conv1.weight"] = state_dict["backbone2.conv1.weight"][:, band_idxs]

        model.load_trained_state_dict(state_dict)
        model = model.backbone2

    elif modelname == "ssltransformerrs-swin":
        """
        from https://github.com/HSG-AIML/SSLTransformerRS
        """

        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]

        swin_conf.model_config.MODEL.SWIN.IN_CHANS = 13
        s2_backbone = build_model(swin_conf.model_config)

        checkpoint = torch.load(snapshot_path)

        weights = checkpoint["state_dict"]
        # Sentinel-1 stream weights
        s1_weights = {
            k[len("backbone1."):]: v for k, v in weights.items() if "backbone1" in k
        }

        # Sentinel-2 stream weights
        s2_weights = {
            k[len("backbone2."):]: v for k, v in weights.items() if "backbone2" in k
        }

        # dynamically modify for different number of input channels
        s2_backbone.patch_embed.proj = nn.Conv2d(inplanes if select_bands is None else len(select_bands),
                                                 96, kernel_size=(4, 4), stride=(4, 4))

        if select_bands is not None:
            band_idxs = np.array([bands.index(b) for b in select_bands])
            s2_weights["patch_embed.proj.weight"] = s2_weights["patch_embed.proj.weight"][:, band_idxs]

        s2_backbone.load_state_dict(s2_weights)

        model = DoubleSwinTransformerClassifier(s2_backbone, out_dim=8, device="cuda").to("cuda")
        model.fc = nn.Identity()


    elif modelname == "seco_resnet50":
        # model downloaded from https://github.com/ElementAI/seasonal-contrast
        state_dict = torch.load(snapshot_path)
        resnet = models.resnet50()

        model = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        modified_state_dict = OrderedDict(
            {k.replace("encoder_k.", ""): v for k, v in state_dict.items() if "encoder_k" in k})
        model.load_state_dict(modified_state_dict)

    elif modelname == "dino_resnet50":
        dino_state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth")
        model = models.resnet50()
        model.fc = nn.Identity()
        model.load_state_dict(dino_state_dict)

    elif modelname == "swav_resnet50":
        swav_state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar")
        model = models.resnet50()
        model.fc = nn.Identity()
        state_dict = {k.replace("module.", ""): v for k, v in swav_state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    elif modelname == "imagenet_resnet50":
        model = models.resnet50(pretrained=True)

    elif modelname == "scratch_resnet50":
        model = models.resnet50(pretrained=False)

    elif modelname == "seco_resnet18":
        # model downloaded from https://github.com/ElementAI/seasonal-contrast
        state_dict = torch.load(snapshot_path)
        resnet = models.resnet18()

        model = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        modified_state_dict = OrderedDict(
            {k.replace("encoder_k.", ""): v for k, v in state_dict["state_dict"].items() if "encoder_k" in k})
        model.load_state_dict(modified_state_dict)

    elif modelname == "proto":
        s2only = inplanes == 13
        rgbonly = inplanes == 3

        args = Namespace(
            one_vs_all=True,
            resnet=True,
            s2only=True,
            rgbonly=False,
            norm="instancenorm",
            prototypicalnetwork=True
        )

        transform, model, mask, mask_optimizer = prepare_transform_and_model(args)
        state_dict = torch.load(snapshot_path)

        bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
         "S2B12"]
        band_idxs = np.array([bands.index(b) for b in select_bands])

        # modify model
        model.conv1 = nn.Conv2d(len(select_bands), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # modify state dict
        state_dict["conv1.weight"] = state_dict["conv1.weight"][:,band_idxs]
        model.load_state_dict(state_dict)

    elif modelname == "bagofmaml":
        #model, *_ = prepare_classification_model(nclasses=1, inplanes=inplanes, resnet=True, norm="instancenorm")
        #model.load_state_dict(torch.load(snapshot_path))

        sel_bands = select_bands

        s1bands = ["S1VV", "S1VH"]
        s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                   "S2B12"]
        bands = s1bands + s2bands

        band_idxs = np.array([bands.index(b) for b in sel_bands])

        model, *_ = prepare_classification_model(nclasses=1, inplanes=15, resnet=True, norm="instancenorm")
        state_dict = torch.load(snapshot_path)

        # modify model
        model.layer1[0].conv1 = MetaConv2d(len(sel_bands), 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                           bias=False)
        model.layer1[0].downsample[0] = MetaConv2d(len(sel_bands), 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

        state_dict = OrderedDict({k.replace("module.", ""):v for k,v in state_dict.items()})

        # modify state dict
        state_dict["layer1.0.conv1.weight"] = state_dict["layer1.0.conv1.weight"][:, band_idxs]
        state_dict["layer1.0.downsample.0.weight"] = state_dict["layer1.0.downsample.0.weight"][:, band_idxs]

        model.load_state_dict(state_dict)

    else:
        raise ValueError(f"modelname {modelname} unknown")

    return model


class PrototypicalWrapper():
    def __init__(self, model):
        self.model = model
        self.labels = None

    @torch.no_grad()
    def fit(self, support_input, support_target):
        self.model.eval()
        self.labels = torch.unique(support_target, sorted=True)

        support_features = self.model(support_input.float())
        y = reset_indices(support_target.clone().unsqueeze(0)).squeeze()

        prototypes = []
        for c in torch.unique(y, sorted=True):
            prototypes.append(support_features[y == c].mean(0))
        self.prototypes = torch.stack(prototypes)

    @torch.no_grad()
    def predict(self, query_input, batch_size=8):
        self.model.eval()
        query_features = torch.vstack([self.model(inp.float()) for inp in torch.split(query_input, batch_size)])
        sq_distances = torch.sum((self.prototypes
                                  - query_features.unsqueeze(1)) ** 2, dim=-1)

        y_pred = sq_distances.argmin(-1)
        return y_pred, sq_distances
