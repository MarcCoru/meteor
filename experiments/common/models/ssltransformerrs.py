import torchvision.models as models
import torch
from torchvision.transforms import Resize
import numpy as np

"""
code from https://github.com/HSG-AIML/SSLTransformerRS/blob/main/demo/demo_resnet_backbone.ipynb
"""

import albumentations as A

def get_norm(bands, select_bands):

    # e.g. remove S2 from "S2B1" to make band names match across different datasets
    s = [s.replace("S2","") for s in select_bands]
    all_bands = [s.replace("S2", "") for s in bands]
    band_idxs = np.array([all_bands.index(b) for b in s])

    # values from https://github.com/HSG-AIML/SSLTransformerRS/blob/7bc460680d3f709bdef55186f001de606efcc34b/Transformer_SSL/data/build.py#L126
    means = np.array([
        80.2513,
        67.1305,
        61.9878,
        61.7679,
        73.5373,
        105.9787,
        121.4665,
        118.3868,
        132.6419,
        42.9694,
        1.3114,
        110.6207,
        74.3797,
    ])

    stds = np.array([
        4.5654,
        7.4498,
        9.4785,
        14.4985,
        14.3098,
        20.0204,
        24.3366,
        25.5085,
        27.1181,
        7.5455,
        0.1892,
        24.8511,
        20.4592,
    ])

    normalize = A.Normalize(
                        mean=means[band_idxs],
                        std=stds[band_idxs],
                    )

    # folloing this implementation
    # https://github.com/HSG-AIML/SSLTransformerRS/blob/7bc460680d3f709bdef55186f001de606efcc34b/dfc_dataset.py#L383
    def norm_s2maxs(arr):
        s2_norm = []
        for s2 in arr:
            s2_maxs = []
            for b_idx in range(s2.shape[0]):
                s2_maxs.append(
                    torch.ones((s2.shape[-2], s2.shape[-1])) * s2[b_idx].max().item() + 1e-5
                )
            s2_maxs = torch.stack(s2_maxs)
            s2_norm.append(s2 / s2_maxs)
        return torch.stack(s2_norm)
    return norm_s2maxs

class DoubleSwinTransformerClassifier(torch.nn.Module):
    """
    modified from https://github.com/HSG-AIML/SSLTransformerRS/blob/main/demo/demo_swin_backbone.ipynb
    """
    def __init__(self, encoder2, out_dim, device, freeze_layers=True):
        super(DoubleSwinTransformerClassifier, self).__init__()

        # If you're only using one of the two backbones, just comment the one you don't need
        #self.backbone1 = encoder1
        self.backbone2 = encoder2
        self.resize = Resize(224) # swin is fixed to 224x224 px images

        self.device = device

        # add final linear layer
        self.fc = torch.nn.Linear(
            self.backbone2.num_features,
            out_dim,
            bias=True,
        )

        # freeze all layers but the last fc
        if freeze_layers:
            for name, param in self.named_parameters():
                if name not in ["fc.weight", "fc.bias"]:
                    param.requires_grad = False

    def forward(self, x):
        #x1, _, _ = self.backbone1.forward_features(x["s1"].to(self.device))
        x = self.resize(x)

        x2, _, _ = self.backbone2.forward_features(x.to(self.device))

        z = self.fc(x2)

        # If you're only using one of the two backbones, you may comment the lines above and use the following:
        # x1, _, _ = self.backbone1.forward_features(x["s1"].to(self.device))
        # z = self.fc(x1)

        return z


class DoubleResNetSimCLRDownstream(torch.nn.Module):
    """concatenate outputs from two backbones and add one linear layer"""

    def __init__(self, base_model, out_dim):
        super(DoubleResNetSimCLRDownstream, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50,}


        self.backbone2 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
        dim_mlp2 = self.backbone2.fc.in_features

        # If you are using multimodal data you can un-comment the following lines:
        # self.backbone1 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
        # dim_mlp1 = self.backbone1.fc.in_features

        # add final linear layer
        self.fc = torch.nn.Linear(dim_mlp2, out_dim, bias=True)
        # self.fc = torch.nn.Linear(dim_mlp1 + dim_mlp2, out_dim, bias=True)

        # self.backbone1.fc = torch.nn.Identity()
        self.backbone2.fc = torch.nn.Identity()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x2 = self.backbone2(x["s2"])

        # If you are using multimodal data you can un-comment the following lines and comment z = self.fc(x2):
        # x1 = self.backbone1(x["s1"])
        # z = torch.cat([x1, x2], dim=1)
        # z = self.fc(z)

        z = self.fc(x2)

        return z

    def load_trained_state_dict(self, weights):
        """load the pre-trained backbone weights"""

        # remove the MLP projection heads
        for k in list(weights.keys()):
            if k.startswith(('backbone1.fc', 'backbone2.fc')):
                del weights[k]

        log = self.load_state_dict(weights, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
