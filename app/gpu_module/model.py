import math

import numpy as np
from torchmeta.modules import (MetaModule)
from torchvision import models

MODEL_URL = "https://bagofmaml.s3.eu-central-1.amazonaws.com/app/model.pth"


def get_model():
    sel_bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                 "S2B12"]

    s1bands = ["S1VV", "S1VH"]
    s2bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
               "S2B12"]
    bands = s1bands + s2bands

    band_idxs = np.array([bands.index(b) for b in sel_bands])

    model = ResNet(inplanes=15, out_features=1, normtype="instancenorm", avg_pool=True)
    state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")

    # modify model
    model.layer1[0].conv1 = MetaConv2d(len(sel_bands), 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                       bias=False)
    model.layer1[0].downsample[0] = MetaConv2d(len(sel_bands), 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # modify state dict
    state_dict["layer1.0.conv1.weight"] = state_dict["layer1.0.conv1.weight"][:, band_idxs]
    state_dict["layer1.0.downsample.0.weight"] = state_dict["layer1.0.downsample.0.weight"][:, band_idxs]

    model.load_state_dict(state_dict)
    return model


def prepare_classification_model(nclasses, inplanes=15, resnet=True, norm="tbn", avg_pool=True,
                                 prototypicalnetwork=False, gradient_mask=False, device="cpu"):
    if prototypicalnetwork:
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(inplanes, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        if resnet:
            model = ResNet(inplanes=inplanes, out_features=nclasses, normtype=norm, avg_pool=avg_pool)
        else:
            raise NotImplementedError()
            # model = ConvolutionalNeuralNetwork(inplanes, nclasses, input_size=128, hidden_size=64,
            #                               inner_update_lr_init=None, no_batchnorm=no_batchnorm)
    if gradient_mask:
        raise NotImplementedError()
    else:
        mask, mask_optimizer = None, None
    return model, mask, mask_optimizer


class BandEncoder(MetaModule):
    def __init__(self, out_features=128):
        super(BandEncoder, self).__init__()
        self.encoder = MetaConv2d(1, out_features, kernel_size=3, padding=1)

    def modify_input_filters(self, params, filters):
        params["band_encoder.encoder.weight"] = params["band_encoder.encoder.weight"].repeat(1, filters, 1, 1)
        return params

    def forward(self, inputs, params):
        # initialize temporary multi-band encoder (takes 94 micro seconds)
        temporary_encoder = MetaConv2d(inputs.shape[1], 128, kernel_size=3, padding=1)

        # forward takes (82 milli seconds with an [10, 13, 256, 256] image)
        return temporary_encoder(inputs, params=self.get_subdict(params, "encoder"))


class DynamicResNet(MetaModule):
    def __init__(self):
        super(DynamicResNet, self).__init__()
        self.band_encoder = BandEncoder(out_features=32)
        self.feature_model = ResNet(inplanes=32, out_features=1, normtype="instancenorm", avg_pool=True)

    def forward(self, inputs, params=None):
        features = self.band_encoder(inputs, params=self.get_subdict(params, "band_encoder"))
        return self.feature_model(features, params=self.get_subdict(params, "feature_model"))


def conv3x3(in_channels, out_channels, no_batchnorm=False, track_running_stats=False, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1.,
                        track_running_stats=track_running_stats) if not no_batchnorm else nn.InstanceNorm2d(
            out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, input_size=128, hidden_size=64, inner_update_lr_init=None,
                 no_batchnorm=False, track_running_stats=False):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size, no_batchnorm=no_batchnorm, track_running_stats=track_running_stats),
            *[conv3x3(hidden_size, hidden_size, no_batchnorm=no_batchnorm) for _ in
              range(int(math.log2(input_size) - 1))]
        )

        self.classifier = MetaLinear(hidden_size, out_features)

        # additional config parameters of MAML++
        if inner_update_lr_init is not None:
            self.learning_rates = self.add_learning_rate_parameters(inner_update_lr_init)

    """MAML++ feature of learning inner learning rates on the fly"""

    def add_learning_rate_parameters(self, inner_update_lr_init):
        parameter_names = [x[0].replace('.', '-') for x in list(self.named_parameters())]
        return nn.ParameterDict({
            x: torch.nn.Parameter(torch.FloatTensor([inner_update_lr_init]),
                                  requires_grad=True)
            for x in parameter_names
        })

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.mean([2, 3])
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


#################
### ResNet-12 ###
#################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from torch.distributions import Bernoulli

"""
ResNet Code copied from https://github.com/HJ-Yoo/BOIL
"""


def get_subdict(adict, name):
    if adict is None:
        return adict
    tmp = {k[len(name) + 1:]: adict[k] for k in adict if name in k}
    return tmp


class DropBlock(MetaModule):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - \
                                     (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)

            countM = block_mask.size()[0] * block_mask.size()[1] * \
                     block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = torch.nonzero(mask)
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack([
            torch.arange(self.block_size).view(-1, 1). \
                expand(self.block_size, self.block_size).reshape(-1),
            torch.arange(self.block_size).repeat(self.block_size),
        ]).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2). \
                             cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding,
                                       left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1],
                        block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding,
                                       left_padding, right_padding))

        block_mask = 1 - padded_mask
        return block_mask


def get_normlayer(norm, planes):
    if norm == "instancenorm":
        # return nn.GroupNorm(planes, planes)
        return nn.InstanceNorm2d(planes, track_running_stats=False, affine=True)
    elif norm == "layernorm":
        return nn.GroupNorm(1, planes)
    elif "groupnorm" in norm:
        groups = int(norm.replace("groupnorm", ""))
        return nn.GroupNorm(planes // groups, planes)
    elif norm == "tasknorm":
        raise NotImplementedError()
    elif norm == "tbn":
        return MetaBatchNorm2d(planes, track_running_stats=False)
    elif norm == "cbn":
        return MetaBatchNorm2d(planes, track_running_stats=True)
    else:
        raise ValueError(f"wrong norm {norm} specificed")


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, max_padding=0, normtype="tbn"):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)

        self.bn1 = get_normlayer(normtype, planes)

        self.relu1 = nn.LeakyReLU()
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)

        self.bn2 = get_normlayer(normtype, planes)

        self.relu2 = nn.LeakyReLU()
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)

        self.bn3 = get_normlayer(normtype, planes)
        self.relu3 = nn.LeakyReLU()

        self.maxpool = nn.MaxPool2d(stride=stride, kernel_size=[stride, stride],
                                    padding=max_padding)

        self.max_pool = True if stride != max_padding else False
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        residual = x
        out = self.conv1(x, params=get_subdict(params, 'conv1'))
        if isinstance(self.bn1, MetaBatchNorm2d):
            out = self.bn1(out, params=get_subdict(params, 'bn1'))
        else:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out, params=get_subdict(params, 'conv2'))
        if isinstance(self.bn2, MetaBatchNorm2d):
            out = self.bn2(out, params=get_subdict(params, 'bn2'))
        else:
            out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out, params=get_subdict(params, 'conv3'))
        if isinstance(self.bn3, MetaBatchNorm2d):
            out = self.bn3(out, params=get_subdict(params, 'bn3'))
        else:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x, params=get_subdict(params, 'downsample'))
        out += residual
        out = self.relu3(out)

        if self.max_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * \
                                (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / \
                        (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate,
                                training=self.training, inplace=True)

        return out


class ResNet(MetaModule):
    def __init__(self, inplanes=3, keep_prob=1.0, avg_pool=True, drop_rate=0.0,
                 dropblock_size=5, out_features=5, wh_size=1, big_network=False, normtype="tbn"):

        # NOTE  keep_prob < 1 and drop_rate > 0 are NOT supported!

        super(ResNet, self).__init__()
        self.inplanes = inplanes

        blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]

        if big_network:
            num_chn = [64, 160, 320, 640]
        else:
            num_chn = [64, 128, 256, 512]

        self.layer1 = self._make_layer(blocks[0], num_chn[0], stride=2,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=0, normtype=normtype)
        self.layer2 = self._make_layer(blocks[1], num_chn[1], stride=2,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=0, normtype=normtype)
        self.layer3 = self._make_layer(blocks[2], num_chn[2], stride=2,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=1, normtype=normtype)
        self.layer4 = self._make_layer(blocks[3], num_chn[3], stride=1,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=1, normtype=normtype)

        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = MetaLinear(num_chn[-1] * wh_size * wh_size, out_features)
        else:
            self.avgpool = nn.Identity()
            self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
            self.classifier = MetaConv2d(in_channels=num_chn[-1] * wh_size * wh_size, out_channels=1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, MetaLinear):
                nn.init.xavier_uniform_(m.weight)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0,
                    drop_block=False, block_size=1, max_padding=0, normtype="tbn"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            norm = get_normlayer(normtype, planes * block.expansion)

            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=1, bias=False),
                norm,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, drop_rate, drop_block, block_size, max_padding, normtype=normtype))
        self.inplanes = planes * block.expansion
        return MetaSequential(*layers)

    def forward(self, x, params=None):
        x = self.layer1(x, params=get_subdict(params, 'layer1'))
        x = self.layer2(x, params=get_subdict(params, 'layer2'))
        x = self.layer3(x, params=get_subdict(params, 'layer3'))
        x = self.layer4(x, params=get_subdict(params, 'layer4'))
        if self.keep_avg_pool:
            x = self.avgpool(x)
            features = x.view((x.size(0), -1))
        else:
            features = x
        return self.classifier(self.dropout(features),
                               params=get_subdict(params, 'classifier'))
