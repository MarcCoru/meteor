import math

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

def conv3x3(in_channels, out_channels, no_batchnorm=False, track_running_stats=False, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=track_running_stats) if not no_batchnorm else nn.InstanceNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, input_size=128, hidden_size=64, inner_update_lr_init=None, no_batchnorm=False, track_running_stats=False):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size, no_batchnorm=no_batchnorm, track_running_stats=track_running_stats),
            *[conv3x3(hidden_size, hidden_size, no_batchnorm=no_batchnorm) for _ in range(int(math.log2(input_size) - 1))]
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
        features = features.mean([2,3])
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

class UNet(MetaModule):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, params=None):
        x1 = self.inc(x, params=self.get_subdict(params, 'inc'))
        x2 = self.down1(x1, params=self.get_subdict(params, 'down1'))
        x3 = self.down2(x2, params=self.get_subdict(params, 'down2'))
        x4 = self.down3(x3, params=self.get_subdict(params, 'down3'))
        x5 = self.down4(x4, params=self.get_subdict(params, 'down4'))
        x = self.up1(x5, x4, params=self.get_subdict(params, 'up1'))
        x = self.up2(x, x3, params=self.get_subdict(params, 'up2'))
        x = self.up3(x, x2, params=self.get_subdict(params, 'up3'))
        x = self.up4(x, x1, params=self.get_subdict(params, 'up4'))
        logits = self.outc(x, params=self.get_subdict(params, 'outc'))
        return logits


""" Parts of the U-Net model """


class DoubleConv(MetaModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = MetaSequential(
            MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MetaConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, params=None):
        return self.double_conv(x, params=self.get_subdict(params, 'double_conv'))


class Down(MetaModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = MetaSequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x, params=None):
        return self.maxpool_conv(x, params=self.get_subdict(params, 'maxpool_conv'))


class Up(MetaModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, params=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, params=self.get_subdict(params, 'conv'))


class OutConv(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = MetaConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, params=None):
        return self.conv(x, params=self.get_subdict(params, 'conv'))


def prepare_classification_model(nclasses, inplanes=15, resnet=True, norm="tbn", avg_pool=True):
    if resnet:
        assert norm in ["cbm", "bn", "layernorm", "tasknorm"], "specified normalization is not implemented"
        if norm == "cbm": # conventional batch norm
            track_running_stats = True
            layer_norm = False
            tasknorm = False
        elif norm == "tbn": # transductive batch norm (batch norm without trunning statistics at test time)
            track_running_stats = False
            layer_norm = False
            tasknorm = False
        elif norm == "layernorm":
            track_running_stats = False
            layer_norm = True
            tasknorm = False
        elif norm == "tasknorm":
            track_running_stats = False
            layer_norm = False
            tasknorm = True

        model = ResNet(inplanes=inplanes, out_features=nclasses, track_running_stats=track_running_stats, layer_norm=layer_norm, avg_pool=avg_pool, tasknorm=tasknorm)

    else:
        model = ConvolutionalNeuralNetwork(inplanes, nclasses, input_size=128, hidden_size=64,
                                       inner_update_lr_init=None, no_batchnorm=no_batchnorm)
    return model

def prepare_segmentation_model(nclasses):
    return UNet(n_channels=15, n_classes=nclasses)


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
    tmp = {k[len(name) + 1:]:adict[k] for k in adict if name in k}
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

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, max_padding=0, track_running_stats=False, layer_norm=False, tasknorm=False):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        if layer_norm:
            self.bn1 = nn.GroupNorm(planes, planes)
        else:
            self.bn1 = MetaBatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        if layer_norm:
            self.bn2 = nn.GroupNorm(planes, planes)
        else:
            self.bn2 = MetaBatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        if layer_norm:
            self.bn3 = nn.GroupNorm(planes, planes)
        else:
            self.bn3 = MetaBatchNorm2d(planes, track_running_stats=track_running_stats)
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
    def __init__(self, inplanes = 3, keep_prob=1.0, avg_pool=True, drop_rate=0.0,
                 dropblock_size=5, out_features=5, wh_size=1, big_network=False,
                 track_running_stats=False, layer_norm=False, tasknorm=False):

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
                                       block_size=dropblock_size, max_padding=0, track_running_stats=track_running_stats, layer_norm=layer_norm, tasknorm=tasknorm)
        self.layer2 = self._make_layer(blocks[1], num_chn[1], stride=2,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=0, track_running_stats=track_running_stats, layer_norm=layer_norm, tasknorm=tasknorm)
        self.layer3 = self._make_layer(blocks[2], num_chn[2], stride=2,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=1, track_running_stats=track_running_stats, layer_norm=layer_norm, tasknorm=tasknorm)
        self.layer4 = self._make_layer(blocks[3], num_chn[3], stride=1,
                                       drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, max_padding=1, track_running_stats=track_running_stats, layer_norm=layer_norm, tasknorm=tasknorm)

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
                    drop_block=False, block_size=1, max_padding=0, track_running_stats=False, layer_norm=False, tasknorm=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if layer_norm:
                norm = nn.GroupNorm(planes * block.expansion, planes * block.expansion)
            elif tasknorm:
                norm = TaskNormI(planes * block.expansion)
            else:
                norm = MetaBatchNorm2d(planes * block.expansion, track_running_stats=track_running_stats)

            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=1, bias=False),
                norm,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, drop_rate, drop_block, block_size, max_padding, track_running_stats=track_running_stats, layer_norm=layer_norm, tasknorm=tasknorm))
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

