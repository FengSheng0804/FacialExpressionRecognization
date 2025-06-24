import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """CBAM中的通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    """CBAM中的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """结合通道注意力和空间注意力的CBAM模块"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, use_cbam=True):
        super().__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
        # 添加CBAM注意力机制
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(growth_rate)

    def forward(self, x):
        new_features = self.bottle_neck(x)
        if self.use_cbam:
            new_features = self.cbam(new_features)
        return torch.cat([x, new_features], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_adaptive_growth = use_adaptive_growth
        self.adaptive_growth_list = adaptive_growth_list

        # 支持自适应增长率
        if use_adaptive_growth and adaptive_growth_list is not None:
            growth_rates = adaptive_growth_list
        else:
            growth_rates = [growth_rate] * len(nblocks)

        inner_channels = 2 * growth_rates[0]
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index),
                                   self._make_dense_layers(block, inner_channels, nblocks[index], growth_rates[index]))
            inner_channels += growth_rates[index] * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module("transition_layer_{}".format(index),
                                   Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1),
                               self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1], growth_rates[-1]))
        inner_channels += growth_rates[-1] * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks, growth_rate):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index),
                                 block(in_channels, growth_rate, self.use_cbam))
            in_channels += growth_rate
        return dense_block

def densenet121(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    nblocks = [6,12,24,16]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(Bottleneck, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(Bottleneck, nblocks, growth_rate=32, num_class=num_class, use_cbam=use_cbam)

def densenet169(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    nblocks = [6,12,32,32]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(Bottleneck, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(Bottleneck, nblocks, growth_rate=32, num_class=num_class, use_cbam=use_cbam)

def densenet201(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    nblocks = [6,12,48,32]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(Bottleneck, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(Bottleneck, nblocks, growth_rate=32, num_class=num_class, use_cbam=use_cbam)

def densenet161(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    nblocks = [6,12,36,24]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(Bottleneck, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(Bottleneck, nblocks, growth_rate=48, num_class=num_class, use_cbam=use_cbam)