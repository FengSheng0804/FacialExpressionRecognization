import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """CBAM中的通道注意力模块"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                 # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)                 # 全局最大池化

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))               # 通过平均池化获得的通道注意力
        maxout = self.sharedMLP(self.max_pool(x))               # 通过最大池化获得的通道注意力
        return self.sigmoid(avgout + maxout)                    # 将两种注意力相加并通过Sigmoid激活函数

class SpatialAttention(nn.Module):
    """CBAM中的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 确保卷积核大小为3或7
        padding = 3 if kernel_size == 7 else 1                      # 设置填充大小

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 对2个通道的特征描述符进行卷积操作
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
        # 先通过通道注意力模块，得到通道加权后的特征图，结果作为空间注意力模块的输入
        out = x * self.ca(x)
        # 再通过空间注意力模块，得到空间加权后的特征图
        result = out * self.sa(out)
        return result

class DenseLayer(nn.Module):
    """DenseNet中的基本层，包含BN-ReLU-Conv1x1-BN-ReLU-Conv3x3的结构"""
    def __init__(self, in_channels, growth_rate, use_cbam=True):
        super().__init__()

        # 在DenseLayer中使用1x1卷积来减少通道数
        # in_channels是输入特征图的通道数，growth_rate是增长率
        # inner_channel是1x1卷积后的通道数，通常设置为4 * growth_rate
        inner_channel = 4 * growth_rate

        self.dense_layer = nn.Sequential(
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
        new_features = self.dense_layer(x)
        if self.use_cbam:
            new_features = self.cbam(new_features)
        return torch.cat([x, new_features], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            # 通过1x1卷积来压缩通道数
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # 通过平均池化来下采样特征图
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

        # 计算输入层的输出通道数，第一层是指输入层
        # 通常设置为2倍的增长率
        inner_channels = 2 * growth_rates[0]
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        # 特征提取部分，使用nn.Sequential来构建DenseNet的特征提取部分
        self.features = nn.Sequential()

        # 依次添加密集块
        for index in range(len(nblocks) - 1):
            # 添加密集块
            self.features.add_module("dense_block_layer_{}".format(index),
                                   self._make_dense_layers(block, inner_channels, nblocks[index], growth_rates[index]))
            # 计算当前密集块的输出通道数，使用+=是因为
            inner_channels += growth_rates[index] * nblocks[index]
            out_channels = int(reduction * inner_channels)
            # 添加过渡层（压缩通道 + 下采样）
            self.features.add_module("transition_layer_{}".format(index),
                                   Transition(inner_channels, out_channels))
            # 更新inner_channels为Transition Layer的输出通道数
            inner_channels = out_channels

        # 最后一个Dense Block，因为没有过渡层，所以不在循环中添加，而是单独处理
        self.features.add_module("dense_block{}".format(len(nblocks) - 1),
                               self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1], growth_rates[-1]))
        # 使用最后一个growth_rate来更新inner_channels
        inner_channels += growth_rates[-1] * nblocks[len(nblocks) - 1]
        # 添加BatchNorm-ReLU层
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        # 添加全局平均池化层，参数为(1, 1)，表示将特征图的空间维度池化到1x1，这样可以将每个通道的特征图压缩为一个数，效果类似于通道注意力的作用
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终的输出通道数为inner_channels，分类数为num_class
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        # 输入x经过卷积层，得到的形状为[batch_size, channels, height, width]
        output = self.conv1(x)
        # 经过特征提取层（Dense Blocks + Transition Layers），得到的形状为[batch_size, channels, height', width']
        output = self.features(output)
        # 经过全局平均池化层，得到的形状为[batch_size, channels, 1, 1]
        output = self.avgpool(output)
        # 将输出展平为一维向量，得到的形状为[batch_size, channels]
        output = output.view(output.size()[0], -1)
        # 最后经过全连接层得到分类结果，得到的形状为[batch_size, num_class]
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks, growth_rate):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('dense_layer_{}'.format(index),
                                 block(in_channels, growth_rate, self.use_cbam))
            in_channels += growth_rate
        return dense_block

def densenet121(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    # 每个Dense Block的层数为[6,12,24,16]，增长率为32
    nblocks = [6,12,24,16]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(DenseLayer, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(DenseLayer, nblocks, growth_rate=32, num_class=num_class, use_cbam=use_cbam)

def densenet169(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    # 每个Dense Block的层数为[6,12,32,32]，增长率为32
    nblocks = [6,12,32,32]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(DenseLayer, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(DenseLayer, nblocks, growth_rate=32, num_class=num_class, use_cbam=use_cbam)

def densenet201(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    # 每个Dense Block的层数为[6,12,48,32]，增长率为32
    nblocks = [6,12,48,32]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(DenseLayer, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(DenseLayer, nblocks, growth_rate=32, num_class=num_class, use_cbam=use_cbam)

def densenet161(num_class=7, use_cbam=True, use_adaptive_growth=False, adaptive_growth_list=None):
    # 每个Dense Block的层数为[6,12,36,24]，增长率为48
    nblocks = [6,12,36,24]
    if use_adaptive_growth and adaptive_growth_list is not None:
        return DenseNet(DenseLayer, nblocks, growth_rate=adaptive_growth_list[0], num_class=num_class, use_cbam=use_cbam, use_adaptive_growth=True, adaptive_growth_list=adaptive_growth_list)
    return DenseNet(DenseLayer, nblocks, growth_rate=48, num_class=num_class, use_cbam=use_cbam)