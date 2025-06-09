import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        # 使用更小的标准差初始化权重，防止梯度爆炸
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # 初始化BatchNorm层的权重和偏置
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size, device):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model.forward(images)
        pred = torch.argmax(pred, dim=1).cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

class PReLU(nn.PReLU):
    """PReLU激活函数包装器"""
    def __init__(self):
        super(PReLU, self).__init__(init=0.0)

class FaceCNN(nn.Module):
    """
    根据TensorFlow modelB设计的改进版FaceCNN
    参考结构：
    1. 卷积层(64, 3x3) -> 批归一化 -> 最大池化(3x3, stride=2)
    2. 卷积层(128, 3x3) -> 批归一化 -> 最大池化(3x3, stride=2)
    3. 卷积层(256, 3x3) -> 批归一化 -> 最大池化(3x3, stride=2)
    4. Dropout(keep_prob) -> 全连接层(4096) -> Dropout(keep_prob) -> 全连接层(1024) -> 输出层(7)
    """
    def __init__(self, input_size=48, use_batchnorm=True, activation='relu', keep_prob=0.5):
        super(FaceCNN, self).__init__()
        self.input_size = input_size
        self.use_batchnorm = use_batchnorm
        self.keep_prob = keep_prob
        
        # 激活函数选择
        if activation == 'prelu':
            self.activation = PReLU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # 第一个卷积块
        layers = [
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 批归一化放在激活函数前
            self.activation
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv1 = nn.Sequential(*layers)
        
        # 第二个卷积块
        layers = [
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.activation
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(*layers)
        
        # 第三个卷积块
        layers = [
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.activation
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(nn.Dropout(p=1-keep_prob))
        self.conv3 = nn.Sequential(*layers)
        
        # 计算卷积后的特征图大小
        self._calculate_feature_size()
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 4096),
            self.activation,
            nn.Dropout(p=1-keep_prob),
            nn.Linear(4096, 1024),
            self.activation,
            nn.Linear(1024, 7),
            nn.Softmax(dim=1)
        )
        
        # 参数初始化
        self.apply(gaussian_weights_init)
    
    def _calculate_feature_size(self):
        """计算经过所有卷积和池化层后的特征图大小"""
        # 创建一个假的输入张量来跟踪尺寸变化
        x = torch.zeros(1, 1, self.input_size, self.input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 获取特征图的形状
        self.feature_shape = x.shape
        # 计算特征的总数量（batch_size维度除外）
        self.feature_size = x.shape[1] * x.shape[2] * x.shape[3]

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y
        
    def summary(self):
        """打印模型结构摘要"""
        print(self)
        # 打印特征图大小
        print(f"\n特征图大小: {self.feature_size}")
        print(f"特征图形状: {self.feature_shape}")
        
        # 创建一个假的输入来追踪每层的输出大小
        x = torch.zeros(1, 1, self.input_size, self.input_size)
        print("\n层输出形状:")
        
        x1 = self.conv1(x)
        print(f"conv1 输出: {x1.shape}")
        
        x2 = self.conv2(x1)
        print(f"conv2 输出: {x2.shape}")
        
        x3 = self.conv3(x2)
        print(f"conv3 输出: {x3.shape}")
        
        x3_flat = x3.view(x3.size(0), -1)
        print(f"flatten 输出: {x3_flat.shape}")
        
        y = self.fc(x3_flat)
        print(f"fc 输出: {y.shape}")