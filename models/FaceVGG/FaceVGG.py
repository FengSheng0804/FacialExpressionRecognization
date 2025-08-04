import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import torchvision.models as models

# 参数初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

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

class FaceVGG(nn.Module):
    """
    基于VGG架构的人脸表情识别模型
    支持VGG11, VGG13, VGG16, VGG19架构
    可选使用预训练权重
    """
    def __init__(self, input_size=48, input_channels=1, num_classes=7, 
                 vgg_type='vgg16', use_batchnorm=True, pretrained=False, 
                 feature_extract=False, activation='relu', keep_prob=0.5):
        super(FaceVGG, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.vgg_type = vgg_type
        self.use_batchnorm = use_batchnorm
        self.pretrained = pretrained
        self.feature_extract = feature_extract
        self.keep_prob = keep_prob
        
        # 激活函数选择
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # 根据选择的VGG类型获取基础模型
        self.base_model = self._get_base_model()
        
        # 修改第一层以适应输入通道数
        if input_channels != 3:  # VGG默认是3通道输入
            self._modify_first_layer()
            
        # 提取特征部分（卷积层）
        self.features = self.base_model.features
        
        # 计算卷积后的特征图大小
        self._calculate_feature_size()
        
        # 自定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 4096),
            self.activation,
            nn.Dropout(p=1-keep_prob),
            nn.Linear(4096, 1024),
            self.activation,
            nn.Dropout(p=1-keep_prob),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 如果只训练分类器，冻结特征提取部分
        if feature_extract and pretrained:
            self._freeze_features()
            
        # 初始化权重（如果不使用预训练）
        if not pretrained:
            self.apply(weights_init)
    
    def _get_base_model(self):
        """获取基础VGG模型"""
        vgg_models = {
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
        }
        
        if self.vgg_type not in vgg_models:
            raise ValueError(f"不支持的VGG类型: {self.vgg_type}，支持的类型有: {list(vgg_models.keys())}")
        
        # 获取对应的VGG模型，根据需要使用批归一化版本
        if self.use_batchnorm:
            model_fn = getattr(models, f"{self.vgg_type}_bn")
        else:
            model_fn = vgg_models[self.vgg_type]
            
        # 加载模型，根据需要使用预训练权重
        model = model_fn(pretrained=self.pretrained)
        return model
    
    def _modify_first_layer(self):
        """修改第一层卷积以适应输入通道数"""
        # 获取第一层卷积
        first_conv = list(self.base_model.features.children())[0]
        
        # 创建新的卷积层，保持其他参数不变
        new_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # 如果使用预训练权重，尝试调整权重
        if self.pretrained:
            # 对于单通道输入，可以将RGB三个通道的权重平均
            if self.input_channels == 1:
                new_conv.weight.data = torch.mean(first_conv.weight.data, dim=1, keepdim=True)
            # 对于其他情况，使用默认初始化
        
        # 替换第一层
        self.base_model.features[0] = new_conv
    
    def _freeze_features(self):
        """冻结特征提取部分的参数"""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def _calculate_feature_size(self):
        """计算经过所有卷积和池化层后的特征图大小"""
        # 创建一个假的输入张量来跟踪尺寸变化
        x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
        x = self.features(x)
        # 获取特征图的形状
        self.feature_shape = x.shape
        # 计算特征的总数量（batch_size维度除外）
        self.feature_size = x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        """前向传播"""
        # 特征提取
        x = self.features(x)
        # 数据扁平化
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x
    
    def summary(self):
        """打印模型结构摘要"""
        print(self)
        # 打印特征图大小
        print(f"\n特征图大小: {self.feature_size}")
        print(f"特征图形状: {self.feature_shape}")
        
        # 创建一个假的输入来追踪每层的输出大小
        x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
        print("\n层输出形状:")
        
        # 跟踪特征提取部分
        x_features = self.features(x)
        print(f"features 输出: {x_features.shape}")
        
        # 扁平化
        x_flat = x_features.view(x_features.size(0), -1)
        print(f"flatten 输出: {x_flat.shape}")
        
        # 分类器
        y = self.classifier(x_flat)
        print(f"classifier 输出: {y.shape}")
        
        # 打印可训练参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"固定参数数量: {total_params - trainable_params:,}")

# 创建VGG模型的辅助函数
def create_vgg_model(config):
    """根据配置创建VGG模型"""
    model = FaceVGG(
        input_size=config.IMAGE_SIZE,
        input_channels=config.INPUT_CHANNELS,
        num_classes=config.OUTPUT_SIZE,
        vgg_type=config.VGG_TYPE,
        use_batchnorm=config.USE_BATCHNORM,
        pretrained=config.PRETRAINED,
        feature_extract=config.FEATURE_EXTRACT,
        activation=config.ACTIVATION,
        keep_prob=config.KEEP_PROB
    )
    return model 