"""
DenseNet201教师模型定义
简化版本，只包含必要的模型结构
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os


def create_densenet201_teacher(num_classes=7, pretrained=True, device='cpu'):
    """
    创建DenseNet201教师模型
    
    Args:
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        device: 设备
    
    Returns:
        DenseNet201教师模型
    """
    print("创建DenseNet201教师模型...")
    
    # 使用torchvision的DenseNet201
    model = models.densenet201(pretrained=pretrained)
    
    # 修改第一层卷积以适应灰度输入
    original_conv0 = model.features.conv0
    model.features.conv0 = nn.Conv2d(
        1, original_conv0.out_channels,
        kernel_size=original_conv0.kernel_size,
        stride=original_conv0.stride,
        padding=original_conv0.padding,
        bias=original_conv0.bias is not None
    )
    
    # 如果使用预训练权重，转换第一层权重
    if pretrained:
        with torch.no_grad():
            rgb_weight = original_conv0.weight
            gray_weight = rgb_weight.mean(dim=1, keepdim=True)
            model.features.conv0.weight.copy_(gray_weight)
            print("成功转换RGB预训练权重为灰度权重")
    
    # 修改分类器
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # 移动到设备
    model.to(device)
    
    return model


def load_teacher_weights(model, weight_path, device='cpu'):
    """
    加载教师模型权重
    
    Args:
        model: 模型实例
        weight_path: 权重文件路径
        device: 设备
    
    Returns:
        加载权重后的模型
    """
    if os.path.exists(weight_path):
        try:
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"成功加载教师模型权重: {weight_path}")
        except Exception as e:
            print(f"加载权重失败: {e}")
            print("使用随机初始化权重")
    else:
        print(f"权重文件不存在: {weight_path}")
        print("使用预训练或随机初始化权重")
    
    return model


if __name__ == "__main__":
    # 测试教师模型创建
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建教师模型
    teacher = create_densenet201_teacher(num_classes=7, pretrained=True, device=device)
    
    # 测试输入
    test_input = torch.randn(2, 1, 48, 48).to(device)
    with torch.no_grad():
        output = teacher(test_input)
    
    print(f"教师模型输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in teacher.parameters()):,}")
