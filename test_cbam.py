"""
测试CBAM注意力机制功能
"""
import torch
import torch.nn as nn
import numpy as np
from models.DenseNet.DenseNet import densenet121, CBAM, ChannelAttention, SpatialAttention
from models.DenseNet.DenseNetConfig import DenseNetConfig

def test_channel_attention():
    """测试通道注意力模块"""
    print("=== 测试通道注意力模块 ===")
    
    # 创建测试数据
    batch_size = 4
    channels = 64
    height, width = 16, 16
    
    # 创建通道注意力模块
    ca = ChannelAttention(in_planes=channels, ratio=16)
    
    print(f"输入特征形状: ({batch_size}, {channels}, {height}, {width})")
    print(f"通道缩放比例: 16")
    
    # 测试前向传播
    input_tensor = torch.randn(batch_size, channels, height, width)
    attention_weights = ca(input_tensor)
    
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重范围: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
    print(f"注意力权重均值: {attention_weights.mean().item():.4f}")
    
    # 验证注意力权重在[0,1]范围内
    assert attention_weights.min() >= 0 and attention_weights.max() <= 1, "注意力权重应在[0,1]范围内"
    print("✓ 通道注意力模块测试通过")

def test_spatial_attention():
    """测试空间注意力模块"""
    print("\n=== 测试空间注意力模块 ===")
    
    # 创建测试数据
    batch_size = 4
    channels = 64
    height, width = 16, 16
    
    # 创建空间注意力模块
    sa = SpatialAttention(kernel_size=7)
    
    print(f"输入特征形状: ({batch_size}, {channels}, {height}, {width})")
    print(f"卷积核大小: 7")
    
    # 测试前向传播
    input_tensor = torch.randn(batch_size, channels, height, width)
    attention_weights = sa(input_tensor)
    
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重范围: [{attention_weights.min().item():.4f}, {attention_weights.max().item():.4f}]")
    print(f"注意力权重均值: {attention_weights.mean().item():.4f}")
    
    # 验证输出形状
    expected_shape = (batch_size, 1, height, width)
    assert attention_weights.shape == expected_shape, f"输出形状应为{expected_shape}"
    
    # 验证注意力权重在[0,1]范围内
    assert attention_weights.min() >= 0 and attention_weights.max() <= 1, "注意力权重应在[0,1]范围内"
    print("✓ 空间注意力模块测试通过")

def test_cbam_module():
    """测试完整的CBAM模块"""
    print("\n=== 测试CBAM模块 ===")
    
    # 创建测试数据
    batch_size = 4
    channels = 64
    height, width = 16, 16
    
    # 创建CBAM模块
    cbam = CBAM(in_planes=channels, ratio=16, kernel_size=7)
    
    print(f"输入特征形状: ({batch_size}, {channels}, {height}, {width})")
    print(f"通道缩放比例: 16")
    print(f"空间卷积核大小: 7")
    
    # 测试前向传播
    input_tensor = torch.randn(batch_size, channels, height, width)
    output_tensor = cbam(input_tensor)
    
    print(f"输出特征形状: {output_tensor.shape}")
    print(f"输入输出形状是否相同: {input_tensor.shape == output_tensor.shape}")
    
    # 计算注意力效果
    input_mean = input_tensor.mean().item()
    output_mean = output_tensor.mean().item()
    print(f"输入特征均值: {input_mean:.4f}")
    print(f"输出特征均值: {output_mean:.4f}")
    print(f"特征变化比例: {(output_mean - input_mean) / input_mean * 100:.2f}%")
    
    # 验证输出形状
    assert output_tensor.shape == input_tensor.shape, "CBAM输出形状应与输入相同"
    print("✓ CBAM模块测试通过")

def test_densenet_with_cbam():
    """测试带CBAM的DenseNet模型"""
    print("\n=== 测试DenseNet + CBAM ===")
    
    config = DenseNetConfig()
    
    # 创建带CBAM的模型
    model_with_cbam = densenet121(
        num_class=7,
        use_cbam=True,
        use_dynamic_reuse=False  # 专注测试CBAM
    )
    
    # 创建不带CBAM的模型
    model_without_cbam = densenet121(
        num_class=7,
        use_cbam=False,
        use_dynamic_reuse=False
    )
    
    # 计算参数量
    params_with_cbam = sum(p.numel() for p in model_with_cbam.parameters())
    params_without_cbam = sum(p.numel() for p in model_without_cbam.parameters())
    
    print(f"不带CBAM的参数量: {params_without_cbam:,}")
    print(f"带CBAM的参数量: {params_with_cbam:,}")
    print(f"CBAM参数增加量: {params_with_cbam - params_without_cbam:,}")
    print(f"参数增加比例: {(params_with_cbam - params_without_cbam) / params_without_cbam * 100:.2f}%")
    
    # 测试前向传播
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 48, 48)  # 灰度图像
    
    print(f"\n测试输入形状: {test_input.shape}")
    
    # 测试带CBAM的模型
    with torch.no_grad():
        output_with_cbam = model_with_cbam(test_input)
    print(f"带CBAM输出形状: {output_with_cbam.shape}")
    print(f"带CBAM输出范围: [{output_with_cbam.min().item():.4f}, {output_with_cbam.max().item():.4f}]")
    
    # 测试不带CBAM的模型
    with torch.no_grad():
        output_without_cbam = model_without_cbam(test_input)
    print(f"不带CBAM输出形状: {output_without_cbam.shape}")
    print(f"不带CBAM输出范围: [{output_without_cbam.min().item():.4f}, {output_without_cbam.max().item():.4f}]")
    
    # 比较输出差异
    output_diff = torch.abs(output_with_cbam - output_without_cbam).mean().item()
    print(f"输出差异均值: {output_diff:.4f}")
    
    print("✓ DenseNet CBAM测试通过")

def test_cbam_parameters():
    """测试不同CBAM参数的影响"""
    print("\n=== 测试不同CBAM参数 ===")
    
    configs = [
        {"ratio": 8, "kernel_size": 3, "name": "ratio_8_kernel_3"},
        {"ratio": 16, "kernel_size": 7, "name": "ratio_16_kernel_7"},
        {"ratio": 32, "kernel_size": 3, "name": "ratio_32_kernel_3"},
        {"ratio": 16, "kernel_size": 3, "name": "ratio_16_kernel_3"},
    ]
    
    test_input = torch.randn(1, 1, 48, 48)
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        # 创建CBAM模块
        cbam = CBAM(in_planes=64, ratio=config['ratio'], kernel_size=config['kernel_size'])
        
        # 计算参数量
        params = sum(p.numel() for p in cbam.parameters())
        print(f"CBAM参数量: {params:,}")
        
        # 测试前向传播
        test_features = torch.randn(1, 64, 24, 24)
        with torch.no_grad():
            output = cbam(test_features)
        
        print(f"输入形状: {test_features.shape}")
        print(f"输出形状: {output.shape}")
        print(f"特征变化范围: [{(output - test_features).min().item():.4f}, {(output - test_features).max().item():.4f}]")

def test_cbam_attention_visualization():
    """测试CBAM注意力可视化数据"""
    print("\n=== 测试CBAM注意力权重 ===")
    
    # 创建CBAM模块
    cbam = CBAM(in_planes=32, ratio=16, kernel_size=7)
    
    # 创建测试输入
    test_input = torch.randn(1, 32, 16, 16)
    
    # 分别获取通道和空间注意力权重
    with torch.no_grad():
        # 通道注意力权重
        channel_weights = cbam.ca(test_input)
        print(f"通道注意力权重形状: {channel_weights.shape}")
        print(f"通道注意力统计: 均值={channel_weights.mean().item():.4f}, 标准差={channel_weights.std().item():.4f}")
        
        # 应用通道注意力后的特征
        channel_attended = test_input * channel_weights
        
        # 空间注意力权重
        spatial_weights = cbam.sa(channel_attended)
        print(f"空间注意力权重形状: {spatial_weights.shape}")
        print(f"空间注意力统计: 均值={spatial_weights.mean().item():.4f}, 标准差={spatial_weights.std().item():.4f}")
        
        # 最终输出
        final_output = channel_attended * spatial_weights
        
        # 分析注意力效果
        original_std = test_input.std().item()
        final_std = final_output.std().item()
        print(f"原始特征标准差: {original_std:.4f}")
        print(f"CBAM处理后标准差: {final_std:.4f}")
        print(f"特征区分度变化: {(final_std - original_std) / original_std * 100:.2f}%")
        
    print("✓ CBAM注意力可视化测试通过")

if __name__ == "__main__":
    print("开始测试CBAM注意力机制功能...\n")
    
    try:
        # 测试各个组件
        test_channel_attention()
        test_spatial_attention()
        test_cbam_module()
        
        # 测试完整模型
        test_densenet_with_cbam()
        
        # 测试不同参数配置
        test_cbam_parameters()
        
        # 测试注意力可视化
        test_cbam_attention_visualization()
        
        print("\n🎉 所有CBAM测试都通过了！")
        print("CBAM注意力机制已成功集成到DenseNet中。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
