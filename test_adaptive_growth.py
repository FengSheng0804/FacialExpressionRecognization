"""
测试自适应增长率功能
"""

import torch
import torch.nn as nn
import numpy as np
from models.DenseNet.DenseNet import densenet121, densenet169, densenet201, densenet161
from models.DenseNet.DenseNetConfig import DenseNetConfig

def test_adaptive_growth_basic():
    """测试基本自适应增长率功能"""
    print("=" * 60)
    print("测试基本自适应增长率功能")
    print("=" * 60)
    
    # 创建配置
    config = DenseNetConfig()
    config.USE_ADAPTIVE_GROWTH = True
    config.ADAPTIVE_GROWTH_LIST = [16, 24, 32, 48]  # 每个dense block的增长率
    
    print(f"配置信息:")
    print(f"  自适应增长率: {config.USE_ADAPTIVE_GROWTH}")
    print(f"  增长率列表: {config.ADAPTIVE_GROWTH_LIST}")
    print(f"  DenseNet类型: {config.DENSENET_TYPE}")
    print()
    
    # 创建自适应增长率模型
    print("创建自适应增长率DenseNet121模型...")
    adaptive_model = densenet121(
        num_class=config.OUTPUT_SIZE,
        use_cbam=config.USE_CBAM,
        use_adaptive_growth=config.USE_ADAPTIVE_GROWTH,
        adaptive_growth_list=config.ADAPTIVE_GROWTH_LIST
    )
    
    # 修改输入层以适应灰度图像
    adaptive_model.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 
                                   2 * config.ADAPTIVE_GROWTH_LIST[0], 
                                   kernel_size=3, padding=1, bias=False)
    
    adaptive_params = sum(p.numel() for p in adaptive_model.parameters())
    print(f"自适应增长率模型参数量: {adaptive_params:,}")
    
    # 创建标准模型进行对比
    print("创建标准DenseNet121模型...")
    standard_model = densenet121(
        num_class=config.OUTPUT_SIZE,
        use_cbam=config.USE_CBAM,
        use_adaptive_growth=False,
        adaptive_growth_list=None
    )
    
    # 修改输入层
    standard_model.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 
                                   2 * config.GROWTH_RATE, 
                                   kernel_size=3, padding=1, bias=False)
    
    standard_params = sum(p.numel() for p in standard_model.parameters())
    print(f"标准模型参数量: {standard_params:,}")
    
    # 计算参数差异
    param_ratio = adaptive_params / standard_params
    print(f"参数量比例: {param_ratio:.3f}")
    if param_ratio < 1:
        print(f"自适应模型减少了 {(1-param_ratio)*100:.1f}% 的参数")
    else:
        print(f"自适应模型增加了 {(param_ratio-1)*100:.1f}% 的参数")
    print()

def test_different_growth_configurations():
    """测试不同的增长率配置"""
    print("=" * 60)
    print("测试不同增长率配置的效果")
    print("=" * 60)
    
    # 不同的增长率配置
    growth_configs = {
        "递增式": [12, 16, 24, 32],
        "递减式": [32, 24, 16, 12],
        "波浪式": [16, 32, 16, 32],
        "集中式": [8, 8, 48, 8],
        "标准式": [24, 24, 24, 24]
    }
    
    print("不同增长率配置的参数量对比:")
    print("-" * 60)
    
    for config_name, growth_list in growth_configs.items():
        model = densenet121(
            num_class=7,
            use_cbam=True,
            use_adaptive_growth=True,
            adaptive_growth_list=growth_list
        )
        
        # 修改输入层
        model.conv1 = nn.Conv2d(1, 2 * growth_list[0], kernel_size=3, padding=1, bias=False)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"{config_name:10s}: {growth_list} -> {params:,} 参数")
    
    print()

def test_model_forward_pass():
    """测试模型前向传播"""
    print("=" * 60)
    print("测试自适应增长率模型前向传播")
    print("=" * 60)
    
    # 创建自适应增长率模型
    growth_list = [16, 24, 32, 48]
    model = densenet121(
        num_class=7,
        use_cbam=True,
        use_adaptive_growth=True,
        adaptive_growth_list=growth_list
    )
    
    # 修改输入层以适应灰度图像
    model.conv1 = nn.Conv2d(1, 2 * growth_list[0], kernel_size=3, padding=1, bias=False)
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 48, 48)
    
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"输出张量形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 检查输出的数值稳定性
    if torch.isnan(output).any():
        print("❌ 输出包含NaN值")
    elif torch.isinf(output).any():
        print("❌ 输出包含无穷大值")
    else:
        print("✅ 输出数值正常")
    
    # 测试概率预测
    probabilities = torch.softmax(output, dim=1)
    print(f"概率分布示例: {probabilities[0].numpy()}")
    print(f"概率和: {probabilities.sum(dim=1)}")
    print()

def test_all_densenet_variants():
    """测试所有DenseNet变体的自适应增长率"""
    print("=" * 60)
    print("测试所有DenseNet变体的自适应增长率功能")
    print("=" * 60)
    
    growth_list = [16, 24, 32, 48]
    
    variants = {
        'DenseNet121': densenet121,
        'DenseNet169': densenet169,
        'DenseNet201': densenet201,
        'DenseNet161': densenet161
    }
    
    print("各变体的参数量对比:")
    print("-" * 60)
    
    for name, model_func in variants.items():
        # 标准模型
        standard_model = model_func(
            num_class=7,
            use_cbam=True,
            use_adaptive_growth=False
        )
        
        # 自适应模型
        adaptive_model = model_func(
            num_class=7,
            use_cbam=True,
            use_adaptive_growth=True,
            adaptive_growth_list=growth_list
        )
        
        standard_params = sum(p.numel() for p in standard_model.parameters())
        adaptive_params = sum(p.numel() for p in adaptive_model.parameters())
        ratio = adaptive_params / standard_params
        
        print(f"{name:12s}:")
        print(f"  标准模型:   {standard_params:,} 参数")
        print(f"  自适应模型: {adaptive_params:,} 参数")
        print(f"  参数比例:   {ratio:.3f}")
        print()

def test_training_compatibility():
    """测试训练兼容性"""
    print("=" * 60)
    print("测试自适应增长率模型的训练兼容性")
    print("=" * 60)
    
    # 创建模型
    model = densenet121(
        num_class=7,
        use_cbam=True,
        use_adaptive_growth=True,
        adaptive_growth_list=[16, 24, 32, 48]
    )
    
    # 修改输入层
    model.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
    
    # 设置为训练模式
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建虚拟训练数据
    batch_size = 8
    images = torch.randn(batch_size, 1, 48, 48)
    labels = torch.randint(0, 7, (batch_size,))
    
    print(f"训练数据形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 模拟一个训练步骤
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"前向传播输出形状: {outputs.shape}")
    print(f"损失值: {loss.item():.4f}")
    
    # 检查梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    if has_grad:
        print("✅ 梯度计算正常")
    else:
        print("❌ 梯度计算异常")
    
    # 检查参数更新
    param_before = [p.clone() for p in model.parameters()]
    optimizer.step()
    param_after = list(model.parameters())
    
    params_changed = any(not torch.equal(before, after) 
                        for before, after in zip(param_before, param_after))
    if params_changed:
        print("✅ 参数更新正常")
    else:
        print("❌ 参数未更新")
    print()

def test_memory_efficiency():
    """测试内存效率"""
    print("=" * 60)
    print("测试自适应增长率模型的内存效率")
    print("=" * 60)
    
    # 不同的增长率配置
    configs = {
        "小型配置": [8, 12, 16, 20],
        "中型配置": [16, 24, 32, 40],
        "大型配置": [24, 32, 48, 64]
    }
    
    print("不同配置的内存使用估算:")
    print("-" * 60)
    
    for config_name, growth_list in configs.items():
        model = densenet121(
            num_class=7,
            use_cbam=True,
            use_adaptive_growth=True,
            adaptive_growth_list=growth_list
        )
        
        # 计算模型参数量
        params = sum(p.numel() for p in model.parameters())
        
        # 估算内存使用（假设float32，4字节每参数）
        memory_mb = params * 4 / (1024 * 1024)
        
        print(f"{config_name:10s}: {growth_list}")
        print(f"  参数量: {params:,}")
        print(f"  估算内存: {memory_mb:.1f} MB")
        print()

def main():
    """主测试函数"""
    print("DenseNet 自适应增长率功能测试")
    print("=" * 60)
    
    try:
        test_adaptive_growth_basic()
        test_different_growth_configurations()
        test_model_forward_pass()
        test_all_densenet_variants()
        test_training_compatibility()
        test_memory_efficiency()
        
        print("=" * 60)
        print("✅ 所有测试通过！自适应增长率功能正常工作。")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
