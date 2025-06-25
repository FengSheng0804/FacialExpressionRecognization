"""
知识蒸馏实现：使用DenseNet201作为教师模型对DenseNet121学生模型进行知识蒸馏
整合了配置、工具函数和主要训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import os
import sys
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import OrderedDict

# wandb导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Training metrics will not be logged to wandb.")

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from models.DenseNet.DenseNet import densenet121
from dataset.FaceDataset import FaceDataset


class DistillationConfig:
    """知识蒸馏的配置类"""
    
    def __init__(self):
        # 基本训练参数
        self.BATCH_SIZE = 64  # 较小的批次大小用于蒸馏
        self.EPOCHS = 30  # 蒸馏轮数较少
        self.LEARNING_RATE = 0.00005  # 更小的学习率，避免震荡
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优化器参数
        self.OPTIMIZER = 'adam'
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.WEIGHT_DECAY = 1e-4
        
        # 学习率调度器
        self.USE_LR_SCHEDULER = True
        self.LEARNING_RATE_DECAY = 0.8  # 更平缓的衰减
        self.DECAY_STEP = 5  # 更频繁的调整
        
        # 知识蒸馏特定参数
        self.ALPHA = 0.5  # 降低蒸馏损失权重，更注重标签准确性
        self.TEMPERATURE = 3.0  # 降低温度，保持更多细节信息
        
        # 梯度裁剪（防止梯度爆炸）
        self.GRAD_CLIP = 1.0
        
        # 预热策略
        self.WARMUP_EPOCHS = 2
        self.WARMUP_LR = 0.00001
        
        # 模型参数
        self.INPUT_CHANNELS = 1
        self.IMAGE_SIZE = 48
        self.OUTPUT_SIZE = 7
        
        # 学生模型参数（DenseNet121）
        self.STUDENT_TYPE = 'densenet121'
        self.USE_CBAM = True
        self.USE_ADAPTIVE_GROWTH = True
        self.ADAPTIVE_GROWTH_LIST = [16, 24, 32, 48]
        self.GROWTH_RATE = 32
        
        # 数据路径（相对于项目根目录）
        self.TRAIN_DATA_PATH = "dataset/train_set"
        self.VALID_DATA_PATH = "dataset/verify_set"
        
        # 模型路径（相对于项目根目录）
        self.TEACHER_MODEL_PATH = "models/DenseNet/model_weight/densenet201.pth"
        self.STUDENT_MODEL_PATH = "models/DenseNet/model_weight/best_facial_expression_model_densenet121_cbam_adaptive_growth.pth"
        self.DISTILLED_MODEL_PATH = "models/DenseNet/model_weight/best_facial_expression_model_densenet121_cbam_adaptive_growth_distilled.pth"

        # 性能优化
        self.PIN_MEMORY = True
        self.NUM_WORKERS = 4
        
        # 早停机制
        self.PATIENCE = 10
        
        # 日志输出频率
        self.LOG_INTERVAL = 20


def calculate_model_size(model):
    """
    计算模型的参数量和大小
    
    Args:
        model: PyTorch模型
    
    Returns:
        dict: 包含参数量和大小信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小（MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb
    }


def evaluate_model(model, data_loader, device, criterion=None):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(data_loader) if criterion else None
    
    return accuracy, avg_loss, all_predictions, all_labels


def compare_models(teacher_model, student_model, data_loader, device):
    """比较教师模型和学生模型的性能"""
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    
    # 计算模型大小
    teacher_info = calculate_model_size(teacher_model)
    student_info = calculate_model_size(student_model)
    
    print(f"教师模型（DenseNet201）:")
    print(f"  参数量: {teacher_info['total_params']:,}")
    print(f"  模型大小: {teacher_info['size_mb']:.2f} MB")
    
    print(f"\n学生模型（DenseNet121）:")
    print(f"  参数量: {student_info['total_params']:,}")
    print(f"  模型大小: {student_info['size_mb']:.2f} MB")
    
    print(f"\n压缩比:")
    print(f"  参数压缩: {student_info['total_params']/teacher_info['total_params']:.3f}")
    print(f"  大小压缩: {student_info['size_mb']/teacher_info['size_mb']:.3f}")
    
    # 评估性能
    if data_loader and len(data_loader) > 0:
        criterion = nn.CrossEntropyLoss()
        
        teacher_acc, teacher_loss, _, _ = evaluate_model(teacher_model, data_loader, device, criterion)
        student_acc, student_loss, _, _ = evaluate_model(student_model, data_loader, device, criterion)
        
        print(f"\n性能对比:")
        print(f"  教师模型准确率: {teacher_acc:.4f}")
        print(f"  学生模型准确率: {student_acc:.4f}")
        print(f"  性能保持率: {student_acc/teacher_acc:.3f}")
        
        return {
            'teacher': {'accuracy': teacher_acc, 'loss': teacher_loss, 'info': teacher_info},
            'student': {'accuracy': student_acc, 'loss': student_loss, 'info': student_info}
        }
    
    return {
        'teacher': {'info': teacher_info},
        'student': {'info': student_info}
    }


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs, val_accs, 'r-', label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 验证准确率趋势
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_accs, 'g-', label='验证准确率')
    plt.title('验证准确率趋势')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    else:
        plt.show()


def save_distillation_results(results, save_dir="results/distillation_analysis"):
    """保存蒸馏结果分析"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型对比结果
    # 将torch.device转换为字符串以便JSON序列化
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, dict):
                    serializable_results[key][k] = {
                        sub_k: float(sub_v) if isinstance(sub_v, (int, float, np.number)) else str(sub_v)
                        for sub_k, sub_v in v.items()
                    }
                else:
                    serializable_results[key][k] = float(v) if isinstance(v, (int, float, np.number)) else str(v)
        else:
            serializable_results[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
    
    with open(os.path.join(save_dir, "distillation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"蒸馏结果已保存到: {save_dir}")


def load_teacher_model(model_path, device, num_classes=7):
    """
    加载教师模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        num_classes: 分类数量
    
    Returns:
        加载好的教师模型
    """
    print(f"正在加载教师模型：{model_path}")
    
    # 使用torchvision的DenseNet201
    try:
        # 创建DenseNet201模型
        teacher_model = models.densenet201(pretrained=False)
        
        # 修改第一层以适应灰度图像
        original_conv0 = teacher_model.features.conv0
        teacher_model.features.conv0 = nn.Conv2d(
            1, original_conv0.out_channels, 
            kernel_size=original_conv0.kernel_size, 
            stride=original_conv0.stride, 
            padding=original_conv0.padding, 
            bias=original_conv0.bias is not None
        )
        
        # 修改分类器以适应表情分类
        teacher_model.classifier = nn.Linear(
            teacher_model.classifier.in_features, num_classes
        )
        
        # 如果模型文件存在，尝试加载权重
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                
                # 如果是RGB预训练权重，转换第一层卷积权重
                if 'features.conv0.weight' in state_dict:
                    original_weight = state_dict['features.conv0.weight']
                    if original_weight.shape[1] == 3:  # RGB权重
                        # 转换为灰度权重（取均值）
                        gray_weight = original_weight.mean(dim=1, keepdim=True)
                        teacher_model.features.conv0.weight.data = gray_weight
                        print("成功将RGB预训练权重转换为灰度权重")
                
                # 加载其他权重（除了第一层卷积和分类层）
                model_dict = teacher_model.state_dict()
                filtered_dict = {}
                
                for k, v in state_dict.items():
                    if k in model_dict and k != 'features.conv0.weight':
                        if model_dict[k].shape == v.shape:
                            filtered_dict[k] = v
                        else:
                            print(f"跳过形状不匹配的参数: {k}")
                
                # 跳过分类层权重（因为类别数可能不同）
                if 'classifier.weight' in filtered_dict:
                    if filtered_dict['classifier.weight'].shape[0] != num_classes:
                        print(f"分类层维度不匹配，跳过分类层权重")
                        del filtered_dict['classifier.weight']
                        if 'classifier.bias' in filtered_dict:
                            del filtered_dict['classifier.bias']
                
                missing_keys, unexpected_keys = teacher_model.load_state_dict(filtered_dict, strict=False)
                
                if missing_keys:
                    print(f"缺失的键数量: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"意外的键数量: {len(unexpected_keys)}")
                
                print("教师模型权重加载成功（使用torchvision）")
                
            except Exception as e:
                print(f"加载教师模型权重失败: {e}")
                print("使用随机初始化的教师模型")
        else:
            print(f"教师模型文件不存在: {model_path}")
            print("使用随机初始化的教师模型")
        
        # 重新初始化修改的层
        nn.init.kaiming_normal_(teacher_model.features.conv0.weight)
        nn.init.normal_(teacher_model.classifier.weight, 0, 0.01)
        nn.init.constant_(teacher_model.classifier.bias, 0)
        
        # 移动到指定设备
        teacher_model.to(device)
        teacher_model.eval()
        
        # 冻结参数
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        return teacher_model
        
    except ImportError:
        print("torchvision不可用，无法创建教师模型")
        return None


def create_teacher_from_torchvision(pretrained=True, num_classes=7, device='cpu'):
    """
    使用torchvision创建教师模型
    
    Args:
        pretrained: 是否使用预训练权重
        num_classes: 分类数量
        device: 设备
    
    Returns:
        教师模型
    """
    print("使用torchvision创建DenseNet201教师模型...")
    
    # 创建预训练的DenseNet201
    teacher_model = models.densenet201(pretrained=pretrained)
    
    # 修改第一层卷积以适应灰度输入
    original_conv0 = teacher_model.features.conv0
    teacher_model.features.conv0 = nn.Conv2d(
        1, original_conv0.out_channels,
        kernel_size=original_conv0.kernel_size,
        stride=original_conv0.stride,
        padding=original_conv0.padding,
        bias=original_conv0.bias is not None
    )
    
    # 如果使用预训练权重，需要转换第一层
    if pretrained:
        with torch.no_grad():
            # 将RGB权重转换为灰度权重（取均值）
            rgb_weight = original_conv0.weight
            gray_weight = rgb_weight.mean(dim=1, keepdim=True)
            teacher_model.features.conv0.weight.copy_(gray_weight)
            print("成功将RGB预训练权重转换为灰度权重")
    
    # 修改分类器
    teacher_model.classifier = nn.Linear(
        teacher_model.classifier.in_features, num_classes
    )
    
    # 移动到设备并设置为评估模式
    teacher_model.to(device)
    teacher_model.eval()
    
    # 冻结参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    print("教师模型创建成功")
    return teacher_model


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    结合交叉熵损失和KL散度损失
    """
    def __init__(self, alpha=0.5, temperature=3.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha  # 蒸馏损失权重
        self.temperature = temperature  # 温度参数
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, teacher_outputs, labels):
        # 标准交叉熵损失
        ce_loss = self.ce_loss(student_outputs, labels)
        
        # 知识蒸馏损失
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        kd_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 总损失
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        
        return total_loss, ce_loss, kd_loss


def load_student_model(student_path, config, device):
    """加载学生模型"""
    print("正在加载学生模型...")
    student_model = densenet121(
        num_class=config.OUTPUT_SIZE,
        use_cbam=config.USE_CBAM,
        use_adaptive_growth=config.USE_ADAPTIVE_GROWTH,
        adaptive_growth_list=config.ADAPTIVE_GROWTH_LIST
    )
    
    # 修改第一层卷积以适应灰度图像
    if config.USE_ADAPTIVE_GROWTH and config.ADAPTIVE_GROWTH_LIST:
        first_growth = config.ADAPTIVE_GROWTH_LIST[0]
    else:
        first_growth = config.GROWTH_RATE
    student_model.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 2 * first_growth, 
                                  kernel_size=3, padding=1, bias=False)
    
    # 如果学生模型文件存在，加载权重
    if os.path.exists(student_path):
        try:
            state_dict = torch.load(student_path, map_location=device)
            student_model.load_state_dict(state_dict, strict=False)
            print(f"成功加载学生模型权重：{student_path}")
        except Exception as e:
            print(f"无法加载学生模型权重：{e}")
            print("将使用随机初始化的学生模型")
    else:
        print(f"学生模型文件不存在：{student_path}")
        print("将使用随机初始化的学生模型")
    
    student_model.to(device)
    return student_model


def knowledge_distillation_train(config):
    """知识蒸馏训练主函数"""
    
    device = config.DEVICE
    print(f"使用设备: {device}")
    
    # 教师模型和学生模型路径
    teacher_path = config.TEACHER_MODEL_PATH
    student_path = config.STUDENT_MODEL_PATH
    
    # 检查文件是否存在
    print(f"教师模型路径: {teacher_path}")
    print(f"学生模型路径: {student_path}")
    
    # 尝试使用多种方式加载教师模型
    teacher_model = None
    
    # 方法1：尝试从指定路径加载
    if os.path.exists(teacher_path):
        teacher_model = load_teacher_model(teacher_path, device)
    else:
        print(f"指定的教师模型路径不存在: {teacher_path}")
        
        # 方法2：尝试使用torchvision预训练模型
        try:
            print("尝试使用torchvision预训练DenseNet201...")
            teacher_model = create_teacher_from_torchvision(
                pretrained=True, num_classes=config.OUTPUT_SIZE, device=device
            )
        except Exception as e:
            print(f"torchvision加载失败: {e}")
            
            # 方法3：使用随机初始化的教师模型
            print("使用随机初始化的DenseNet201教师模型")
            teacher_model = load_teacher_model("", device)
    
    if teacher_model is None:
        raise RuntimeError("无法创建教师模型")
    
    # 加载学生模型
    student_model = load_student_model(student_path, config, device)
    
    # 打印模型信息
    teacher_info = calculate_model_size(teacher_model)
    student_info = calculate_model_size(student_model)
    
    print(f"\n模型参数量对比：")
    print(f"教师模型（DenseNet201）: {teacher_info['total_params']:,} 参数, {teacher_info['size_mb']:.2f} MB")
    print(f"学生模型（DenseNet121）: {student_info['total_params']:,} 参数, {student_info['size_mb']:.2f} MB")
    print(f"参数压缩比: {student_info['total_params']/teacher_info['total_params']:.3f}")
    
    # 准备数据
    print("\n正在加载训练数据...")
    train_dataset = FaceDataset(config.TRAIN_DATA_PATH, config.IMAGE_SIZE)
    val_dataset = FaceDataset(config.VALID_DATA_PATH, config.IMAGE_SIZE)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        pin_memory=config.PIN_MEMORY,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 设置损失函数和优化器
    distillation_loss = KnowledgeDistillationLoss(
        alpha=config.ALPHA, 
        temperature=config.TEMPERATURE
    )
    
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    if config.USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.DECAY_STEP,
            gamma=config.LEARNING_RATE_DECAY
        )
    
    # 训练参数
    best_acc = 0
    patience_counter = 0
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n开始知识蒸馏训练（{config.EPOCHS}轮）...")
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # 训练阶段
        student_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            student_outputs = student_model(images)
            
            # 计算损失
            total_loss, ce_loss, kd_loss = distillation_loss(
                student_outputs, teacher_outputs, labels
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            if hasattr(config, 'GRAD_CLIP') and config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.GRAD_CLIP)
            
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item()
            _, predicted = student_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()            # 打印进度和记录到wandb
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                batch_acc = correct / total
                print(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Total: {total_loss.item():.4f}, CE: {ce_loss.item():.4f}, '
                      f'KD: {kd_loss.item():.4f}, Acc: {batch_acc:.4f}')
                
                # 记录batch级别的指标到wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        'batch_total_loss': total_loss.item(),
                        'batch_ce_loss': ce_loss.item(),
                        'batch_kd_loss': kd_loss.item(),
                        'batch_accuracy': batch_acc,
                        'batch_step': epoch * len(train_loader) + batch_idx + 1
                    })
        
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # 验证阶段
        student_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                
                total_loss, _, _ = distillation_loss(
                    student_outputs, teacher_outputs, labels
                )
                
                val_loss += total_loss.item()
                _, predicted = student_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': current_lr,
                'best_accuracy': best_acc
            })
        
        # 更新学习率
        if config.USE_LR_SCHEDULER:
            scheduler.step()
        
        print(f'Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # 保存蒸馏后的学生模型
            torch.save(student_model.state_dict(), config.DISTILLED_MODEL_PATH)
            print(f"发现更好的模型，已保存到 '{config.DISTILLED_MODEL_PATH}'")
            
            # 上传最佳模型到wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    # 方法1: 使用wandb.save()保存模型文件
                    wandb.save(config.DISTILLED_MODEL_PATH)
                    
                    # 方法2: 使用Artifact保存模型（更推荐）
                    artifact = wandb.Artifact(
                        name=f"best_distilled_model_epoch_{epoch+1}",
                        type="model",
                        description=f"Best distilled DenseNet121 model at epoch {epoch+1} with validation accuracy {val_acc:.4f}"
                    )
                    artifact.add_file(config.DISTILLED_MODEL_PATH)
                    wandb.log_artifact(artifact)
                    
                    print(f"最佳模型已上传到wandb")
                except Exception as e:
                    print(f"wandb模型上传失败: {e}")
        else:
            patience_counter += 1
          # 早停机制
        if patience_counter >= config.PATIENCE:
            print(f"连续{config.PATIENCE}个epoch没有改善，提前停止训练")
            break
    
    training_time = time.time() - start_time
    
    # 记录最终训练摘要到wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        # 记录最终指标
        wandb.log({
            'final_best_accuracy': best_acc,
            'total_training_time': training_time,
            'total_epochs_trained': min(epoch + 1, config.EPOCHS),
            'early_stopped': patience_counter >= config.PATIENCE
        })
        
        # 记录训练摘要表格
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Best Validation Accuracy", f"{best_acc:.4f}"],
                ["Total Training Time (seconds)", f"{training_time:.2f}"],
                ["Total Epochs Trained", min(epoch + 1, config.EPOCHS)],
                ["Early Stopped", patience_counter >= config.PATIENCE],
                ["Final Learning Rate", optimizer.param_groups[0]['lr']],
                ["Teacher Model", "DenseNet201"],
                ["Student Model", "DenseNet121 + CBAM + Adaptive Growth"],
                ["Distillation Alpha", config.ALPHA],
                ["Distillation Temperature", config.TEMPERATURE]
            ]
        )
        wandb.log({"training_summary": summary_table})
        
        # 上传训练历史图表（如果生成成功）
        try:
            if os.path.exists("results/distillation_analysis/training_history.png"):
                wandb.log({"training_curves": wandb.Image("results/distillation_analysis/training_history.png")})
        except Exception as e:
            print(f"wandb图表上传失败: {e}")
    
    print(f"\n知识蒸馏训练完成！")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"最佳验证精度: {best_acc:.4f}")
    print(f"蒸馏后的模型已保存为: {config.DISTILLED_MODEL_PATH}")
    
    # 绘制训练历史
    try:
        plot_training_history(
            train_losses, val_losses, train_accs, val_accs,
            save_path="results/distillation_analysis/training_history.png"
        )
    except Exception as e:
        print(f"绘制训练历史失败: {e}")
    
    # 保存结果分析
    try:
        results = compare_models(teacher_model, student_model, val_loader, device)
        results['best_accuracy'] = best_acc
        results['training_time'] = training_time
        save_distillation_results(results)
    except Exception as e:
        print(f"保存结果分析失败: {e}")
    
    return best_acc, training_time


def main():
    """主函数"""
    print("=" * 60)
    print("知识蒸馏训练 - DenseNet201 → DenseNet121")
    print("=" * 60)
    
    # 创建配置
    config = DistillationConfig()
    
    # 创建必要的目录
    os.makedirs("models/DenseNet/model_weight", exist_ok=True)
    os.makedirs("results/distillation_analysis", exist_ok=True)
    
    # 开始蒸馏训练
    best_acc, training_time = knowledge_distillation_train(config)
    
    print(f"\n蒸馏训练总结:")
    print(f"最佳精度: {best_acc:.4f}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"蒸馏模型路径: {config.DISTILLED_MODEL_PATH}")


if __name__ == "__main__":
    main()
