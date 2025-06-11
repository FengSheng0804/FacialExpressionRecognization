import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import sys

from models.DenseNet.DenseNet import densenet121, densenet169, densenet201, densenet161
from models.DenseNet.DenseNetConfig import DenseNetConfig
from dataset.FaceDataset import FaceDataset

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'

# 表情类别映射
EMOTIONS = {
    0: "生气",
    1: "厌恶",
    2: "恐惧",
    3: "开心",
    4: "难过",
    5: "惊讶",
    6: "平静"
}

def init_wandb(config):
    """初始化wandb"""
    try:
        wandb.init(
            project="facial-expression-recognition",
            name=f"densenet_{config.OPTIMIZER}_lr{config.LEARNING_RATE}",
            config={
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "optimizer": config.OPTIMIZER,
                "model_architecture": "DenseNet",
                "dataset": "FER2013",
                "densenet_type": config.DENSENET_TYPE,
                "growth_rate": config.GROWTH_RATE,
                "reduction": config.REDUCTION,
                "device": str(config.DEVICE)
            },
            settings=wandb.Settings(init_timeout=180)  # 增加超时时间到180秒
        )
        return True
    except Exception as e:
        print(f"警告: wandb初始化失败: {str(e)}")
        print("将继续训练，但不会记录到wandb")
        return False

def load_data(config):
    """加载数据"""
    print("正在加载数据...")
    # 创建数据集
    train_dataset = FaceDataset(config.TRAIN_DATA_PATH, is_train=True)
    val_dataset = FaceDataset(config.VALID_DATA_PATH, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    return train_loader, val_loader

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total  # 改为小数形式
    return val_loss / len(val_loader), accuracy

def test_model(model, val_loader, criterion, device):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    
    confusion_matrix = np.zeros((7, 7), dtype=int)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新混淆矩阵
            for i in range(len(labels)):
                pred_label = predicted[i].item()
                true_label = labels[i].item()
                confusion_matrix[true_label][pred_label] += 1
    
    accuracy = 100. * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print("预测类别 →")
    print("实际类别 ↓")
    print("    ", end="")
    for i in range(7):
        print(f"{EMOTIONS[i]:^8}", end="")
    print()
    
    for i in range(7):
        print(f"{EMOTIONS[i]:4}", end="")
        for j in range(7):
            print(f"{confusion_matrix[i][j]:^8}", end="")
        print()
    
    # 计算每个类别的精确率和召回率
    print("\n每个类别的性能指标:")
    for i in range(7):
        true_positives = confusion_matrix[i][i]
        false_positives = sum(confusion_matrix[j][i] for j in range(7)) - true_positives
        false_negatives = sum(confusion_matrix[i][j] for j in range(7)) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{EMOTIONS[i]}:")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")

def train(config):
    """训练模型"""
    # 初始化wandb
    use_wandb = init_wandb(config)
    
    # 加载数据
    train_loader, val_loader = load_data(config)
    
    # 创建模型
    if config.DENSENET_TYPE == 'densenet121':
        model = densenet121()
    elif config.DENSENET_TYPE == 'densenet169':
        model = densenet169()
    elif config.DENSENET_TYPE == 'densenet201':
        model = densenet201()
    elif config.DENSENET_TYPE == 'densenet161':
        model = densenet161()
    else:
        raise ValueError(f"不支持的DenseNet类型: {config.DENSENET_TYPE}")
    
    # 修改第一层卷积以适应灰度图像
    model.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 2 * config.GROWTH_RATE, 
                           kernel_size=3, padding=1, bias=False)
    
    # 修改最后的全连接层以适应表情分类
    model.linear = nn.Linear(model.linear.in_features, config.OUTPUT_SIZE)
    
    model = model.to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY
        )
    else:  # SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    
    # 学习率调度器
    if config.USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.DECAY_STEP,
            gamma=config.LEARNING_RATE_DECAY
        )
    
    # 创建保存目录
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 训练循环
    best_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 记录模型架构到wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=10)
    
    print("开始训练...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印训练进度
            if (batch_idx + 1) % 10 == 0:
                batch_acc = correct / total
                print(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
                
                # 记录到wandb
                if use_wandb:
                    try:
                        wandb.log({
                            "batch": epoch * len(train_loader) + batch_idx,
                            "batch_loss": loss.item(),
                            "batch_accuracy": batch_acc
                        })
                    except Exception as e:
                        print(f"警告: wandb记录失败: {str(e)}")
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total  # 改为小数形式
        
        # 验证
        val_loss, valid_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # 更新学习率
        if config.USE_LR_SCHEDULER:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'当前学习率: {current_lr:.6f}')
        
        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        
        # 记录到wandb（如果可用）
        if use_wandb:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": valid_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            except Exception as e:
                print(f"警告: wandb记录失败: {str(e)}")
        
        # 打印epoch结果
        print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}')
        
        # 保存最佳模型
        if valid_acc > best_acc:
            print(f"发现更好的模型，已保存到 '{config.BEST_MODEL_PATH}'")
            best_acc = valid_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            
            # 记录最佳模型到wandb
            if use_wandb:
                try:
                    wandb.run.summary["best_accuracy"] = best_acc
                    wandb.run.summary["best_epoch"] = epoch
                except Exception as e:
                    print(f"警告: wandb记录最佳模型失败: {str(e)}")
        
        # 保存最新模型
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    
    print(f"最终模型已保存为 '{config.MODEL_SAVE_PATH}'")
    
    # 记录最终结果到wandb
    if use_wandb:
        try:
            wandb.run.summary["final_accuracy"] = val_accs[-1]
        except Exception as e:
            print(f"警告: wandb记录最终结果失败: {str(e)}")
    
    # 测试最佳模型
    print("\n在验证集上测试最佳模型:")
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    test_model(model, val_loader, criterion, config.DEVICE)
    
    # 完成实验，关闭wandb
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    try:
        config = DenseNetConfig()
        train(config)
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        sys.exit(1)