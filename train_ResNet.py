import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from dataset.FaceDataset import FaceDataset
from models.ResNet.ResNet import ResNet50, ResNet101, ResNet152
from models.ResNet.ResNetConfig import ResNetConfig

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'

# 设置随机种子，保证结果可复现
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

# 表情标签映射
NUM_CLASSES = 7  # 表情类别数：生气、厌恶、恐惧、开心、伤心、惊讶、中性
EMOTIONS = {
    0: '生气', 
    1: '厌恶', 
    2: '恐惧', 
    3: '开心', 
    4: '伤心', 
    5: '惊讶', 
    6: '中性'
}

# 初始化wandb
def init_wandb(config, model_name):
    """初始化wandb，设置项目名称和配置参数"""
    wandb.init(
        project="facial-expression-recognition",
        settings=wandb.Settings(init_timeout=180),
        config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "optimizer": config.OPTIMIZER,
            "model_architecture": model_name,
            "dataset": "FER2013",
            "resnet_type": config.RESNET_TYPE,
            "pretrained": config.PRETRAINED,
            "use_bottleneck": config.USE_BOTTLENECK,
            "device": str(config.DEVICE)
        }
    )
    
    wandb.run.name = f"{model_name}_{config.OPTIMIZER}_lr{config.LEARNING_RATE}"
    return wandb.config

# 加载数据集
def load_data(config):
    print("正在加载数据...")
    train_dataset = FaceDataset(config.TRAIN_DATA_PATH, is_train=True)
    valid_dataset = FaceDataset(config.VALID_DATA_PATH, is_train=False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(valid_dataset)}")
    return train_loader, valid_dataset  # 返回valid_dataset而不是valid_loader

def validate(model, dataset, batch_size, device):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        pred = model(images)
        pred = torch.argmax(pred, dim=1).cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

# 训练模型
def train(model, train_loader, valid_dataset, config):
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                             betas=(config.BETA1, config.BETA2))
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, 
                            momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    if config.USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.DECAY_STEP, 
                                           gamma=config.LEARNING_RATE_DECAY)
    
    train_losses = []
    valid_accs = []
    
    patience = 20
    best_acc = 0.0
    patience_counter = 0
    
    wandb.watch(model, log="all", log_freq=10)
    
    print("开始训练...")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss).any():
                print(f"警告: 第{epoch+1}轮第{i+1}步出现NaN损失，跳过此批次")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                batch_acc = correct / total
                print(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
                
                wandb.log({
                    "batch": epoch * len(train_loader) + i,
                    "batch_loss": loss.item(),
                    "batch_accuracy": batch_acc
                })
        
        if config.USE_LR_SCHEDULER:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'当前学习率: {current_lr:.6f}')
        
        if running_loss > 0:
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            train_acc = correct / total
            
            model.eval()
            valid_acc = validate(model, valid_dataset, config.BATCH_SIZE, config.DEVICE)
            valid_accs.append(valid_acc)
            
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}')
            
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": train_acc,
                "val_accuracy": valid_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if valid_acc > best_acc:
                best_acc = valid_acc
                patience_counter = 0
                os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                print(f"发现更好的模型，已保存到 '{config.BEST_MODEL_PATH}'")
                
                wandb.run.summary["best_accuracy"] = best_acc
                wandb.run.summary["best_epoch"] = epoch
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"连续{patience}个epoch没有改善，提前停止训练")
                break
        else:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], 所有批次损失均为NaN，跳过此轮')
    
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"最终模型已保存为 '{config.MODEL_SAVE_PATH}'")
    
    wandb.run.summary["final_accuracy"] = valid_accs[-1]
    
    return train_losses, valid_accs

def main():
    # 创建配置对象
    config = ResNetConfig()
    
    # 根据配置选择ResNet类型
    if config.RESNET_TYPE == 'resnet50':
        model = ResNet50(num_classes=config.OUTPUT_SIZE)
    elif config.RESNET_TYPE == 'resnet101':
        model = ResNet101(num_classes=config.OUTPUT_SIZE)
    elif config.RESNET_TYPE == 'resnet152':
        model = ResNet152(num_classes=config.OUTPUT_SIZE)
    else:
        raise ValueError(f"不支持的ResNet类型: {config.RESNET_TYPE}")
    
    # 初始化wandb
    init_wandb(config, config.RESNET_TYPE.upper())
    
    # 加载数据
    train_loader, valid_dataset = load_data(config)  # 修改这里，接收valid_dataset
    
    # 训练模型
    train_losses, valid_accs = train(model, train_loader, valid_dataset, config)
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    main() 