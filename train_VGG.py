import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb  # 导入wandb库
from dataset.FaceDataset import FaceDataset
from models.FaceVGG.FaceVGG import FaceVGG, create_vgg_model
from models.FaceVGG.FaceVGGConfig import FaceVGGConfig

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'

# 设置随机种子，保证结果可复现
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

# 表情标签映射，放在全局位置便于其他模型重复使用
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

# 创建配置对象
config = FaceVGGConfig()
# 调整学习率
config.LEARNING_RATE = 0.0001
# 设置VGG类型
config.VGG_TYPE = 'vgg16'
# 是否使用预训练权重
config.PRETRAINED = False
# 是否只训练分类器部分
config.FEATURE_EXTRACT = False

# 初始化wandb
def init_wandb(config, model_name):
    """初始化wandb，设置项目名称和配置参数"""
    wandb.init(
        project="facial-expression-recognition",  # 项目名称
        settings=wandb.Settings(init_timeout=180),  # 增加超时时间到180秒
        config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "optimizer": config.OPTIMIZER,
            "model_architecture": model_name,
            "dataset": "FER2013",
            "activation": config.ACTIVATION,
            "use_batchnorm": config.USE_BATCHNORM,
            "keep_prob": config.KEEP_PROB,
            "vgg_type": config.VGG_TYPE,
            "pretrained": config.PRETRAINED,
            "feature_extract": config.FEATURE_EXTRACT,
            "device": str(config.DEVICE)
        }
    )
    
    # 设置运行名称
    wandb.run.name = f"{model_name}_{config.VGG_TYPE}_{config.OPTIMIZER}_lr{config.LEARNING_RATE}"
    
    return wandb.config

# 加载数据集
def load_data():
    print("正在加载数据...")
    train_dataset = FaceDataset(config.TRAIN_DATA_PATH, is_train=True)
    valid_dataset = FaceDataset(config.VALID_DATA_PATH, is_train=False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(valid_dataset)}")
    return train_loader, valid_loader

def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        # 将数据移动到与模型相同的设备上
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        pred = model.forward(images)
        pred = torch.argmax(pred, dim=1).cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

# 训练模型
def train(model, train_loader, valid_dataset, epochs, lr):
    model = model.to(config.DEVICE)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 根据配置选择优化器
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(config.BETA1, config.BETA2), 
                              weight_decay=config.WEIGHT_DECAY)
    else:  # 使用SGD优化器
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.MOMENTUM, 
                             weight_decay=config.WEIGHT_DECAY)
    
    # 学习率调度器
    if config.USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.DECAY_STEP, 
                                           gamma=config.LEARNING_RATE_DECAY)
    
    # 记录训练过程的损失和准确率
    train_losses = []
    valid_accs = []
    
    # 早停机制参数
    patience = 20  # 容忍多少个epoch没有改善
    best_acc = 0.0
    patience_counter = 0
    
    # 记录模型架构到wandb
    wandb.watch(model, log="all", log_freq=10)
    
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 检查loss是否为nan
            if torch.isnan(loss).any():
                print(f"警告: 第{epoch+1}轮第{i+1}步出现NaN损失，跳过此批次")
                continue
                
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每10个批次打印一次信息并记录到wandb
            if (i + 1) % 10 == 0:
                batch_acc = correct / total
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
                
                # 记录到wandb
                wandb.log({
                    "batch": epoch * len(train_loader) + i,
                    "batch_loss": loss.item(),
                    "batch_accuracy": batch_acc
                })
        
        # 更新学习率
        if config.USE_LR_SCHEDULER:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'当前学习率: {current_lr:.6f}')
        
        # 计算每个epoch的平均损失和训练准确率
        if running_loss > 0:  # 确保有有效的loss
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            train_acc = correct / total
            
            # 在验证集上评估模型
            model.eval()
            valid_acc = validate(model, valid_dataset, config.BATCH_SIZE)
            valid_accs.append(valid_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}')
            
            # 记录到wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": train_acc,
                "val_accuracy": valid_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            # 早停机制
            if valid_acc > best_acc:
                best_acc = valid_acc
                patience_counter = 0
                # 保存最佳模型
                os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                print(f"发现更好的模型，已保存到 '{config.BEST_MODEL_PATH}'")
                
                # 记录最佳模型到wandb
                wandb.run.summary["best_accuracy"] = best_acc
                wandb.run.summary["best_epoch"] = epoch
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"连续{patience}个epoch没有改善，提前停止训练")
                break
        else:
            print(f'Epoch [{epoch+1}/{epochs}], 所有批次损失均为NaN，跳过此轮')
    
    # 创建保存模型的目录（如果不存在）
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 保存最终模型
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"最终模型已保存为 '{config.MODEL_SAVE_PATH}'")
    
    # 记录最终结果到wandb
    wandb.run.summary["final_accuracy"] = valid_accs[-1]
    
    return train_losses, valid_accs


# 测试模型
def test_model(model, test_dataset):
    model.eval()
    test_loader = data.DataLoader(test_dataset, batch_size=1)
    
    correct = 0
    total = 0
    
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    # 记录预测结果，用于wandb可视化
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新混淆矩阵
            pred_label = predicted.item()
            true_label = labels.item()
            confusion_matrix[true_label][pred_label] += 1
            
            # 收集预测结果
            all_preds.append(pred_label)
            all_labels.append(true_label)
            all_probs.append(probs.cpu().numpy())
            
            print(f"实际表情: {EMOTIONS[labels.item()]}, 预测表情: {EMOTIONS[predicted.item()]}")
    
    accuracy = correct / total
    print(f'测试集准确率: {accuracy:.4f}')
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print("预测类别 →")
    print("实际类别 ↓")
    print("    ", end="")
    for i in range(NUM_CLASSES):
        print(f"{EMOTIONS[i]:^8}", end="")
    print()
    
    for i in range(NUM_CLASSES):
        print(f"{EMOTIONS[i]:4}", end="")
        for j in range(NUM_CLASSES):
            print(f"{confusion_matrix[i][j]:^8}", end="")
        print()
    
    # 计算每个类别的精确率和召回率
    print("\n每个类别的性能指标:")
    class_metrics = {}
    for i in range(NUM_CLASSES):
        precision = confusion_matrix[i][i] / confusion_matrix[:, i].sum() if confusion_matrix[:, i].sum() > 0 else 0
        recall = confusion_matrix[i][i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{EMOTIONS[i]}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")
        
        # 记录到字典
        class_metrics[EMOTIONS[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # 记录测试结果到wandb
    wandb.log({
        "test_accuracy": accuracy,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=[EMOTIONS[i] for i in range(NUM_CLASSES)]
        )
    })
    
    # 记录每个类别的指标
    for emotion, metrics in class_metrics.items():
        wandb.log({
            f"test_{emotion}_precision": metrics["precision"],
            f"test_{emotion}_recall": metrics["recall"],
            f"test_{emotion}_f1": metrics["f1"]
        })
    
    # 可选：记录一些预测样本到wandb
    sample_images = []
    sample_count = min(20, len(test_dataset))  # 最多记录20个样本
    indices = np.random.choice(len(test_dataset), sample_count, replace=False)
    
    for idx in indices:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            output = model(image)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
        
        # 转换图像以便可视化
        img_np = image.cpu().numpy()[0][0]  # 获取单通道图像
        img_np = (img_np * 0.5 + 0.5) * 255  # 反归一化
        
        # 记录到wandb
        caption = f"真实: {EMOTIONS[label]}, 预测: {EMOTIONS[pred]}"
        sample_images.append(wandb.Image(img_np, caption=caption))
    
    # 记录样本图像
    wandb.log({"sample_predictions": sample_images})

if __name__ == "__main__":
    # 加载数据
    train_loader, valid_loader = load_data()
    
    # 初始化wandb
    model_name = "FaceVGG"
    wandb_config = init_wandb(config, model_name)
    
    # 初始化模型
    model = create_vgg_model(config)
    print("模型结构:")
    model.summary()
    
    # 训练模型
    train_losses, valid_accs = train(model, train_loader, valid_loader.dataset, config.EPOCHS, config.LEARNING_RATE)
    
    # 加载最佳模型进行测试
    best_model = create_vgg_model(config)
    best_model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    best_model = best_model.to(config.DEVICE)
    
    # 测试模型
    print("\n在验证集上测试最佳模型:")
    test_model(best_model, valid_loader.dataset)
    
    # 完成实验，关闭wandb
    wandb.finish() 