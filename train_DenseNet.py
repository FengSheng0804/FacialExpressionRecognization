import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import sys
import argparse
from contextlib import nullcontext

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

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def get_gpu_config():
    """获取GPU配置信息"""
    if not torch.cuda.is_available():
        return {
            'use_multi_gpu': False,
            'gpu_count': 0,
            'device': torch.device('cpu'),
            'use_distributed': False
        }
    
    gpu_count = torch.cuda.device_count()
    use_multi_gpu = gpu_count > 1
    
    print(f"检测到 {gpu_count} 个GPU:")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    return {
        'use_multi_gpu': use_multi_gpu,
        'gpu_count': gpu_count,
        'device': torch.device('cuda:0'),
        'use_distributed': False  # 可以设置为True启用分布式训练
    }

def init_wandb(config, gpu_config, rank=0):
    """初始化wandb"""
    # 只在主进程中初始化wandb
    if rank != 0:
        return False
        
    try:
        wandb.init(
            project="facial-expression-recognition",
            name=f"densenet_{config.DENSENET_TYPE}_{'cbam_' if config.USE_CBAM else 'no_cbam_'}{config.OPTIMIZER}_lr{config.LEARNING_RATE}_{'multi_gpu' if gpu_config['use_multi_gpu'] else 'single_gpu'}",
            config={
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "effective_batch_size": config.BATCH_SIZE * gpu_config['gpu_count'] if gpu_config['use_multi_gpu'] else config.BATCH_SIZE,
                "optimizer": config.OPTIMIZER,
                "model_architecture": "DenseNet",
                "dataset": "FER2013",
                "densenet_type": config.DENSENET_TYPE,
                "growth_rate": config.GROWTH_RATE,
                "reduction": config.REDUCTION,
                "use_cbam": config.USE_CBAM,
                "cbam_ratio": config.CBAM_RATIO if config.USE_CBAM else None,
                "cbam_kernel_size": config.CBAM_KERNEL_SIZE if config.USE_CBAM else None,
                "device": str(config.DEVICE),
                "gpu_count": gpu_config['gpu_count'],
                "use_multi_gpu": gpu_config['use_multi_gpu'],
                "use_distributed": gpu_config['use_distributed']
            },
            settings=wandb.Settings(init_timeout=180)  # 增加超时时间到180秒
        )
        return True
    except Exception as e:
        print(f"警告: wandb初始化失败: {str(e)}")
        print("将继续训练，但不会记录到wandb")
        return False

def load_data(config, gpu_config, rank=0, world_size=1):
    """加载数据"""
    if rank == 0:
        print("正在加载数据...")
    
    # 创建数据集
    train_dataset = FaceDataset(config.TRAIN_DATA_PATH, is_train=True)
    val_dataset = FaceDataset(config.VALID_DATA_PATH, is_train=False)
    
    # 为分布式训练创建采样器
    if gpu_config['use_distributed']:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # 调整num_workers以优化多GPU性能
    num_workers = min(8, os.cpu_count()) if gpu_config['use_multi_gpu'] else 4
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,  # 启用内存固定以加速GPU传输
        drop_last=True   # 确保每个GPU获得相同大小的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        if gpu_config['use_multi_gpu']:
            effective_batch_size = config.BATCH_SIZE * gpu_config['gpu_count']
            print(f"单GPU批次大小: {config.BATCH_SIZE}")
            print(f"有效批次大小: {effective_batch_size}")
    
    return train_loader, val_loader, train_sampler

def validate(model, val_loader, criterion, device, gpu_config, rank=0):
    """验证模型"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # 对于分布式训练，需要同步所有进程的结果
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
      # 如果使用分布式训练，需要汇总所有GPU的结果
    if gpu_config['use_distributed']:
        # 将指标转换为tensor并同步
        metrics = torch.tensor([val_loss * len(val_loader), correct, total], dtype=torch.float, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_loss_sum, correct, total = metrics.cpu().numpy()
        val_loss = val_loss_sum / (len(val_loader) * dist.get_world_size())
    else:
        val_loss = val_loss / len(val_loader)
    
    accuracy = correct / total if total > 0 else 0
    return val_loss, accuracy

def test_model(model, val_loader, criterion, device, gpu_config, rank=0):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    
    confusion_matrix = np.zeros((7, 7), dtype=int)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新混淆矩阵
            for i in range(len(labels)):
                pred_label = predicted[i].item()
                true_label = labels[i].item()
                confusion_matrix[true_label][pred_label] += 1
    
    # 如果使用分布式训练，汇总所有GPU的结果
    if gpu_config['use_distributed']:
        # 汇总准确率
        metrics = torch.tensor([correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        correct, total = metrics.cpu().numpy()
        
        # 汇总混淆矩阵
        confusion_tensor = torch.from_numpy(confusion_matrix).to(device)
        dist.all_reduce(confusion_tensor, op=dist.ReduceOp.SUM)
        confusion_matrix = confusion_tensor.cpu().numpy()
    
    # 只在主进程中打印结果
    if rank == 0:
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

def print_model_info(model, config, gpu_config, rank=0):
    """打印模型信息"""
    if rank != 0:
        return
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"  架构: {config.DENSENET_TYPE}")
    print(f"  CBAM注意力机制: {'启用' if config.USE_CBAM else '禁用'}")
    if config.USE_CBAM:
        print(f"  CBAM通道缩放比例: {config.CBAM_RATIO}")
        print(f"  CBAM空间卷积核大小: {config.CBAM_KERNEL_SIZE}")
    print(f"  自适应增长率: {'启用' if config.USE_ADAPTIVE_GROWTH else '禁用'}")
    if config.USE_ADAPTIVE_GROWTH and config.ADAPTIVE_GROWTH_LIST:
        print(f"  自适应增长率: {config.ADAPTIVE_GROWTH_LIST}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  设备: {config.DEVICE}")
    print(f"  GPU配置:")
    print(f"    GPU数量: {gpu_config['gpu_count']}")
    print(f"    多GPU训练: {'启用' if gpu_config['use_multi_gpu'] else '禁用'}")
    print(f"    分布式训练: {'启用' if gpu_config['use_distributed'] else '禁用'}")

def setup_model_for_multi_gpu(model, gpu_config, rank=0):
    """为多GPU训练设置模型"""
    if gpu_config['use_distributed']:
        # 分布式训练
        model = DDP(model, device_ids=[rank], output_device=rank)
    elif gpu_config['use_multi_gpu']:
        # 数据并行训练
        model = nn.DataParallel(model)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, filepath, gpu_config):
    """保存模型检查点"""
    # 获取原始模型（去除DataParallel或DDP包装）
    if gpu_config['use_multi_gpu'] or gpu_config['use_distributed']:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_model_checkpoint(model, optimizer, filepath, device):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def train(config, gpu_config=None, rank=0, world_size=1):
    """训练模型"""
    # 设置分布式训练环境
    if gpu_config and gpu_config['use_distributed']:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = config.DEVICE
        
    # 初始化wandb（仅在主进程）
    use_wandb = init_wandb(config, gpu_config, rank)
    
    # 加载数据
    train_loader, val_loader, train_sampler = load_data(config, gpu_config, rank, world_size)
    
    # 创建模型
    if rank == 0:
        print(f"创建 {config.DENSENET_TYPE} 模型，CBAM: {'启用' if config.USE_CBAM else '禁用'}")
    
    if config.DENSENET_TYPE == 'densenet121':
        model = densenet121(num_class=config.OUTPUT_SIZE, use_cbam=config.USE_CBAM, use_adaptive_growth=config.USE_ADAPTIVE_GROWTH, adaptive_growth_list=config.ADAPTIVE_GROWTH_LIST)
    elif config.DENSENET_TYPE == 'densenet169':
        model = densenet169(num_class=config.OUTPUT_SIZE, use_cbam=config.USE_CBAM, use_adaptive_growth=config.USE_ADAPTIVE_GROWTH, adaptive_growth_list=config.ADAPTIVE_GROWTH_LIST)
    elif config.DENSENET_TYPE == 'densenet201':
        model = densenet201(num_class=config.OUTPUT_SIZE, use_cbam=config.USE_CBAM, use_adaptive_growth=config.USE_ADAPTIVE_GROWTH, adaptive_growth_list=config.ADAPTIVE_GROWTH_LIST)
    elif config.DENSENET_TYPE == 'densenet161':
        model = densenet161(num_class=config.OUTPUT_SIZE, use_cbam=config.USE_CBAM, use_adaptive_growth=config.USE_ADAPTIVE_GROWTH, adaptive_growth_list=config.ADAPTIVE_GROWTH_LIST)
    else:
        raise ValueError(f"不支持的DenseNet类型: {config.DENSENET_TYPE}")
    
    # 修改第一层卷积以适应灰度图像
    if config.USE_ADAPTIVE_GROWTH and config.ADAPTIVE_GROWTH_LIST:
        first_growth = config.ADAPTIVE_GROWTH_LIST[0]
    else:
        first_growth = 48 if config.DENSENET_TYPE == 'densenet161' else config.GROWTH_RATE
    model.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 2 * first_growth, kernel_size=3, padding=1, bias=False)
    
    model = model.to(device)
    
    # 设置多GPU训练
    if gpu_config:
        model = setup_model_for_multi_gpu(model, gpu_config, rank)
    
    # 打印模型信息
    print_model_info(model, config, gpu_config, rank)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 调整学习率以适应多GPU训练
    lr = config.LEARNING_RATE
    if gpu_config and gpu_config['use_multi_gpu']:
        # 根据GPU数量调整学习率（线性缩放规则）
        lr = config.LEARNING_RATE * gpu_config['gpu_count']
        if rank == 0:
            print(f"调整学习率以适应多GPU训练: {config.LEARNING_RATE} -> {lr}")
    
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY
        )
    else:  # SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
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
    if rank == 0:
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 训练循环
    best_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 早停机制参数
    patience = 20  # 容忍多少个epoch没有改善
    patience_counter = 0
    
    # 记录模型架构到wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=10)
    
    if rank == 0:
        print("开始训练...")
    
    for epoch in range(config.EPOCHS):
        # 设置分布式采样器的epoch
        if gpu_config and gpu_config['use_distributed'] and train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印训练进度（仅主进程）
            if rank == 0 and (batch_idx + 1) % 10 == 0:
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
          # 同步分布式训练的指标
        if gpu_config and gpu_config['use_distributed']:
            # 汇总训练指标
            metrics = torch.tensor([train_loss * len(train_loader), correct, total], dtype=torch.float, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            train_loss_sum, correct, total = metrics.cpu().numpy()
            train_loss = train_loss_sum / (len(train_loader) * world_size)
        else:
            train_loss = train_loss / len(train_loader)
        
        train_acc = correct / total if total > 0 else 0
        
        # 验证
        val_loss, valid_acc = validate(model, val_loader, criterion, device, gpu_config, rank)
        
        # 更新学习率
        if config.USE_LR_SCHEDULER:
            scheduler.step()
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'当前学习率: {current_lr:.6f}')
        
        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(valid_acc)
          # 记录到wandb（如果可用且为主进程）
        if use_wandb and rank == 0:
            try:
                wandb.log({
                    "epoch": epoch,
                    "batch_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": valid_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            except Exception as e:
                print(f"警告: wandb记录失败: {str(e)}")
        
        # 打印epoch结果（仅主进程）
        if rank == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}')
        
        # 保存最佳模型（仅主进程）
        if rank == 0:
            if valid_acc > best_acc:
                print(f"发现更好的模型，已保存到 '{config.BEST_MODEL_PATH}'")
                best_acc = valid_acc
                patience_counter = 0  # 重置早停计数器
                
                # 保存模型状态
                if gpu_config and (gpu_config['use_multi_gpu'] or gpu_config['use_distributed']):
                    torch.save(model.module.state_dict(), config.BEST_MODEL_PATH)
                else:
                    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
                
                # 记录最佳模型到wandb
                if use_wandb:
                    try:
                        wandb.run.summary["best_accuracy"] = best_acc
                        wandb.run.summary["best_epoch"] = epoch
                    except Exception as e:
                        print(f"警告: wandb记录最佳模型失败: {str(e)}")
            else:
                patience_counter += 1
            
            # 早停机制检查
            if patience_counter >= patience:
                print(f"连续{patience}个epoch没有改善，提前停止训练")
                break
            
            # 保存最新模型
            if gpu_config and (gpu_config['use_multi_gpu'] or gpu_config['use_distributed']):
                torch.save(model.module.state_dict(), config.MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        
        # 同步所有进程
        if gpu_config and gpu_config['use_distributed']:
            dist.barrier()
    
    if rank == 0:
        print(f"最终模型已保存为 '{config.MODEL_SAVE_PATH}'")
        
        # 记录最终结果到wandb
        if use_wandb:
            try:
                wandb.run.summary["final_accuracy"] = val_accs[-1] if val_accs else 0
            except Exception as e:
                print(f"警告: wandb记录最终结果失败: {str(e)}")
        
        # 测试最佳模型
        print("\n在验证集上测试最佳模型:")
        if gpu_config and (gpu_config['use_multi_gpu'] or gpu_config['use_distributed']):
            model.module.load_state_dict(torch.load(config.BEST_MODEL_PATH))
        else:
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
        test_model(model, val_loader, criterion, device, gpu_config, rank)
        
        # 完成实验，关闭wandb
        if use_wandb:
            wandb.finish()
    
    # 清理分布式训练环境
    if gpu_config and gpu_config['use_distributed']:
        cleanup_distributed()

def train_distributed(rank, world_size, config, gpu_config):
    """分布式训练的包装函数"""
    try:
        train(config, gpu_config, rank, world_size)
    except Exception as e:
        print(f"分布式训练进程 {rank} 出错: {str(e)}")
        cleanup_distributed()
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DenseNet 表情识别训练')
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='指定使用的GPU ID，用逗号分隔')
    args = parser.parse_args()
    
    # 创建配置
    config = DenseNetConfig()
    
    # 获取GPU配置
    gpu_config = get_gpu_config()
    
    # 根据命令行参数调整GPU配置
    if args.distributed and gpu_config['gpu_count'] > 1:
        gpu_config['use_distributed'] = True
        gpu_config['use_multi_gpu'] = False  # 分布式训练时不使用DataParallel
    elif gpu_config['gpu_count'] > 1:
        gpu_config['use_multi_gpu'] = True
        gpu_config['use_distributed'] = False
    
    # 解析指定的GPU ID
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        available_gpus = min(len(gpu_ids), gpu_config['gpu_count'])
        gpu_config['gpu_count'] = available_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids[:available_gpus]))
    
    print(f"开始训练配置:")
    print(f"  模型: {config.DENSENET_TYPE}")
    print(f"  CBAM: {'启用' if config.USE_CBAM else '禁用'}")
    print(f"  自适应增长率: {'启用' if config.USE_ADAPTIVE_GROWTH else '禁用'}")
    print(f"  增长率: {config.GROWTH_RATE}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  训练轮数: {config.EPOCHS}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  优化器: {config.OPTIMIZER}")
    print(f"  设备: {config.DEVICE}")
    print(f"  GPU配置: {gpu_config}")
    if gpu_config['use_multi_gpu']:
        print(f"  有效批次大小: {config.BATCH_SIZE * gpu_config['gpu_count']}")
    print("-" * 50)
    
    try:
        if gpu_config['use_distributed']:
            # 启动分布式训练
            print(f"启动分布式训练，使用 {gpu_config['gpu_count']} 个GPU")
            mp.spawn(
                train_distributed,
                args=(gpu_config['gpu_count'], config, gpu_config),
                nprocs=gpu_config['gpu_count'],
                join=True
            )
        else:
            # 单GPU或多GPU DataParallel训练
            train(config, gpu_config)
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()