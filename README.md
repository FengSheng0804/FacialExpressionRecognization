# 🎭 FacialExpressionRecognization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于深度学习的人脸表情识别项目，支持多种先进的神经网络架构和训练技术，可以准确识别人脸的7种基本表情。

## 🌟 项目特色

- **多种模型架构**：支持CNN、VGG、ResNet、DenseNet等多种经典和先进架构
- **注意力机制**：集成CBAM（Convolutional Block Attention Module）提升模型性能
- **知识蒸馏**：使用大模型指导小模型训练，实现模型压缩
- **自适应增长**：DenseNet支持自适应增长率，优化特征提取
- **实时检测**：支持摄像头实时表情识别
- **完整评估**：提供详细的模型评估和对比分析

## 📊 数据集

本项目使用FER2013数据集：
- **来源**：[Kaggle FER2013数据集](https://www.kaggle.com/datasets/ahmedmoorsy/facial-expression)
- **类别**：7种基本表情（愤怒、厌恶、恐惧、高兴、悲伤、惊讶、中性）
- **规格**：48×48像素灰度图像
- **规模**：约35,000张训练图像，约3,500张测试图像

## 🏗️ 项目架构

```
FacialExpressionRecognization/
├── 📁 dataset/                    # 数据集模块
│   ├── datasetPreprocess.py       # 数据预处理脚本
│   ├── FaceDataset.py            # PyTorch数据集类
│   ├── fer2013/                  # 原始FER2013数据
│   ├── train_set/                # 预处理后的训练集
│   └── verify_set/               # 预处理后的验证集
├── 📁 models/                     # 模型架构
│   ├── FaceCNN/                  # 基础CNN模型
│   ├── FaceVGG/                  # VGG系列模型
│   ├── ResNet/                   # ResNet系列模型
│   └── DenseNet/                 # DenseNet系列模型
│       ├── DenseNet.py           # DenseNet实现（含CBAM）
│       ├── knowledge_distillation.py  # 知识蒸馏
│       └── model_weight/         # 训练好的模型权重
├── 📁 results/                    # 训练结果和分析
├── 🔧 train_*.py                  # 各模型训练脚本
├── 🧪 test_*.py                   # 模型测试脚本
├── 📹 realtime_emotion_detection.py  # 实时表情检测
└── 📋 evaluate_densenet_models.py    # 模型评估脚本
```

## 🚀 快速开始

### 环境要求

```bash
Python 3.8+
PyTorch 1.9+
torchvision 0.10+
OpenCV 4.5+
scikit-learn 1.0+
matplotlib 3.3+
seaborn 0.11+
pandas 1.3+
numpy 1.21+
```

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install scikit-learn matplotlib seaborn pandas numpy
pip install wandb  # 可选，用于训练监控
```

### 数据准备

1. **下载数据集**
   ```bash
   # 从Kaggle下载FER2013数据集
   # 解压到 dataset/fer2013/ 目录
   ```

2. **数据预处理**
   ```bash
   python dataset/datasetPreprocess.py
   ```
   
   此脚本会：
   - 将原始数据分割为训练集、验证集和测试集
   - 将像素数据转换为图像文件
   - 生成对应的标签文件

## 🎯 模型训练

### 1. 基础模型训练

```bash
# CNN模型
python train_CNN.py

# VGG模型
python train_VGG.py

# ResNet模型
python train_ResNet.py

# DenseNet模型
python train_DenseNet.py
```

### 2. 高级特性训练

```bash
# 测试CBAM注意力机制
python test_cbam.py

# 测试自适应增长率
python test_adaptive_growth.py

# 知识蒸馏训练
python train_distillation.py
```

## 📈 模型评估

### 评估DenseNet模型

```bash
python evaluate_densenet_models.py
```

该脚本会：
- 加载所有DenseNet变体模型
- 计算详细的性能指标（准确率、精确率、召回率、F1分数、ROC-AUC）
- 生成混淆矩阵和ROC曲线
- 输出模型对比报告

### 性能对比

| 模型                      | 参数量 | 准确率 | 推理时间 | 特色功能   |
| ------------------------- | ------ | ------ | -------- | ---------- |
| FaceCNN                   | ~1M    | ~65%   | 快       | 基础架构   |
| FaceVGG                   | ~15M   | ~70%   | 中等     | 经典架构   |
| ResNet                    | ~11M   | ~72%   | 中等     | 残差连接   |
| DenseNet121               | ~7M    | ~75%   | 中等     | 密集连接   |
| DenseNet121+CBAM          | ~8M    | ~77%   | 中等     | 注意力机制 |
| DenseNet121+CBAM+Adaptive | ~8M    | ~78%   | 中等     | 自适应增长 |
| DenseNet121+Distilled     | ~8M    | ~79%   | 中等     | 知识蒸馏   |

## 🎬 实时检测

```bash
python realtime_emotion_detection.py
```

功能特性：
- 实时摄像头输入
- 人脸检测和对齐
- 表情实时分类
- 置信度显示
- 支持多人脸检测

## 🔬 核心技术

### 1. CBAM注意力机制
- **通道注意力**：学习"关注什么"特征
- **空间注意力**：学习"关注哪里"区域
- **性能提升**：相比基础模型提升2-3%准确率

### 2. 知识蒸馏
- **教师模型**：DenseNet201（更大更强）
- **学生模型**：DenseNet121（轻量高效）
- **蒸馏损失**：结合硬标签和软标签
- **压缩效果**：模型大小减少50%，性能仅下降1%

### 3. 自适应增长率
- **传统DenseNet**：固定增长率32
- **自适应方案**：[16, 24, 32, 40]
- **优势**：更好的特征层次表示

## 📊 训练监控

项目支持Wandb进行训练监控：

```python
# 在训练脚本中启用
import wandb
wandb.init(project="facial-expression-recognition")
```

监控内容：
- 训练/验证损失和准确率
- 学习率变化
- 模型参数分布
- 梯度范数
- 训练时间统计

## 🛠️ 自定义配置

### 模型配置

每个模型都有对应的配置文件：
- `models/FaceCNN/FaceCNNConfig.py`
- `models/FaceVGG/FaceVGGConfig.py`
- `models/ResNet/ResNetConfig.py`
- `models/DenseNet/DenseNetConfig.py`

### 训练参数调整

```python
# 示例：DenseNet配置
class DenseNetConfig:
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    USE_CBAM = True
    USE_ADAPTIVE_GROWTH = True
    ADAPTIVE_GROWTH_LIST = [16, 24, 32, 40]
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   BATCH_SIZE = 32  # 改为16或8
   ```

2. **数据集路径错误**
   ```bash
   # 确保数据集结构正确
   dataset/
   ├── fer2013/
   ├── train_set/
   └── verify_set/
   ```

3. **模型加载失败**
   ```bash
   # 检查模型文件是否存在
   ls models/DenseNet/model_weight/
   ```

## 📝 更新日志

### v2.0.0 (2025-01-01)
- ✅ 添加知识蒸馏支持
- ✅ 集成CBAM注意力机制
- ✅ 实现自适应增长率
- ✅ 完善模型评估系统

### v1.0.0 (2024-12-01)
- ✅ 基础CNN、VGG、ResNet、DenseNet实现
- ✅ 数据预处理管道
- ✅ 实时检测功能

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- FER2013数据集提供者
- PyTorch团队
- 开源社区的各位贡献者

## 📞 联系方式

- **作者**：FengSheng0804
- **邮箱**：[15903641403@163.com]
- **项目链接**：[https://github.com/FengSheng0804/FacialExpressionRecognization](https://github.com/FengSheng0804/FacialExpressionRecognization)

---

如果这个项目对您有帮助，请给它一个⭐！