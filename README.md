# FacialExpressionRecognization

这是一个用于实现人类表情识别的项目，基于深度学习技术，可以识别和分类人脸表情。

## 项目简介

本项目使用深度学习技术实现人脸表情识别，可以识别多种基本表情，如开心、悲伤、愤怒等。项目使用FER2013数据集进行训练和测试。

## 数据集

本项目使用FER2013数据集，该数据集包含以下特点：
- 来源：[Kaggle FER2013数据集](https://www.kaggle.com/datasets/ahmedmoorsy/facial-expression)
- 包含7种基本表情类别
- 图像尺寸为48x48像素的灰度图像
- 训练集和测试集已预先划分

## 项目结构

```
FacialExpressionRecognization/
├── dataset/                 # 数据集目录
│   └── datasetPreprocess.py # 数据集预处理脚本
├── README.md               # 项目说明文档
```

## 环境要求

- Python 3.x
- 相关依赖包（待补充）

## 使用说明

1. 数据集准备
   - 从Kaggle下载FER2013数据集
   - 将数据集放置在dataset目录下的fer2013中

2. 数据预处理
   - 运行datasetPreprocess.py进行数据预处理
     - 实现了数据集分割功能`(split_dataset)`：将原始FER2013数据集分割为训练集、验证集和测试集
     - 实现了表情标签和像素数据分离功能`(split_emotion_pixels)`：将数据集中的表情标签和像素数据分别保存
     - 实现了像素转图像功能`(pix2image)`：将像素数据转换为可查看的jpg格式图像

3. 模型训练（待实现）
   - 待补充

4. 表情识别（待实现）
   - 待补充

## 待完成功能

- [ ] 模型训练脚本
- [ ] 表情识别实现
- [ ] 实时摄像头识别
- [ ] 模型评估和优化

## 贡献

欢迎提交Issue和Pull Request来帮助改进这个项目。

## 许可证

待补充