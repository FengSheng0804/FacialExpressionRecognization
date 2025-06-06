import torch
from torch.utils import data
import numpy as np
import pandas as pd
import cv2
import random
import os
from torchvision import transforms
 
# 我们通过继承Dataset类来创建我们自己的数据加载类，命名为FaceDataset
class FaceDataset(data.Dataset):
    # 类初始化：
    # 1. 读取image-emotion对照表数据
    # 2. 使用pandas读取CSV文件
    # 3. 将数据存入numpy数组便于索引
    # 初始化
    def __init__(self, root, is_train=True):
        super(FaceDataset, self).__init__()
        self.root = root
        self.is_train = is_train
        
        # 使用os.path.join构建路径，避免不同操作系统路径分隔符问题
        csv_path = os.path.join(root, 'image_emotion.csv')
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}，请确保数据集已正确放置")
            
        df_path = pd.read_csv(csv_path, header=None, usecols=[0])
        df_label = pd.read_csv(csv_path, header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]
        
        # 定义基本转换
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])
        
        # 定义训练时的数据增强
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),  # 转换为PIL图像以便进行变换
            transforms.RandomRotation(40),  # 随机旋转
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomAffine(0, translate=(0.2, 0.2)),  # 随机平移
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
        ])

    # 图片读取与处理：
    # 1. 读取灰度图像
    # 2. 进行直方图均衡化增强
    # 3. 将48x48像素reshape为1x48x48适配卷积网络
    # 4. 转换为FloatTensor类型
    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        # 使用os.path.join构建图片路径
        img_path = os.path.join(self.root, self.path[item])
        
        # 检查图片文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片文件: {img_path}")
            
        face = cv2.imread(img_path)
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)
        
        # 根据是否为训练集决定是否应用数据增强
        if self.is_train:
            face_tensor = self.train_transform(face_hist)
        else:
            # 像素值标准化并转换为tensor
            face_tensor = self.basic_transform(face_hist)
            
        label = self.label[item]
        return face_tensor, label

    # 返回数据集样本总数
    def __len__(self):
        return self.path.shape[0]