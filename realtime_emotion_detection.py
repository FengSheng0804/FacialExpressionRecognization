import cv2
import torch
import numpy as np
import os
import time
from datetime import datetime
from models.FaceCNN.FaceCNN import FaceCNN
from models.FaceCNN.FaceCNNConfig import FaceCNNConfig
from models.FaceVGG.FaceVGG import FaceVGG, create_vgg_model
from models.FaceVGG.FaceVGGConfig import FaceVGGConfig
from models.ResNet.ResNet import ResNet50, ResNet101, ResNet152
from models.ResNet.ResNetConfig import ResNetConfig
from models.DenseNet.DenseNet import densenet121, densenet169, densenet201, densenet161
from models.DenseNet.DenseNetConfig import DenseNetConfig
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import torch.nn as nn

# 表情标签映射
EMOTIONS = {
    0: '生气', 
    1: '厌恶', 
    2: '恐惧', 
    3: '开心', 
    4: '伤心', 
    5: '惊讶', 
    6: '中性'
}

# 表情对应的颜色 (BGR格式)
EMOTION_COLORS = {
    0: (0, 0, 255),    # 生气 - 红色
    1: (0, 140, 255),  # 厌恶 - 橙色
    2: (0, 255, 255),  # 恐惧 - 黄色
    3: (0, 255, 0),    # 开心 - 绿色
    4: (255, 0, 0),    # 伤心 - 蓝色
    5: (255, 0, 255),  # 惊讶 - 紫色
    6: (255, 255, 255) # 中性 - 白色
}

class RealtimeEmotionDetector:
    def __init__(self, model_type='cnn', model_path=None, simple_mode=False):
        # 初始化配置和模型类型
        self.model_type = model_type.lower()
        self.simple_mode = simple_mode
        
        # 标记模型是否可用
        self.model_available = True
        
        # 根据模型类型选择配置
        if self.model_type == 'vgg':
            self.config = FaceVGGConfig()
            print("使用VGG模型进行表情识别")
            # 设置VGG模型的默认参数
            self.vgg_params = {
                'input_size': self.config.IMAGE_SIZE,
                'input_channels': self.config.INPUT_CHANNELS,
                'num_classes': self.config.OUTPUT_SIZE,
                'vgg_type': self.config.VGG_TYPE,
                'use_batchnorm': self.config.USE_BATCHNORM,
                'pretrained': self.config.PRETRAINED,
                'feature_extract': self.config.FEATURE_EXTRACT,
                'activation': self.config.ACTIVATION,
                'keep_prob': self.config.KEEP_PROB
            }
        elif self.model_type == 'resnet':
            self.config = ResNetConfig()
            print("使用ResNet模型进行表情识别")
        elif self.model_type == 'densenet':
            self.config = DenseNetConfig()
            print("使用DenseNet模型进行表情识别")
        else:
            self.config = FaceCNNConfig()
            print("使用CNN模型进行表情识别")
        
        # 如果未指定模型路径，使用配置中的最佳模型路径
        if model_path is None:
            model_path = self.config.BEST_MODEL_PATH
            
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            print("将只进行人脸检测，不进行表情识别")
            self.model_available = False
        
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型（如果可用）
        if self.model_available:
            if self.model_type == 'vgg':
                try:
                    # 使用create_vgg_model创建模型
                    self.model = create_vgg_model(self.config)
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("VGG模型加载成功（使用非严格模式strict=False）")
                    self.model.to(self.device)
                    self.model.eval()  # 设置为评估模式
                except Exception as e:
                    print(f"警告: VGG模型加载出现问题: {str(e)}")
                    print("将只进行人脸检测，不进行表情识别")
                    self.model_available = False
            elif self.model_type == 'resnet':
                try:
                    # 根据配置选择ResNet类型
                    if self.config.RESNET_TYPE == 'resnet50':
                        self.model = ResNet50(num_classes=self.config.OUTPUT_SIZE)
                    elif self.config.RESNET_TYPE == 'resnet101':
                        self.model = ResNet101(num_classes=self.config.OUTPUT_SIZE)
                    elif self.config.RESNET_TYPE == 'resnet152':
                        self.model = ResNet152(num_classes=self.config.OUTPUT_SIZE)
                    else:
                        raise ValueError(f"不支持的ResNet类型: {self.config.RESNET_TYPE}")
                    
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"{self.config.RESNET_TYPE.upper()}模型加载成功（使用非严格模式strict=False）")
                    self.model.to(self.device)
                    self.model.eval()
                except Exception as e:
                    print(f"警告: ResNet模型加载出现问题: {str(e)}")
                    print("将只进行人脸检测，不进行表情识别")
                    self.model_available = False
            elif self.model_type == 'densenet':
                try:
                    # 根据配置选择DenseNet类型
                    if self.config.DENSENET_TYPE == 'densenet121':
                        self.model = densenet121()
                    elif self.config.DENSENET_TYPE == 'densenet169':
                        self.model = densenet169()
                    elif self.config.DENSENET_TYPE == 'densenet201':
                        self.model = densenet201()
                    elif self.config.DENSENET_TYPE == 'densenet161':
                        self.model = densenet161()
                    else:
                        raise ValueError(f"不支持的DenseNet类型: {self.config.DENSENET_TYPE}")
                    
                    # 修改第一层卷积以适应灰度图像
                    self.model.conv1 = nn.Conv2d(self.config.INPUT_CHANNELS, 2 * self.config.GROWTH_RATE, 
                                               kernel_size=3, padding=1, bias=False)
                    
                    # 修改最后的全连接层以适应表情分类
                    self.model.linear = nn.Linear(self.model.linear.in_features, self.config.OUTPUT_SIZE)
                    
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                except Exception as e:
                    print(f"警告: DenseNet模型加载出现问题: {str(e)}")
                    print("将只进行人脸检测，不进行表情识别")
                    self.model_available = False
            else:
                self.model = FaceCNN(
                    input_size=self.config.IMAGE_SIZE,
                    use_batchnorm=self.config.USE_BATCHNORM,
                    activation=self.config.ACTIVATION,
                    keep_prob=1.0  # 推理时不使用dropout
                )
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("CNN模型加载成功（使用非严格模式strict=False）")
                    self.model.to(self.device)
                    self.model.eval()
                except Exception as e:
                    print(f"警告: CNN模型加载出现问题: {str(e)}")
                    print("将只进行人脸检测，不进行表情识别")
                    self.model_available = False
        
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise ValueError("无法加载人脸检测器，请确保OpenCV安装正确")
            
        # 图像预处理参数
        self.target_size = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        
        # FPS计算
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        
        # 视频录制参数
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "output"
        
        # 状态消息
        self.status_message = ""
        self.status_time = 0
        
        # 加载字体文件
        try:
            # 尝试加载系统中的中文字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # Windows 默认黑体
                "C:/Windows/Fonts/simsun.ttc",  # Windows 默认宋体
                "C:/Windows/Fonts/msyh.ttc",    # Windows 默认雅黑
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿字体
                "/System/Library/Fonts/PingFang.ttc"  # macOS 字体
            ]
            
            self.font_path = None
            for path in font_paths:
                if os.path.exists(path):
                    self.font_path = path
                    print(f"找到中文字体: {path}")
                    break
                    
            if self.font_path is None:
                print("警告: 未找到中文字体文件，将使用默认字体")
        except Exception as e:
            print(f"加载字体时出错: {str(e)}")
            self.font_path = None
        
    def preprocess_face(self, face_img):
        """预处理人脸图像，与训练时保持一致"""
        try:
            # 调整大小
            face_resized = cv2.resize(face_img, self.target_size)
            # 转换为灰度图
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            # 直方图均衡化
            face_hist = cv2.equalizeHist(face_gray)
            # 归一化并转换为tensor
            face_tensor = torch.from_numpy(face_hist).float()
            # 添加批次和通道维度 [H,W] -> [1,1,H,W]
            face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
            # 标准化，与训练时一致
            face_tensor = face_tensor / 255.0 * 2 - 1
            return face_tensor
        except Exception as e:
            print(f"预处理人脸图像时出错: {str(e)}")
            return None
            
    def cv2_img_add_text(self, img, text, left, top, text_color=(0, 255, 0), text_size=20):
        """使用PIL绘制中文"""
        if isinstance(img, np.ndarray):
            # 将OpenCV格式图片转换为PIL格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            
            # 加载字体
            if self.font_path:
                font = ImageFont.truetype(self.font_path, text_size)
            else:
                font = ImageFont.load_default()
                
            # 绘制文本
            draw.text((left, top), text, text_color, font=font)
            
            # 转换回OpenCV格式
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
        
    def detect_emotion(self, frame):
        """检测帧中的人脸并识别表情"""
        # 计算FPS
        self.new_frame_time = time.time()
        self.fps = 1 / (self.new_frame_time - self.prev_frame_time) if (self.new_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = self.new_frame_time
        
        # 创建一个副本用于显示
        display_frame = frame.copy()
        
        # 显示FPS
        display_frame = self.cv2_img_add_text(display_frame, f"FPS: {int(self.fps)}", 10, 30, (0, 255, 0), 20)
        
        # 显示模型类型和状态
        model_text = f"模型: {self.model_type.upper()}"
        if not self.model_available:
            model_text += " (仅检测人脸)"
        display_frame = self.cv2_img_add_text(display_frame, model_text, 10, display_frame.shape[0] - 60, (255, 255, 255), 16)
        
        # 显示录制状态
        if self.is_recording:
            display_frame = self.cv2_img_add_text(display_frame, "录制中...", 10, 60, (0, 0, 255), 20)
            
        # 显示状态消息
        if time.time() - self.status_time < 3.0:  # 显示3秒
            display_frame = self.cv2_img_add_text(display_frame, self.status_message, 10, 90, (255, 255, 0), 20)
        
        # 显示帮助信息
        help_text = "按 'q' 退出 | 's' 切换模式 | 'r' 录制 | 'c' 截图 | 'm' 切换模型"
        display_frame = self.cv2_img_add_text(
            display_frame, 
            help_text, 
            10, 
            display_frame.shape[0] - 30, 
            (255, 255, 255), 
            16
        )
        
        # 转换为灰度图进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 处理每个检测到的人脸
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]
            
            # 如果模型不可用，只绘制人脸框
            if not self.model_available:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                continue
            
            try:
                # 预处理人脸
                face_tensor = self.preprocess_face(face_roi)
                if face_tensor is None:
                    continue
                    
                face_tensor = face_tensor.to(self.device)
                
                # 进行预测
                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(outputs, dim=1).item()
                    confidence = probs[0][pred_idx].item()
                
                # 获取预测的表情和置信度
                emotion = EMOTIONS[pred_idx]
                color = EMOTION_COLORS[pred_idx]
                
                # 如果是简单模式，只显示人脸框和预测结果
                if self.simple_mode:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # 显示表情和置信度
                    text = f"{emotion} ({confidence:.2f})"
                    display_frame = self.cv2_img_add_text(
                        display_frame, 
                        text, 
                        x, 
                        y-30, 
                        (color[2], color[1], color[0]), 
                        20
                    )
                    continue
                    
                # 详细模式下的显示
                # 在图像上绘制结果
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # 显示表情和置信度
                text = f"{emotion} ({confidence:.2f})"
                display_frame = self.cv2_img_add_text(
                    display_frame, 
                    text, 
                    x, 
                    y-30, 
                    (color[2], color[1], color[0]), 
                    20
                )
                
                # 绘制表情条形图
                bar_width = 20
                max_bar_height = 100
                start_x = x + w + 10
                
                for i, prob in enumerate(probs[0].cpu().numpy()):
                    bar_height = int(prob * max_bar_height)
                    bar_color = EMOTION_COLORS[i]
                    cv2.rectangle(display_frame, 
                                 (start_x, y + i * bar_width), 
                                 (start_x + bar_height, y + (i+1) * bar_width - 2), 
                                 bar_color, -1)
                    
                    # 使用PIL绘制中文标签
                    label_text = f"{EMOTIONS[i]}: {prob:.2f}"
                    display_frame = self.cv2_img_add_text(
                        display_frame, 
                        label_text, 
                        start_x + bar_height + 5, 
                        y + i * bar_width, 
                        (bar_color[2], bar_color[1], bar_color[0]), 
                        16
                    )
                
            except Exception as e:
                print(f"处理人脸时出错: {str(e)}")
                # 仍然绘制人脸框，但不显示表情
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        return display_frame
    
    def take_screenshot(self, frame):
        """保存当前帧为图片"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        self.status_message = f"截图已保存: {filename}"
        self.status_time = time.time()
        print(self.status_message)
        
    def toggle_recording(self, frame):
        """切换视频录制状态"""
        if self.is_recording:
            # 停止录制
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            self.status_message = "录制已停止"
            print(self.status_message)
        else:
            # 开始录制
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            
            # 获取视频参数
            height, width = frame.shape[:2]
            fps = 20.0  # 固定帧率
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者尝试 'XVID'
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            self.is_recording = True
            self.status_message = f"开始录制: {filename}"
            print(self.status_message)
        
        self.status_time = time.time()
        
    def switch_model(self):
        """切换使用的模型类型"""
        if self.model_type == 'cnn':
            # 切换到VGG模型
            self.model_type = 'vgg'
            self.config = FaceVGGConfig()
            model_path = self.config.BEST_MODEL_PATH
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                self.model_available = False
                self.status_message = "已切换到VGG模型（仅检测人脸）"
                self.status_time = time.time()
                print(f"警告: VGG模型文件不存在: {model_path}")
                print("将只进行人脸检测，不进行表情识别")
                return
                
            try:
                # 使用create_vgg_model创建模型
                self.model = create_vgg_model(self.config)
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model_available = True
                self.status_message = "已切换到VGG模型"
            except Exception as e:
                self.model_available = False
                self.status_message = f"VGG模型加载失败（仅检测人脸）: {str(e)}"
        elif self.model_type == 'vgg':
            # 切换到ResNet模型
            self.model_type = 'resnet'
            self.config = ResNetConfig()
            model_path = self.config.BEST_MODEL_PATH
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                self.model_available = False
                self.status_message = "已切换到ResNet模型（仅检测人脸）"
                self.status_time = time.time()
                print(f"警告: ResNet模型文件不存在: {model_path}")
                print("将只进行人脸检测，不进行表情识别")
                return
                
            try:
                # 根据配置选择ResNet类型
                if self.config.RESNET_TYPE == 'resnet50':
                    self.model = ResNet50(num_classes=self.config.OUTPUT_SIZE)
                elif self.config.RESNET_TYPE == 'resnet101':
                    self.model = ResNet101(num_classes=self.config.OUTPUT_SIZE)
                elif self.config.RESNET_TYPE == 'resnet152':
                    self.model = ResNet152(num_classes=self.config.OUTPUT_SIZE)
                else:
                    raise ValueError(f"不支持的ResNet类型: {self.config.RESNET_TYPE}")
                
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model_available = True
                self.status_message = f"已切换到{self.config.RESNET_TYPE.upper()}模型"
            except Exception as e:
                self.model_available = False
                self.status_message = f"ResNet模型加载失败（仅检测人脸）: {str(e)}"
        elif self.model_type == 'resnet':
            # 切换到DenseNet模型
            self.model_type = 'densenet'
            self.config = DenseNetConfig()
            model_path = self.config.BEST_MODEL_PATH
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                self.model_available = False
                self.status_message = "已切换到DenseNet模型（仅检测人脸）"
                self.status_time = time.time()
                print(f"警告: DenseNet模型文件不存在: {model_path}")
                print("将只进行人脸检测，不进行表情识别")
                return
                
            try:
                # 根据配置选择DenseNet类型
                if self.config.DENSENET_TYPE == 'densenet121':
                    self.model = densenet121()
                elif self.config.DENSENET_TYPE == 'densenet169':
                    self.model = densenet169()
                elif self.config.DENSENET_TYPE == 'densenet201':
                    self.model = densenet201()
                elif self.config.DENSENET_TYPE == 'densenet161':
                    self.model = densenet161()
                else:
                    raise ValueError(f"不支持的DenseNet类型: {self.config.DENSENET_TYPE}")
                
                # 修改第一层卷积以适应灰度图像
                self.model.conv1 = nn.Conv2d(self.config.INPUT_CHANNELS, 2 * self.config.GROWTH_RATE, 
                                           kernel_size=3, padding=1, bias=False)
                
                # 修改最后的全连接层以适应表情分类
                self.model.linear = nn.Linear(self.model.linear.in_features, self.config.OUTPUT_SIZE)
                
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model_available = True
                self.status_message = f"已切换到{self.config.DENSENET_TYPE.upper()}模型"
            except Exception as e:
                self.model_available = False
                self.status_message = f"DenseNet模型加载失败（仅检测人脸）: {str(e)}"
        else:
            # 切换到CNN模型
            self.model_type = 'cnn'
            self.config = FaceCNNConfig()
            model_path = self.config.BEST_MODEL_PATH
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                self.model_available = False
                self.status_message = "已切换到CNN模型（仅检测人脸）"
                self.status_time = time.time()
                print(f"警告: CNN模型文件不存在: {model_path}")
                print("将只进行人脸检测，不进行表情识别")
                return
            
            try:
                self.model = FaceCNN(
                    input_size=self.config.IMAGE_SIZE,
                    use_batchnorm=self.config.USE_BATCHNORM,
                    activation=self.config.ACTIVATION,
                    keep_prob=1.0
                )
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model_available = True
                self.status_message = "已切换到CNN模型"
            except Exception as e:
                self.model_available = False
                self.status_message = f"CNN模型加载失败（仅检测人脸）: {str(e)}"
                
        self.status_time = time.time()
        print(self.status_message)
    
    def start_camera(self):
        """启动摄像头并开始实时检测"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        print("按 'q' 键退出")
        print("按 's' 键切换简单/详细模式")
        print("按 'r' 键开始/停止录制")
        print("按 'c' 键截图")
        print("按 'm' 键切换模型类型（CNN/VGG）")
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            if not ret:
                print("无法获取视频帧")
                break
                
            # 检测表情
            result_frame = self.detect_emotion(frame)
            
            # 如果正在录制，写入帧
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(result_frame)
            
            # 显示结果
            cv2.imshow('FacialExpressionPrediction', result_frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.simple_mode = not self.simple_mode
                mode_text = "简单" if self.simple_mode else "详细"
                self.status_message = f"切换到{mode_text}模式"
                self.status_time = time.time()
                print(self.status_message)
            elif key == ord('r'):
                self.toggle_recording(result_frame)
            elif key == ord('c'):
                self.take_screenshot(result_frame)
            elif key == ord('m'):
                self.switch_model()
                
        # 如果正在录制，停止录制
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='实时人脸表情识别')
    parser.add_argument('--model', type=str, default='cnn', 
                        choices=['cnn', 'vgg', 'resnet', 'densenet'], 
                        help='使用的模型类型: cnn, vgg, resnet 或 densenet (默认: cnn)')
    parser.add_argument('--simple', action='store_true', 
                        help='使用简单显示模式')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 创建检测器实例
        detector = RealtimeEmotionDetector(
            model_type=args.model,
            simple_mode=args.simple
        )
        # 启动摄像头检测
        detector.start_camera()
    except Exception as e:
        print(f"发生错误: {str(e)}") 