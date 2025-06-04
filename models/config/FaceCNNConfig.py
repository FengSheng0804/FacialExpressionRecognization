import torch

class FaceCNNConfig:
    """
    FaceCNN模型的配置类，包含所有训练和模型参数
    """
    def __init__(self):
        # 训练参数
        self.BATCH_SIZE = 128               # 与Keras模型一致的批次大小
        self.EPOCHS = 300                   # 训练轮数，Keras模型为1200，但我们可以使用早停机制
        self.LEARNING_RATE = 0.1            # 与Keras模型的Adadelta优化器初始学习率一致
        self.MOMENTUM = 0.95                # 与Keras的Adadelta rho参数一致
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优化器参数
        self.OPTIMIZER = 'adam'             # 可选：'adam' 或 'momentum'
        self.BETA1 = 0.9                    # Adam优化器参数
        self.BETA2 = 0.999                  # Adam优化器参数
        self.WEIGHT_DECAY = 1e-4            # 权重衰减
        
        # 学习率调度器参数
        self.USE_LR_SCHEDULER = True        # 是否使用学习率调度器
        self.LEARNING_RATE_DECAY = 0.1      # 学习率衰减率
        self.DECAY_STEP = 30                # 学习率衰减步长
        
        # Dropout参数
        self.KEEP_PROB = 0.5                # Dropout保留率
        
        # 激活函数
        self.ACTIVATION = 'relu'            # 可选：'relu'、'prelu'
        
        # 批归一化参数
        self.USE_BATCHNORM = True           # 是否使用批归一化
        self.USE_BATCHNORM_AFTER_CONV = True    # 卷积层后是否使用批归一化
        self.USE_BATCHNORM_AFTER_FC = False     # 全连接层后是否使用批归一化
        
        # 模型参数
        self.INPUT_CHANNELS = 1             # 输入图像通道数
        self.IMAGE_SIZE = 48                # 输入图像尺寸
        self.OUTPUT_SIZE = 7                # 输出类别数
        
        # 特征参数
        self.USE_LANDMARKS = False          # 是否使用特征点
        self.USE_HOG = False                # 是否使用HOG特征
        
        # 数据路径
        self.TRAIN_DATA_PATH = "dataset/train_set"
        self.VALID_DATA_PATH = "dataset/verify_set"
        
        # 模型保存路径
        self.MODEL_SAVE_PATH = "./models/model_weight/facial_expression_model_CNN.pth"
        self.BEST_MODEL_PATH = "./models/model_weight/best_facial_expression_model_CNN.pth"
        
        # 可视化参数
        self.PLOT_SAVE_PATH = "./results/training_plot_CNN.png"