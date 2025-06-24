import torch

class DenseNetConfig:
    def __init__(self):
        # 训练参数
        self.BATCH_SIZE = 128
        self.EPOCHS = 150
        self.LEARNING_RATE = 0.001
        self.MOMENTUM = 0.9
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优化器参数
        self.OPTIMIZER = 'adam'  # 'adam' 或 'sgd'
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.WEIGHT_DECAY = 1e-4
        
        # 学习率调度器参数
        self.USE_LR_SCHEDULER = True
        self.LEARNING_RATE_DECAY = 0.1
        self.DECAY_STEP = 30
        
        # 模型参数
        self.INPUT_CHANNELS = 1  # 灰度图像
        self.IMAGE_SIZE = 48
        self.OUTPUT_SIZE = 7  # 7种表情
          # DenseNet特定参数
        self.DENSENET_TYPE = 'densenet121'  # 'densenet121', 'densenet169', 'densenet201', 'densenet161'
        self.GROWTH_RATE = 32
        self.REDUCTION = 0.5
        self.PRETRAINED = False
        self.FEATURE_EXTRACT = False
        
        # CBAM注意力机制参数
        self.USE_CBAM = True  # 是否启用CBAM注意力机制
        self.CBAM_RATIO = 16  # CBAM通道注意力的缩放比例
        self.CBAM_KERNEL_SIZE = 7  # CBAM空间注意力的卷积核大小
        
        # 数据路径
        self.TRAIN_DATA_PATH = "dataset/train_set"
        self.VALID_DATA_PATH = "dataset/verify_set"
          # 模型保存路径
        cbam_suffix = "_cbam" if self.USE_CBAM else "_no_cbam"
        self.MODEL_SAVE_PATH = f"models/DenseNet/model_weight/facial_expression_model_{self.DENSENET_TYPE}{cbam_suffix}.pth"
        self.BEST_MODEL_PATH = f"models/DenseNet/model_weight/best_facial_expression_model_{self.DENSENET_TYPE}{cbam_suffix}.pth"