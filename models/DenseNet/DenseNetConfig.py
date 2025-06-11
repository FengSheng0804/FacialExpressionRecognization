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
        
        # 数据路径
        self.TRAIN_DATA_PATH = "dataset/train_set"
        self.VALID_DATA_PATH = "dataset/verify_set"
        
        # 模型保存路径
        self.MODEL_SAVE_PATH = "models/DenseNet/model_weight/facial_expression_model_DenseNet.pth"
        self.BEST_MODEL_PATH = "models/DenseNet/model_weight/best_facial_expression_model_DenseNet.pth"