import torch

class ResNetConfig:
    """
    ResNet模型的配置类，包含所有训练和模型参数
    """
    def __init__(self):
        # 训练参数
        self.BATCH_SIZE = 32                # 批次大小：每次训练时使用的样本数量
        self.EPOCHS = 100                   # 训练轮数：训练的次数
        self.LEARNING_RATE = 0.001          # 学习率：控制模型参数的更新速度
        self.MOMENTUM = 0.9                 # 动量：加速模型训练
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 优化器参数
        self.OPTIMIZER = 'adam'             # 优化器类型：'adam' 或 'sgd'
        self.BETA1 = 0.9                    # Adam优化器参数
        self.BETA2 = 0.999                  # Adam优化器参数
        self.WEIGHT_DECAY = 1e-4            # 权重衰减
        
        # 学习率调度器参数
        self.USE_LR_SCHEDULER = True        # 是否使用学习率调度器
        self.LEARNING_RATE_DECAY = 0.1      # 学习率衰减率
        self.DECAY_STEP = 30                # 学习率衰减步长
        
        # 模型参数
        self.INPUT_CHANNELS = 1             # 输入图像通道数
        self.IMAGE_SIZE = 48                # 输入图像尺寸
        self.OUTPUT_SIZE = 7                # 输出类别数
        
        # ResNet特定参数
        self.RESNET_TYPE = 'resnet50'       # ResNet类型：'resnet50'、'resnet101'、'resnet152'
        self.PRETRAINED = False             # 是否使用预训练权重
        self.FEATURE_EXTRACT = False        # 是否只训练分类器部分
        self.USE_BOTTLENECK = True          # 是否使用Bottleneck结构
        
        # 数据路径
        self.TRAIN_DATA_PATH = "dataset/train_set"
        self.VALID_DATA_PATH = "dataset/verify_set"
        self.TEST_DATA_PATH = "dataset/test_set"
        
        # 模型保存路径
        self.MODEL_SAVE_PATH = "./models/ResNet/model_weight/facial_expression_model_ResNet.pth"
        self.BEST_MODEL_PATH = "./models/ResNet/model_weight/best_facial_expression_model_ResNet.pth" 