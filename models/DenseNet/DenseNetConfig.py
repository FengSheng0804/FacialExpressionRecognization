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
        
        # 自适应增长率参数
        self.USE_ADAPTIVE_GROWTH = True  # 是否启用自适应增长率
        self.ADAPTIVE_GROWTH_LIST = [16, 24, 32, 48]  # 可自定义每个dense block的增长率
        
        # CBAM注意力机制参数
        self.USE_CBAM = True  # 是否启用CBAM注意力机制
        self.CBAM_RATIO = 16  # CBAM通道注意力的缩放比例
        self.CBAM_KERNEL_SIZE = 7  # CBAM空间注意力的卷积核大小
        
        # 数据路径
        self.TRAIN_DATA_PATH = "dataset/train_set"
        self.VALID_DATA_PATH = "dataset/verify_set"
        
        # 模型保存路径
        cbam_suffix = "_cbam" if self.USE_CBAM else ""
        adaptive_suffix = "_adaptive_growth" if self.USE_ADAPTIVE_GROWTH else ""
        self.MODEL_SAVE_PATH = f"models/DenseNet/model_weight/facial_expression_model_{self.DENSENET_TYPE}{cbam_suffix}{adaptive_suffix}.pth"
        self.BEST_MODEL_PATH = f"models/DenseNet/model_weight/best_facial_expression_model_{self.DENSENET_TYPE}{cbam_suffix}{adaptive_suffix}.pth"
        
        # 多GPU训练配置
        self.USE_MULTI_GPU = True  # 是否自动启用多GPU训练（如果有多个GPU）
        self.USE_DISTRIBUTED = False  # 是否使用分布式训练（推荐用于多机多卡）
        self.SYNC_BATCH_NORM = True  # 是否在多GPU训练时同步批归一化
        self.GPU_IDS = None  # 指定使用的GPU ID列表，None表示使用所有可用GPU
        
        # 分布式训练配置
        self.MASTER_ADDR = 'localhost'
        self.MASTER_PORT = '12355'
        self.BACKEND = 'nccl'  # 分布式后端：nccl（推荐GPU）或gloo（CPU）
        
        # 性能优化配置
        self.PIN_MEMORY = True  # 启用内存固定以加速GPU数据传输
        self.NON_BLOCKING = True  # 启用非阻塞数据传输
        self.MIXED_PRECISION = False  # 是否启用混合精度训练（需要Apex或PyTorch 1.6+）
        self.COMPILE_MODEL = False  # 是否编译模型（PyTorch 2.0+）