"""
知识蒸馏训练运行脚本
使用 wandb 记录训练过程
"""

import os
import sys
import torch
import wandb

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("知识蒸馏系统 - DenseNet201 → DenseNet121")
    print("=" * 60)
    
    # 初始化 wandb
    try:
        wandb.init(
            project="facial-expression-recognition",
            name=f"distillation_densenet201_to_121_alpha0.5_temp3.0",
            config={
                "learning_rate": 0.00005,
                "epochs": 30,
                "batch_size": 64,
                "distillation_alpha": 0.5,
                "temperature": 3.0,
                "grad_clip": 1.0,
                "teacher_model": "DenseNet201",
                "student_model": "DenseNet121_CBAM_AdaptiveGrowth",
                "dataset": "FER2013",
                "optimizer": "adam"
            },
            settings=wandb.Settings(init_timeout=180)
        )
        print("✅ wandb 初始化成功")
        use_wandb = True
    except Exception as e:
        print(f"⚠️ wandb 初始化失败: {e}")
        print("将继续训练，但不会记录到wandb")
        use_wandb = False
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA不可用，使用CPU")
    
    # 检查必要文件
    print("\n检查必要文件...")
    
    files_to_check = [
        "dataset/train_set",
        "dataset/verify_set",
        "models/DenseNet/model_weight/best_facial_expression_model_densenet121_cbam_adaptive_growth.pth"
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print("\n缺失文件:")
        for file_path in missing_files:
            print(f"✗ {file_path}")
        
        # 如果学生模型不存在，询问是否继续
        if "models/DenseNet/model_weight/best_facial_expression_model_densenet121_cbam_adaptive_growth.pth" in missing_files:
            response = input("\n学生模型文件不存在，是否继续训练（将使用随机初始化）？(y/n): ")
            if response.lower() != 'y':
                print("训练取消")
                return
    
    print("\n=== 优化的训练参数 ===")
    print("- 学习率: 0.00005")
    print("- 蒸馏权重: alpha=0.5")
    print("- 温度: T=3.0") 
    print("- 梯度裁剪: 1.0")
    print("- 批次大小: 64")
    print("- 训练轮数: 30")
    print("- 早停耐心: 10")
    print("- wandb记录: 已启用")
    
    print("\n准备开始知识蒸馏训练...")
    print("🚀 训练过程将自动记录到 wandb 平台")
    
    try:
        # 设置环境变量来传递wandb状态
        os.environ['USE_WANDB'] = 'true' if use_wandb else 'false'
        
        # 导入并运行重构后的知识蒸馏模块
        from models.DenseNet.knowledge_distillation import main as kd_main
        kd_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装")
        print("正在尝试使用新的运行脚本...")
        try:
            os.system("python models/DenseNet/run_distillation.py")
        except Exception as e2:
            print(f"备用方案也失败: {e2}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 结束wandb会话
        if use_wandb:
            try:
                wandb.finish()
                print("✅ wandb会话已结束")
            except Exception as e:
                print(f"wandb结束失败: {e}")

if __name__ == "__main__":
    main()
