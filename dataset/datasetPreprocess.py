import csv
import os
import random
import shutil
import numpy as np
import pandas as pd
import cv2

# 将emotion和pixels像素数据分离
def split_emotion_pixels(csv_file):
    """
    将emotion和pixels像素数据分离
    
    参数:
    csv_file: fer2013.csv文件路径
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    # 提取emotion数据
    df_y = df[['emotion']]
    # 提取pixels数据
    df_x = df[['pixels']]
    
    # 创建fer2013目录
    fer2013_dir = os.path.dirname(csv_file)
    os.makedirs(fer2013_dir, exist_ok=True)
    
    # 将emotion写入emotion.csv
    emotion_path = os.path.join(fer2013_dir, 'emotion.csv')
    df_y.to_csv(emotion_path, index=False, header=False)
    
    # 将pixels数据写入pixels.csv
    pixels_path = os.path.join(fer2013_dir, 'pixels.csv')
    df_x.to_csv(pixels_path, index=False, header=False)
    
    print(f"已将emotion和pixels数据分离并保存到：\n{emotion_path}\n{pixels_path}")
    return emotion_path, pixels_path

# 将pixels转成图片
def pix2image(pixels_path, output_path):
    """
    将pixels数据转换为图片
    
    参数:
    pixels_path: pixels.csv文件路径
    output_path: 输出图片目录路径
    """
    # 读取像素数据
    data = np.loadtxt(pixels_path)
    
    # 创建目标目录
    os.makedirs(output_path, exist_ok=True)
 
    # 按行取数据
    for i in range(data.shape[0]):
        face_array = data[i, :].reshape((48, 48)) # reshape
        cv2.imwrite(output_path + '//' + '{}.jpg'.format(i), face_array) # 写图片
    
    print(f"已将pixels数据转换为图片，共 {data.shape[0]} 张图片保存到 {output_path}")

# 对图像进行数据增强
def augment_images(source_dir, emotion_path):
    """
    对图像进行数据增强
    
    参数:
    source_dir: 源图像目录路径
    emotion_path: emotion.csv文件路径
    """
    # 读取emotion.csv文件
    df_emotion = pd.read_csv(emotion_path, header=None)
    # 统计每个类别的数量
    emotion_count = df_emotion[0].value_counts()
    print("各类别原始数量:")
    print(emotion_count)
    
    # 获取最多的类别数量
    max_count = emotion_count.max()
    print(f"最多的类别数量为: {max_count}")
    
    # 目标目录
    target_dir = './dataset/face_images_augmented'
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 为每个类别创建子目录
    for emotion in emotion_count.index:
        os.makedirs(os.path.join(target_dir, str(emotion)), exist_ok=True)
    
    # 获取source_dir中的所有图像
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    print(f"源目录中共有 {len(image_files)} 张图像")
    
    # 定义可用的增强方式
    augmentation_methods = [
        "original",    # 原始图像
        "horizontal_flip",  # 水平翻转
        "rotation",    # 旋转
        "translation"  # 平移
    ]
    
    # 按类别处理图像
    for emotion in emotion_count.index:
        # 该类别的目标目录
        emotion_dir = os.path.join(target_dir, str(emotion))
        
        # 获取该类别的所有图像索引
        emotion_indices = df_emotion.index[df_emotion[0] == emotion].tolist()
        
        # 将该类别的原始图像复制到目标目录
        for idx in emotion_indices:
            img_name = f"{idx}.jpg"
            if img_name in image_files:
                source_path = os.path.join(source_dir, img_name)
                target_path = os.path.join(emotion_dir, img_name)
                shutil.copy(source_path, target_path)
        
        # 获取复制后目录中的图像数量
        current_count = len(os.listdir(emotion_dir))
        print(f"类别 {emotion} 原始图像数量: {current_count}")
        
        # 如果该类别的图像数量已经达到最大值，则不需要增强
        if current_count >= max_count:
            print(f"类别 {emotion} 的数量已经达到目标，无需增强")
            continue
        
        # 需要增强的数量，但最多不超过原始数量的3倍（加上原始图像共4倍）
        max_augment_count = current_count * 3
        augment_count = min(max_count - current_count, max_augment_count)
        print(f"类别 {emotion} 需要增强 {augment_count} 张图像")
        
        # 获取该类别的所有图像
        images = os.listdir(emotion_dir)
        
        # 对每张图像进行增强
        count = 0
        for img_name in images:
            if count >= augment_count:
                break
                
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 对每张图像，每种增强方式最多使用一次
            # 随机打乱增强方式的顺序
            methods_to_use = augmentation_methods[1:]  # 排除"original"，因为原始图像已经复制
            random.shuffle(methods_to_use)
            
            # 对于每种增强方式
            for method in methods_to_use:
                if count >= augment_count:
                    break
                    
                aug_img = None
                
                if method == "horizontal_flip":
                    # 水平翻转
                    aug_img = cv2.flip(img, 1)
                elif method == "rotation":
                    # 旋转
                    angle = random.uniform(-20, 20)
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    aug_img = cv2.warpAffine(img, M, (w, h))
                elif method == "translation":
                    # 平移
                    h, w = img.shape[:2]
                    tx = random.uniform(-0.1, 0.1) * w
                    ty = random.uniform(-0.1, 0.1) * h
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    aug_img = cv2.warpAffine(img, M, (w, h))
                
                if aug_img is not None:
                    # 保存增强后的图像
                    save_path = os.path.join(emotion_dir, f'aug_{method}_{emotion}_{count}.jpg')
                    cv2.imwrite(save_path, aug_img)
                    count += 1
        
        print(f"类别 {emotion} 增强完成，现有图像 {len(os.listdir(emotion_dir))} 张")
    
    print("所有类别数据增强完成")
    
    # 删除源目录中的全部图像
    deleted_count = 0
    for file in os.listdir(source_dir):
        if file.endswith('.jpg'):
            file_path = os.path.join(source_dir, file)
            os.remove(file_path)
            deleted_count += 1
    
    print(f"已删除源目录 {source_dir} 中的 {deleted_count} 张图像")
    
    return target_dir

# 创建映射表
def create_emotion_csv(path, image_emotion_pairs):
    """
    根据图像和情感标签对创建emotion.csv文件
    
    参数:
    path: 目标目录路径
    image_emotion_pairs: 包含(图像文件名, 情感标签)对的列表
    """
    # 用于存放图片名
    path_list = [pair[0] for pair in image_emotion_pairs]
    # 用于存放图片对应的emotion
    emotion_list = [pair[1] for pair in image_emotion_pairs]

    # 将两个列表写进image_emotion.csv文件
    path_s = pd.Series(path_list, dtype='object')
    emotion_s = pd.Series(emotion_list, dtype='object')
    df = pd.DataFrame()
    df['path'] = path_s
    df['emotion'] = emotion_s
    df.to_csv(os.path.join(path, 'image_emotion.csv'), index=False, header=False)
    print(f"已创建 {path}/image_emotion.csv 映射文件")

# 分离训练集和验证集
def split_train_verify(augmented_dir, emotion_path, train_ratio=0.9):
    """
    将增强后的图像按比例分配到训练集和验证集
    完成后删除augmented_dir中的所有文件
    
    参数:
    augmented_dir: 增强后的图像目录
    emotion_path: emotion.csv文件路径
    train_ratio: 训练集比例，默认为0.9 (9:1)
    """
    print(f"开始按照 {train_ratio}:{1-train_ratio} 的比例划分训练集和验证集...")
    
    # 创建训练集和验证集目录
    train_set_path = './dataset/train_set'
    verify_set_path = './dataset/verify_set'
    os.makedirs(train_set_path, exist_ok=True)
    os.makedirs(verify_set_path, exist_ok=True)
    
    # 获取增强后的各类别目录
    emotion_dirs = [d for d in os.listdir(augmented_dir) if os.path.isdir(os.path.join(augmented_dir, d))]
    
    # 用于存储所有图像及其标签
    all_images = []  # 格式: [(image_filename, emotion), ...]
    
    # 收集所有图像及其标签
    for emotion_dir in emotion_dirs:
        emotion = int(emotion_dir)  # 类别标签
        emotion_path_full = os.path.join(augmented_dir, emotion_dir)
        # 获取该类别的所有图像
        images = [f for f in os.listdir(emotion_path_full) if f.endswith('.jpg')]
        
        # 添加到总列表中
        for img in images:
            all_images.append((img, emotion))
    
    print(f"共收集了 {len(all_images)} 张图像")
    
    # 随机打乱所有图像
    random.shuffle(all_images)
    
    # 按比例划分训练集和验证集
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    verify_images = all_images[split_idx:]
    
    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(verify_images)} 张图像")
    
    # 复制图像到各自目录
    for img, emotion in train_images:
        # 找到原始图像路径
        src_path = os.path.join(augmented_dir, str(emotion), img)
        # 复制到训练集
        shutil.copy(src_path, os.path.join(train_set_path, img))
    
    for img, emotion in verify_images:
        # 找到原始图像路径
        src_path = os.path.join(augmented_dir, str(emotion), img)
        # 复制到验证集
        shutil.copy(src_path, os.path.join(verify_set_path, img))
    
    # 创建训练集和验证集的emotion.csv文件
    create_emotion_csv(train_set_path, train_images)
    create_emotion_csv(verify_set_path, verify_images)
    
    print(f"已完成训练集和验证集的划分")
    
    # 删除augmented_dir中的所有文件和子目录
    deleted_files = 0
    deleted_dirs = 0
    
    # 先删除各个子目录中的文件
    for emotion_dir in emotion_dirs:
        emotion_path_full = os.path.join(augmented_dir, emotion_dir)
        for file in os.listdir(emotion_path_full):
            file_path = os.path.join(emotion_path_full, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted_files += 1
        
        # 然后删除子目录
        os.rmdir(emotion_path_full)
        deleted_dirs += 1
    
    print(f"已删除增强目录 {augmented_dir} 中的 {deleted_files} 个文件和 {deleted_dirs} 个子目录")

if __name__ == "__main__":
    print("开始处理数据集...")
    
    # 1. 分离emotion和pixels数据
    print("步骤1：分离emotion和pixels数据...")
    csv_file = "./dataset/fer2013/fer2013.csv"
    emotion_path, pixels_path = split_emotion_pixels(csv_file)
    
    # 2. 将pixels转成图片
    print("\n步骤2：将pixels转成图片...")
    face_images_path = './dataset/face_images'
    pix2image(pixels_path, face_images_path)
    
    # 3. 对图像进行数据增强，使每个类别的数量达到最多类别的数量
    print("\n步骤3：对图像进行数据增强...")
    augmented_dir = augment_images(face_images_path, emotion_path)
    
    # 4. 分离训练集和验证集
    print("\n步骤4：分离训练集和验证集...")
    split_train_verify(augmented_dir, emotion_path)
    
    print("\n数据处理完成！")