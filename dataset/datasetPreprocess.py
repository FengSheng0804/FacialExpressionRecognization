import csv
import os
import numpy as np
import pandas as pd
import cv2

def split_dataset(csv_file):
    datasets_path = os.path.dirname(csv_file)
    csv_file = os.path.join(datasets_path,"fer2013.csv")
    train_csv = os.path.join(datasets_path,"train.csv")
    val_csv = os.path.join(datasets_path,"val.csv")
    test_csv = os.path.join(datasets_path,"test.csv")

    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        print(header)
        rows = [row for row in csvr]
        
        trn = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
        print(len(trn))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
        print(len(val))        

        tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
        print(len(tst))

# 将emotion和pixels像素数据分离
def split_emotion_pixels():
    # 注意修改train.csv为你电脑上文件所在的相对或绝对路劲地址。
    path = './dataset/fer2013/train.csv'
    # 读取数据
    df = pd.read_csv(path)
    # 提取emotion数据
    df_y = df[['emotion']]
    # 提取pixels数据
    df_x = df[['pixels']]
    # 将emotion写入emotion.csv
    df_y.to_csv('./dataset/fer2013/emotion.csv', index=False, header=False)
    # 将pixels数据写入pixels.csv
    df_x.to_csv('./dataset/fer2013/pixels.csv', index=False, header=False)

# 将pixels转成图片
def pix2image(path):
    # 读取像素数据
    data = np.loadtxt('./dataset/fer2013/pixels.csv')
 
    # 按行取数据
    for i in range(data.shape[0]):
        face_array = data[i, :].reshape((48, 48)) # reshape
        cv2.imwrite(path + '//' + '{}.jpg'.format(i), face_array) # 写图片

# 创建映射表
def image_emotion_mapping(path):
    # 读取emotion文件
    df_emotion = pd.read_csv('dataset/fer2013/emotion.csv', header = None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(path)
    print(f"目录 {path} 下文件数：{len(files_dir)}")  # 新增调试
    # 用于存放图片名
    path_list = []
    # 用于存放图片对应的emotion
    emotion_list = []
    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某文件是图片，则将其文件名以及对应的emotion取出，分别放入path_list和emotion_list这两个列表中
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            try:
                index = int(os.path.splitext(file_dir)[0])
                emotion_list.append(df_emotion.iat[index, 0])
            except Exception as e:
                print(f"处理文件 {file_dir} 时出错: {e}")  # 新增调试
    print(f"找到图片数：{len(path_list)}")  # 新增调试

    # 将两个列表写进image_emotion.csv文件
    path_s = pd.Series(path_list, dtype='object')
    emotion_s = pd.Series(emotion_list, dtype='object')
    df = pd.DataFrame()
    df['path'] = path_s
    df['emotion'] = emotion_s
    df.to_csv(path+'\\image_emotion.csv', index=False, header=False)

if __name__ == "__main__":
    csv_file = "./dataset/fer2013/fer2013.csv"
    # 分离train、test、val
    split_dataset(csv_file)

    # 分离emotion和pixels数据
    split_emotion_pixels()
    
    # 将pix转成图片
    path = './dataset/face_images'
    pix2image(path)

    # 创建训练集和验证集目录
    train_set_path = './dataset/train_set'
    verify_set_path = './dataset/verify_set'
    os.makedirs(train_set_path, exist_ok=True)
    os.makedirs(verify_set_path, exist_ok=True)
    image_emotion_mapping(train_set_path)
    image_emotion_mapping(verify_set_path)

