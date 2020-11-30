# 定义数据读取器
import os
import cv2
import random
import numpy as np
import pandas as pd


# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img


# 定义训练集数据读取器
def data_loader(datadir, id, batch_size=10, mode='train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    df = pd.read_csv(os.path.join(datadir, 'label_{}.csv'.format(id)))
    datadir = os.path.join(datadir, str(id))
    filenames = os.listdir(datadir)
    DICT = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltrate': 3,
            'Mass': 4, 'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7, 'Covid-19': 8}

    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)

            label = df[df['path'] == name]
            label = label['label'].values[0]
            # print(name, label)
            label = DICT[label]
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


def reader_example(id):
    data_dict = {}
    for i in range(3):
        data_dict[str(i)] = np.random.rand(1, 5).astype('float32')
    data_dict["label"] = np.random.randint(2, size=(1, 1)).astype('int64')
    return data_dict


def mreader(id):
    return [reader_example(id)]


if __name__ == '__main__':
    reader = data_loader('data/data', 0, 16)
    for i in reader():
        print(i)
