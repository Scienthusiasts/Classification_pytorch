import numpy as np
import torch
from PIL import Image
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
import albumentations as A
import os
import cv2
# 自定义
from utils import seed_everything



class Transforms():
    '''数据预处理/数据增强(基于albumentations库)
    '''
    def __init__(self, imgSize):
        # 训练时增强
        self.trainTF = A.Compose([
                # 随机旋转
                A.Rotate(limit=15, p=0.5),
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=imgSize),
                # 随机镜像
                A.HorizontalFlip(p=0.5),
                # 参数：随机色调、饱和度、值变化
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                # 随机明亮对比度
                A.RandomBrightnessContrast(p=0.2),   
                # 高斯噪声
                A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),     
                A.OneOf([
                    # 使用随机大小的内核将运动模糊应用于输入图像
                    A.MotionBlur(p=0.2),   
                    # 中值滤波
                    A.MedianBlur(blur_limit=3, p=0.1),    
                    # 使用随机大小的内核模糊输入图像
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
                # 较短的边做padding
                A.PadIfNeeded(imgSize, imgSize, border_mode=cv2.BORDER_CONSTANT, value=[0,0,0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        # 验证时增强
        self.validTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=imgSize),
                # 较短的边做padding
                A.PadIfNeeded(imgSize, imgSize, border_mode=0, mask_value=[0,0,0]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        # 可视化增强(只reshape)
        self.visTF = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=imgSize),
                # 较短的边做padding
                A.PadIfNeeded(imgSize, imgSize, border_mode=0, mask_value=[0,0,0]),
            ])
    








class MyDataset(data.Dataset):      
    '''有监督分类任务对应的数据集读取方式
    '''
    def __init__(self, dir, mode, imgSize):    
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            :param dir:     图像数据集的根目录
            :param mode:    模式(train/valid)
            :param imgSize: 网络要求输入的图像尺寸

        Returns:
            precision, recall
        '''      
        self.tf = Transforms(imgSize = imgSize)
        # 记录数据集大小
        self.dataSize = 0      
        # 数据集类别数      
        self.labelsNum = len(os.listdir(os.path.join(dir, mode)))           
        # 训练/验证 
        self.mode = mode              
        # 数据预处理方法
        self.tf = Transforms(imgSize=imgSize)
        # 遍历所有类别
        self.imgPathList, self.labelList = [], []
        '''对类进行排序，很重要!!!，否则会造成分类时标签匹配不上导致评估的精度很低(默认按字符串,如果类是数字还需要更改)'''
        catDirs = sorted(os.listdir(os.path.join(dir, mode)))
        for idx, cat in enumerate(catDirs):
            catPath = os.path.join(dir, mode, cat)
            labelFiles = os.listdir(catPath)
            # 每个类别里图像数
            length = len(labelFiles)
            # 存放图片路径
            self.imgPathList += [os.path.join(catPath, labelFiles[i]) for i in range(length)]
            # 存放图片对应的标签(根据所在文件夹划分)
            self.labelList += [idx for _ in range(length)]
            self.dataSize += length        

    def __getitem__(self, item):  
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        # 读取图片
        img = Image.open(self.imgPathList[item]).convert('RGB')     
        img = np.array(img)
        # 获取image对应的label
        label = self.labelList[item]                 
        # 数据预处理/数据增强
        if self.mode=='train':
            transformed = self.tf.trainTF(image=img)
        if self.mode=='valid':
            transformed = self.tf.validTF(image=img)          
        img = transformed['image']    
        return img.transpose(2,1,0), torch.LongTensor([label])


    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.dataSize
    
    def get_cls_num(self):
        '''返回数据集类别数
        '''
        return self.labelsNum










# for test only
if __name__ == '__main__':
    datasetDir = 'E:/datasets/Classification/food-101/images'
    mode = 'train'
    bs = 64
    seed = 22
    seed_everything(seed)
    train_data = MyDataset(datasetDir, mode, imgSize=224)
    print(f'数据集大小:{train_data.__len__()}')
    print(f'数据集类别数:{train_data.get_cls_num()}')
    train_data_loader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True)
    # 获取label name
    catNames = sorted(os.listdir(os.path.join(datasetDir, mode)))
    # 可视化一个batch里的图像
    from utils import visBatch
    visBatch(train_data_loader, catNames)
    # 输出数据格式
    for step, batch in enumerate(train_data_loader):
        print(batch[0].shape, batch[1].shape)
        break