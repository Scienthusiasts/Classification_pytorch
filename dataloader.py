import os                                    # 读取文件目录
import torch.utils.data as data              # 自定义的数据集类都需要继承这个父类
from PIL import Image                        # 读取数据集
import torch
import torchvision.transforms as tf



class EasyTransforms():
    '''简单的数据预处理
    '''
    def convertImg2Gray(self, image):
        '''RGB 2 GRAY
        '''
        return image.convert("L")
    
    def __init__(self,): 
        self.imgSize = 64
        # 训练集预处理
        self.trainTF = tf.Compose([
            # self.convertImg2Gray,
            tf.Resize((self.imgSize, self.imgSize), interpolation=tf.InterpolationMode.BICUBIC),
            tf.ToTensor(),
        ])
        # 验证集预处理
        self.validTF = tf.Compose([
            # self.convertImg2Gray,
            tf.Resize((self.imgSize, self.imgSize), interpolation=tf.InterpolationMode.BICUBIC),
            tf.ToTensor(),
        ])




class Transforms():
    '''数据预处理
    '''
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __init__(self, imgSize): 
        '''
        Args:
            :param imgSize:     网络输入的图像尺寸

        Returns:
            None
        '''      
        self.imgSize = imgSize

        # 训练集预处理
        self.trainTF = tf.Compose([
                # 依概率p = 0.5水平翻转
                tf.RandomHorizontalFlip(),
                # 随机10度以内旋转
                tf.RandomRotation(10),
                # BICUBIC双三次差值(按比例缩放, 不会变形)
                tf.Resize(self.imgSize, interpolation=tf.InterpolationMode.BICUBIC),
                # tf.Resize((self.imgSize,self.imgSize),  interpolation=tf.InterpolationMode.BICUBIC),
                # 中心裁剪(正方形)
                tf.CenterCrop(self.imgSize),
                self._convert_image_to_rgb,
                tf.ToTensor(),
                tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # 验证集预处理
        self.validTF = tf.Compose([
            # BICUBIC双三次差值(按比例缩放, 不会变形)
            tf.Resize(self.imgSize, interpolation=tf.InterpolationMode.BICUBIC),
            # tf.Resize((self.imgSize,self.imgSize),  interpolation=tf.InterpolationMode.BICUBIC),
            # 中心裁剪(正方形)
            tf.CenterCrop(self.imgSize),
            self._convert_image_to_rgb,
            tf.ToTensor(),
            tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])








'''有监督分类任务对应的数据集读取方式'''
class MyDataset(data.Dataset):      

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
        # 数据预处理
        if mode=='train':
            self.transform = self.tf.trainTF
        if mode=='valid':
            self.transform = self.tf.validTF
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
        # 获取image对应的label
        label = self.labelList[item]                             
        return self.transform(img), torch.LongTensor([label])


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
    train_data = MyDataset('E:/datasets/Classification/food-101/images', 'train', imgSize=224)
    print(f'数据集大小:{train_data.__len__()}')
    print(f'数据集类别数:{train_data.get_cls_num()}')
    train_data_loader = data.DataLoader(dataset = train_data, batch_size=64, shuffle=True)
    # 可视化一个batch里的图像
    from utils import visBatch
    visBatch(train_data_loader)
    # 输出数据格式
    for step, batch in enumerate(train_data_loader):
        # print(batch[0])
        print(batch[0].shape, batch[1].shape)
        break
