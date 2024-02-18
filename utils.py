import os                                   
import numpy as np
from tabulate import tabulate
import torch
import cv2
import json
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)










def visBatch(dataLoader, catNames):
    '''可视化训练集一个batch
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''

    for step, batch in enumerate(dataLoader):
        # 只可视化一个batch的图像：
        if step > 0: break
        imgs = batch[0]
        labels = batch[1].reshape(-1)  
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (8,10))
        for idx, [img, label] in enumerate(zip(imgs, labels)):
            img = img.numpy().transpose((2,1,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.title(catNames[label])
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.99, wspace=0.05, hspace=0.2)

        plt.show()

            








def normalInit(model, mean, stddev, truncated=False):
    '''权重按高斯分布初始化
        Args:
            :param model:     模型实例
            :param mean:      均值
            :param stddev:    标准差
            :param truncated: 截断

        Returns:
            None
    '''
    if truncated:
        model.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  
    else:
        model.weight.data.normal_(mean, stddev)
        model.bias.data.zero_()




def seed_everything(seed):
    '''设置全局种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def showComMatrix(trueList, predList, cat, evalDir):
    '''可视化混淆矩阵

    Args:
        :param trueList:  验证集的真实标签
        :param predList:  网络预测的标签
        :param cat:       所有类别的字典

    Returns:
        None
    '''
    if len(cat)>=50:
        # 100类正合适的大小  
        plt.figure(figsize=(40, 33))
        plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=0.99) 
    else:
        # 10类正合适的大小
        plt.figure(figsize=(12, 9))
        plt.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.99) 

    conf_mat = confusion_matrix(trueList, predList)
    df_cm = pd.DataFrame(conf_mat, index=cat, columns=cat)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 50, ha = 'right')
    plt.ylabel('true label')
    plt.xlabel('pred label')
    if not os.path.isdir(evalDir):os.makedirs(evalDir)
    # 保存图像
    plt.savefig(os.path.join(evalDir, '混淆矩阵.png'), dpi=200)
    plt.clf() 



def drawPRCurve(cat, trueList, softList, evalDir):
    '''绘制类别的PR曲线 

    Args:
        :param cat:  类别索引列表
        :param trueList:  验证集的真实标签
        :param softList:  网络预测的置信度
        :param evalDir:   PR曲线图保存路径

    Returns:
        None
    '''
    def calcPRThreshold(trueList, softList, clsNum, T):
        '''给定一个类别, 单个阈值下的PR值

        Args:
            :param trueList:  验证集的真实标签
            :param predList:  网络预测的标签
            :param clsNum:    类别索引

        Returns:
            precision, recall
        '''
        label = (trueList==clsNum)
        prob = softList[:,clsNum]>T
        TP = sum(label*prob)   # 正样本预测为正样本
        FN = sum(label*~prob)  # 正样本预测为负样本
        FP = sum(~label*prob)  # 负样本预测为正样本
        precision = TP / (TP + FP) if (TP + FP)!=0 else 1
        recall = TP / (TP + FN) 
        return precision, recall, T


    def clacPRCurve(trueList, softList, clsNum, interval=100):
        '''所有类别下的PR曲线值

        Args:
            :param trueList:  验证集的真实标签
            :param predList:  网络预测的标签
            :param clsNum:    类别索引列表
            :param interval:  阈值变化划分的区间，如interval=100, 则间隔=0.01

        Returns:
            :param PRs:       不同阈值下的PR值[2, interval, cat_num]
        '''
        PRs = []
        print('calculating PR per classes...')
        for cls in trange(clsNum):
            PR_value = [calcPRThreshold(trueList, softList, cls, i/interval) for i in range(interval+1)]
            PRs.append(np.array(PR_value))

        return np.array(PRs)

    
    plt.figure(figsize=(12, 9))
    # 计算所有类别下的PR曲线值
    PRs = clacPRCurve(trueList, softList, len(cat))
    # 绘制每个类别的PR曲线
    for i in range(len(cat)):
        PR = PRs[i]
        plt.plot(PR[:,1], PR[:,0], linewidth=1)
    plt.legend(labels=cat)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    # 保存图像 
    plt.savefig(os.path.join(evalDir, '类别PR曲线.png'), dpi=200)
    plt.clf()  
    return PRs



def clacAP(PRs, cat):
    '''计算每个类别的 AP, F1Score

    Args:
        :param PRs:  不同阈值下的PR值[2, interval, cat_num]
        :param cat:  类别索引列表

    Returns:
        None
    '''
    form = [['catagory', 'AP', 'F1_Score']]
    # 所有类别的平均AP与平均F1Score
    mAP, mF1Score = 0, 0
    for i in range(len(cat)):
        PR = PRs[i]
        AP = 0
        for j in range(PR.shape[0]-1):
            # 每小条梯形的矩形部分+三角形部分面积
            h = PR[j, 0] - PR[j+1, 0]
            w = PR[j, 1] - PR[j+1, 1]
            AP += (PR[j+1, 0] * w) + (w * h / 2)

            if(PR[j, 2]==0.5):
                F1Score0_5 = 2 * PR[j, 0] * PR[j, 1] / (PR[j, 0] + PR[j, 1])

        form.append([cat[i], AP, F1Score0_5])  
        mAP += AP
        mF1Score += F1Score0_5

    form.append(['average', mAP / len(cat), mF1Score / len(cat)]) 

    return mAP, mF1Score, tabulate(form, headers='firstrow') # tablefmt='fancy_grid'



def visArgsHistory(json_dir, save_dir):
    '''可视化训练过程中保存的参数

    Args:
        :param json_dir: 参数的json文件路径
        :param logDir:   可视化json文件保存路径

    Returns:
        None
    '''
    json_path = os.path.join(json_dir, 'args_history.json')
    with open(json_path) as json_file:
        args = json.load(json_file)
        for args_key in args.keys():
            arg = args[args_key]
            plt.plot(arg, linewidth=1)
            plt.xlabel('Epoch')
            plt.ylabel(args_key)
            plt.savefig(os.path.join(save_dir, f'{args_key}.png'), dpi=200)
            plt.clf()