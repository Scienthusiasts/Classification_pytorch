'''utils.py 用于评估, 测试相关'''
import os                                    # 读取文件目录
import torch.utils.data as data              # 自定义的数据集类都需要继承这个父类
import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn  
from tqdm import tqdm, trange
import torch.utils.data as Data              # pytorch的数据集模块
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix # 各种评估指标
from dataloader import *              # 我们自定义的数据集读取模块

plt.rcParams['font.sans-serif']=['SimHei']   # 用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False     # 用来正常显示负号
plt.rc('font', family='Times New Roman')





def visBatch(dataLoader:data.DataLoader):
    '''可视化训练集一个batch(8x8)
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''
    for step, batch in enumerate(dataLoader):
        b_x = batch[0]             
        b_y = batch[1]  
        #只可视化一个batch的图像：
        if step > 0:
            break
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (6,6))
        for img in np.arange(len(b_y)):
            plt.subplot(8,8,img + 1)
            image = b_x[img,:,:,:].numpy().transpose((1,2,0))
            # 由于在数据预处理时我们对数据进行了标准归一化，可视化的时候需要将其还原
            image = image * std + mean
            plt.imshow(image)
            # 在图像上方展示对应的标签
            plt.title(b_y[img][0].numpy())   
            # 取消坐标轴
            plt.axis("off")           
             # 微调行间距            
            plt.subplots_adjust(hspace = 0.5)    
        plt.show()







def eval(DatasetDir:str, BatchSize:int, net:nn.Module, imgSize:int):
    '''得到网络在验证集的真实标签true_list, 预测标签pred_list, 置信度soft_list

    Args:
        :param DatasetDir:   数据集根目录
        :param BatchSize:    BatchSize
        :param net:          网络
        :param imgSize:      图像尺寸



    Returns:
        None
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 记录真实标签和预测标签
    pred_list, true_list, soft_list = [], [], []
    # 导入验证集
    val_data = MyDataset(DatasetDir, 'valid', imgSize=imgSize)
    val_data_loader = Data.DataLoader(dataset = val_data, batch_size=BatchSize, shuffle=True, num_workers=2)
    # 导入网络
    net = net.to(device)
    # 验证：
    net.eval()

    # 验证时无需计算梯度
    with torch.no_grad():
        for batch in tqdm(val_data_loader):
            val_x = batch[0].to(device)   # [batch_size, 3, 64, 64]
            val_y = batch[1].to(device).squeeze()   # [batch_size, 1]
            # 前向传播
            output = net(val_x)
            # 预测结果对应置信最大的那个下标
            pre_lab = torch.argmax(output, dim=1)
            # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
            true_list += list(val_y.cpu().detach())
            pred_list += list(pre_lab.cpu().detach())
            soft_list += list(np.array(output.softmax(dim=-1).cpu().detach()))

        return np.array(pred_list), np.array(true_list), np.array(soft_list)








def showComMatrix(trueList, predList, cat, evalDir):
    '''可视化混淆矩阵

    Args:
        :param trueList:  验证集的真实标签
        :param predList:  网络预测的标签
        :param cat:       所有类别的字典

    Returns:
        None
    '''
    # 100类正合适的大小  
    # plt.figure(figsize=(40, 33))
    # plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=0.99) 
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







def clacAP(PRs, cat):
    '''计算每个类别的 AP, F1Score

    Args:
        :param PRs:  不同阈值下的PR值[2, interval, cat_num]
        :param cat:  类别索引列表

    Returns:
        None
    '''
    form = [['catagory', 'AP', 'F1_Score']]
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
    return tabulate(form, headers='firstrow') # tablefmt='fancy_grid'
        



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
            :param interval:  阈值变化划分的区间

        Returns:
            :param PRs:       不同阈值下的PR值[2, interval, cat_num]
        '''
        PRs = []
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
    




def readLog(logPath, logDir):
    '''读取日志文件，绘制学习率，损失，准确率曲线

    Args:
        :param logPath: 类别索引列表
        :param logDir:  曲线图保存路径

    Returns:
        None
    '''
    lrList = []
    trainLossList, validLossList = [], []
    trainAccList, validAccList = [], []
    with open(logPath, 'r', encoding='utf8') as logFile:
        lines = logFile.readlines()[19:]
        for line in lines:
            line = line[32:].split(' | ')
            lrList.append(float(line[1].split(':')[1]))
            trainLossList.append(float(line[2].split(':')[1]))
            validLossList.append(float(line[4].split(':')[1]))
            trainAccList.append(float(line[3].split(':')[1])) 
            validAccList.append(float(line[5].split(':')[1])) 

    # 学习率曲线
    plt.figure()
    plt.plot(lrList, linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(os.path.join(logDir, 'Learning_Rate.png'), dpi=200)
    plt.cla()
    # 损失函数曲线
    plt.plot(trainLossList, linewidth=1)
    plt.plot(validLossList, linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(labels=['train', 'valid'])
    plt.savefig(os.path.join(logDir, 'Loss.png'), dpi=200)
    plt.cla()   
    # 准确率曲线
    plt.plot(trainAccList, linewidth=1)
    plt.plot(validAccList, linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Acc.')
    plt.legend(labels=['train', 'valid'])
    plt.savefig(os.path.join(logDir, 'Accuracy.png'), dpi=200)
    plt.cla()     




# for test only
if __name__ == '__main__':
    from mynet import Mobi_v3, ViT_16_224

    DatasetDir = './data/IN10'
    loadckpt = './log/2023-08-27-10-44-47/MobileNetv3.pt'
    net = Mobi_v3(catNums=10, loadckpt=loadckpt, pretrain=False)
    # 跑一遍验证集得到结果
    predList, trueList, softList = eval(DatasetDir, 32, net, 224)
    cat = os.listdir('./data/IN10/valid')
    evalDir = './log'
    # 可视化混淆矩阵
    # showComMatrix(trueList, predList, cat, evalDir)
    # 绘制PR曲线
    PRs = drawPRCurve(cat, trueList, softList, evalDir)
    # 绘制损失，学习率，准确率曲线
    readLog('./log_2023-08-26-15-10-17.txt', evalDir)
    # 计算每个类别的 AP, F1Score
    clacAP(PRs, cat)

