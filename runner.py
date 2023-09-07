import os
import logging
import datetime
import torch                       
import torch.nn as nn                # 一些网络模块，比如卷积，池化，全连接等
import torch.utils.data as Data      # pytorch的数据集模块
import numpy as np                   # 矩阵运算
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import argparse
# CAM相关
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataloader import *              # 我们自定义的数据集读取模块
from utils import *
from mynet import Model








# 只训练分类头或只分类
def trainer(modelType:str, DatasetDir:str, Epoch:int, BatchSize:int, LearningRate:float, pretrain:bool, imgSize:int, ckptSavePath:str, logSaveDir:str):
    '''把pytorch训练代码独自分装成一个函数

    Args:
        :param modelType:    模型名称(timm)
        :param DatasetDir:   数据集根目录
        :param Epoch:        训练轮数
        :param BatchSize:    BatchSize
        :param LearningRate: 初始学习率
        :param pretrain:     是否使用预训练权重初始化网络
        :param imgSize:      网络接受的图像输入尺寸
        :param ckptSavePath: 权重保存路径
        :param logSaveDir:   训练日志保存目录

    Returns:
        None
    '''

    '''日志模块'''
    logger = logging.getLogger('runer')
    logger.setLevel(level=logging.DEBUG)
    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    # 写入文件的日志
    logSaveDir = os.path.join(logSaveDir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    if not os.path.isdir(logSaveDir):os.makedirs(logSaveDir)
    logSavePath = os.path.join(logSaveDir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt")
    file_handler = logging.FileHandler(logSavePath, encoding="utf-8", mode="a")
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 终端输出的日志
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'CUDA/CPU: {device}')

    # 记录最佳准确率(用于保存最佳权重)
    max_acc, best_EPOCH = 0,0

    '''导入数据集'''
    # 导入训练集
    train_data = MyDataset(DatasetDir, 'train', imgSize=imgSize)
    train_data_loader = Data.DataLoader(dataset = train_data, batch_size=BatchSize, shuffle=True, num_workers=2)
    logger.info('训练集大小:   %d' % train_data.__len__())
    # 导入验证集
    val_data = MyDataset(DatasetDir, 'valid', imgSize=imgSize)
    val_data_loader = Data.DataLoader(dataset = val_data, batch_size=BatchSize, shuffle=True, num_workers=2)
    logger.info('验证集大小:   %d' % val_data.__len__())
    cls_num = val_data.get_cls_num()
    logger.info('数据集类别数: %d'% cls_num)

    '''实例化tensorboard summaryWriter对象'''
    tb_writer = SummaryWriter(log_dir=os.path.join(logSaveDir, logSavePath.split('.')[0]))

    '''导入网络，定义优化器，损失函数'''
    # 加载网络
    net = Model(catNums=cls_num, modelType=modelType, pretrain=pretrain).to(device)
    # 定义优化器(自适应学习率的带动量梯度下降方法)
    optimizer = torch.optim.AdamW(net.parameters(), lr = LearningRate)
    # 使用余弦退火学习率
    scheduler =  lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=Epoch) #  * iters 
    # 定义损失函数(多分类交叉熵损失)
    loss_func = nn.CrossEntropyLoss()

    logger.info(f'优化器: {optimizer}')
    logger.info(f'损失函数: {loss_func}')
    logger.info('=' * 120)

    '''开始训练'''
    for epoch in range(Epoch):
        train_loss, train_correct, train_acc = 0, 0, 0
        val_loss, val_correct, val_acc = 0, 0, 0
        # 训练模式
        net.train()
        train_step = 0
        valid_step = 0
        
        trainbar = tqdm(train_data_loader)
        for batch in trainbar:
            trainbar.set_description("train epoch %d" % epoch)
            # [bs, channel, w, h] -> [bs, w*h, channel]
            b_x = batch[0].to(device)
            b_y = batch[1].squeeze().to(device)   # 标签[batch_size, 1]
            # 前向传播
            output = net(b_x) # [batchsize, cls_num]
            # 计算loss
            loss = loss_func(output, b_y)
            # 反向传播
            optimizer.zero_grad() # 将上一次迭代计算的梯度清零
            loss.backward()       # 反向传播计算梯度

            # 预测结果对应置信最大的那个下标
            pre_lab = torch.argmax(output, 1)
            # 计算一个batchsize的准确率
            train_correct = torch.sum(pre_lab == b_y.data)  

            # 计算一个epoch的损失和精度：
            train_loss += loss.item() * b_x.shape[0]
            train_acc += train_correct 
            # 记录log
            tb_writer.add_scalar('train_loss', loss.item(), train_step)
            tb_writer.add_scalar('train_acc', train_correct/b_x.shape[0], train_step)
            # 更新参数
            optimizer.step()  
            train_step += 1    
        # 更新学习率
        scheduler.step() 

        # 验证：
        net.eval()
        validbar = tqdm(val_data_loader)
        # 验证时无需计算梯度
        with torch.no_grad():
            for batch in validbar:
                validbar.set_description("valid epoch %d" % epoch)
                val_x = batch[0].to(device)
                val_y = batch[1].to(device).squeeze()   # [batch_size, 1]

                # 前向传播
                output = net(val_x)
                # 计算loss
                loss = loss_func(output, val_y)
                # 预测结果对应置信最大的那个下标
                pre_lab = torch.argmax(output, dim=1)
                val_correct = torch.sum(pre_lab == val_y.data)  # 计算一个batchsize的准确率

                # 计算一个epoch的损失和精度：
                val_loss += loss.item() * val_x.shape[0]
                val_acc += val_correct 
                # 记录log
                tb_writer.add_scalar('valid_loss', loss.item(), valid_step)
                tb_writer.add_scalar('valid_acc', val_correct / val_x.shape[0], valid_step)
                valid_step += 1 

        # 如果本次Epoch的acc最大，则保存参数(网络权重)
        if (float(val_acc) / val_data.__len__()) > max_acc:
            max_acc = float(val_acc) / val_data.__len__()
            best_EPOCH = epoch
            torch.save(net.state_dict(), os.path.join(logSaveDir, ckptSavePath))
            print('checkpoint has saved !')

        # 可视化一个epoch训练效果: 
        log = ("Epoch: %d | lr:%.8f | train loss:%.6f | train accuracy:%.5f | valid loss:%.6f | valid accuracy:%.5f | best Valid Epoch: %d" % 
              (epoch, scheduler.get_last_lr()[0], train_loss / train_data.__len__(), float(train_acc)/ train_data.__len__(), val_loss / val_data.__len__(), float(val_acc)/ val_data.__len__(), best_EPOCH))
        logger.info(log)


    '''结果评估'''
    DatasetDir = './data/IN10'
    net.load_state_dict(torch.load(os.path.join(logSaveDir, ckptSavePath)))
    predList, trueList, softList = eval(DatasetDir, 32, net, 224)
    cat = os.listdir(os.path.join(DatasetDir, 'valid'))
    # 可视化混淆矩阵
    showComMatrix(trueList, predList, cat, logSaveDir)
    # 绘制损失，学习率，准确率曲线
    readLog(logSavePath, logSaveDir)
    # 绘制PR曲线
    PRs = drawPRCurve(cat, trueList, softList, logSaveDir)
    # 计算每个类别的 AP, F1Score
    form = clacAP(PRs, cat)
    logger.info(f'\n{form}')






# 测试
def tester(modelType:str, cats:int, imgPath:str, imgSize:int, ckptSavePath:str, resultDir:str):
    '''把pytorch测试代码独自分装成一个函数

    Args:
        :param modelType:    模型名称(timm)
        :param cats:         类别名列表
        :param imgSize:      网络接受的图像输入尺寸
        :param ckptSavePath: 权重保存路径
        :param resultDir:    测试结果目录

    Returns:
        None
    '''
    # CPU/GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'CUDA/CPU: {device}')
    # 导入数据
    img = Image.open(imgPath)
    tf = Transforms(imgSize = imgSize).validTF
    Img = tf(img).unsqueeze(0).to(device)
    # 加载网络
    net = Model(catNums=len(cats), modelType=modelType, loadckpt=ckptSavePath, pretrain=False, froze=False).to(device)
    net.eval()
    # 预测
    logits = (net(Img).softmax(dim=-1).cpu().detach().numpy())
    pred = np.argmax(logits)
    print(cats)

    '''CAM(存在CAM为均为0的bug)'''
    # # CAM需要网络能反传梯度, 否则会报错
    # # 要可视化网络哪一层的CAM
    # target_layers = [net.model.conv_head]
    # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    # # 要关注的区域对应的类别
    # print(pred)
    # targets = [ClassifierOutputTarget(pred)]
    # grayscale_cam = cam(input_tensor=Img, targets=targets)[0]
    # plt.imshow(grayscale_cam)
    # plt.show()

    '''可视化预测结果'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    # 在第一个子图中绘制图像
    ax1.set_title('image')
    ax1.axis('off')
    ax1.imshow(img)
    # 在第二个子图中绘制置信度(横向)
    ax2.barh(cats, logits.reshape(-1))
    ax2.set_title('classification')
    ax2.set_xlabel('confidence')
    # 将数值最大的条块设置为不同颜色
    bar2 = ax2.patches[pred]
    bar2.set_color('orange')
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.90)
    if not os.path.isdir(resultDir):os.makedirs(resultDir)
    plt.savefig(os.path.join(resultDir, 'res.jpg'), dpi=200)
    plt.clf() 

















def runner():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='train / test')
    parser.add_argument('--model', type=str, help='timm model name')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--ckpt_name', default='model.pt', type=str, help='checkpoint saved name')
    parser.add_argument('--dataset', type=str,  help='dataset path')
    parser.add_argument('--eopch', default=30, type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--log_dir', default='./log', type=str, help='wight, evaluation, log file etc. saved dir')
    parser.add_argument('--pretrain', type=bool, help='initialized with pretrained weight')

    parser.add_argument('--result_dir', default='./result', type=str, help='save test result. required when mode is test')
    parser.add_argument('--img_path', type=str, help='required when mode is test')
    args = parser.parse_args()
    # 训练
    if args.mode == 'train':
        trainer(args.model, args.dataset, args.eopch, args.bs, args.lr, args.pretrain, args.img_size, args.ckpt_name, args.log_dir)  
    # 测试
    if args.mode == 'test':
        cats = [cat for cat in os.listdir('./data/IN10/valid')]
        tester(args.model, cats, args.img_path, args.img_size, args.ckpt_name, args.result_dir)
    else:
        print("mode not valid. it must be 'train' or 'test'")







if __name__ == '__main__':
    runner()
