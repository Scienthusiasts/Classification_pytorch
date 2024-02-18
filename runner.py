import os
import logging
import datetime
import torch                       
import torch.nn as nn                
import torch.utils.data as Data      
import numpy as np                   
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
import argparse
import json
import importlib.util
# CAM相关
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# torch 相关 metrics https://lightning.ai/docs/torchmetrics/stable/
import torchmetrics
# 自定义
from dataloader import *              
from utils import *
from mynet import Model





class ArgsHistory():
    '''记录train或val过程中的一些变量(比如 loss, acc, lr等)
    '''
    def __init__(self, json_save_dir):
        self.json_save_dir = json_save_dir
        self.args_history_dict = {}

    def record(self, key, value):
        '''记录args
        Args:
            :param key:   要记录的当前变量的名字
            :param value: 要记录的当前变量的数值
            
        Returns:
            None
        '''
        # 可能存在json格式不支持的类型, 因此统一转成float类型
        value = float(value)
        # 如果日志中还没有这个变量，则新建
        if key not in self.args_history_dict.keys():
            self.args_history_dict[key] = []
        # 更新
        self.args_history_dict[key].append(value)

    def saveRecord(self):
        '''以json格式保存args
        '''
        if not os.path.isdir(self.json_save_dir):os.makedirs(self.json_save_dir) 
        json_save_path = os.path.join(self.json_save_dir, 'args_history.json')
        # 保存
        with open(json_save_path, 'w') as json_file:
            json.dump(self.args_history_dict, json_file)

    def loadRecord(self, json_load_dir):
        '''导入上一次训练时的args(一般用于resume)
        '''
        json_path = os.path.join(json_load_dir, 'args_history.json')
        with open(json_path, "r", encoding="utf-8") as json_file:
            self.args_history_dict = json.load(json_file)



class Runner():
    '''训练/验证/推理时的流程'''
    def __init__(self, timm_model_name, img_size, ckpt_load_path, dataset_dir, epoch, bs, lr, log_dir, log_interval, pretrain, froze, optim_type, mode, resume=None, seed=0):
        '''Runner初始化
        Args:
            :param timm_model_name: 模型名称(timm)
            :param img_size:        统一图像尺寸的大小
            :param ckpt_load_path:  预加载的权重路径
            :param dataset_dir:     数据集根目录
            :param eopch:           训练批次
            :param bs:              训练batch size
            :param lr:              学习率
            :param log_dir:         日志文件保存目录
            :param log_interval:    训练或验证时隔多少bs打印一次日志
            :param pretrain:        backbone是否用ImageNet预训练权重初始化
            :param froze:           是否冻结Backbone只训练分类头
            :param optim_type:      优化器类型
            :param mode:            训练模式:train/eval/test
            :param resume:          是否从断点恢复训练
            :param seed:            固定全局种子

        Returns:
            None
        '''
        # 设置全局种子
        seed_everything(seed)
        self.timm_model_name = timm_model_name
        self.img_size = img_size
        self.ckpt_load_path = ckpt_load_path
        self.dataset_dir = dataset_dir
        self.epoch = epoch
        self.bs = bs
        self.lr = lr
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.pretrain = pretrain
        self.froze = froze
        self.mode = mode
        self.optim_type = optim_type
        self.cats = os.listdir(os.path.join(self.dataset_dir, 'valid'))
        self.cls_num = len(self.cats)
        '''GPU/CPU'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''日志模块'''
        if mode == 'train' or mode == 'eval':
            self.logger, self.log_save_path = self.myLogger()
            '''训练/验证时参数记录模块'''
            json_save_dir, _ = os.path.split(self.log_save_path)
            self.argsHistory = ArgsHistory(json_save_dir)
        '''实例化tensorboard summaryWriter对象'''
        if mode == 'train':
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.log_save_path.split('.')[0]))
        '''导入数据集'''
        if mode == 'train':
            # 导入训练集
            self.train_data = MyDataset(dataset_dir, 'train', imgSize=img_size)
            self.train_data_loader = Data.DataLoader(dataset = self.train_data, batch_size=bs, shuffle=True, num_workers=2)
        if mode == 'train' or mode == 'eval':
            # 导入验证集
            self.val_data = MyDataset(dataset_dir, 'valid', imgSize=img_size)
            self.val_data_loader = Data.DataLoader(dataset = self.val_data, batch_size=1, shuffle=False, num_workers=2)
        '''导入模型'''
        self.model = Model(catNums=self.cls_num, modelType=timm_model_name, loadckpt=ckpt_load_path, pretrain=pretrain, froze=froze).to(self.device)
        '''定义损失函数(多分类交叉熵损失)'''
        if mode == 'train' or mode == 'eval':
            self.loss_func = nn.CrossEntropyLoss()
        '''定义优化器(自适应学习率的带动量梯度下降方法)'''
        if mode == 'train':
            self.optimizer, self.scheduler = self.defOptimSheduler()
        '''当恢复断点训练'''
        self.start_epoch = 0
        if resume != None:
            checkpoint = torch.load(resume)
            self.start_epoch = checkpoint['epoch'] + 1 # +1是因为从当前epoch的下一个epoch开始训练
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.scheduler.load_state_dict(checkpoint['sched_state_dict'])
            # 导入上一次中断训练时的args
            json_dir, _ = os.path.split(resume)
            self.argsHistory.loadRecord(json_dir)
        # 打印日志
        if mode =='train' or mode =='eval':
            if mode == 'train':
                self.logger.info('训练集大小:   %d' % self.train_data.__len__())
            self.logger.info('验证集大小:   %d' % self.val_data.__len__())
            self.logger.info('数据集类别数: %d' % self.cls_num)
            if mode == 'train':
                self.logger.info(f'损失函数: {self.loss_func}')
                self.logger.info(f'优化器: {self.optimizer}')
            self.logger.info(f'全局种子: {seed}')
            self.logger.info('='*100)



    def defOptimSheduler(self):
        '''定义优化器和学习率衰减策略
        '''
        optimizer = {
            # adam会导致weight_decay错误，使用adam时建议设置为 0
            'adamw' : torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0),
            'adam' : torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0),
            'sgd'  : torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True, weight_decay=0)
        }[self.optim_type]
        # 使用warmup+余弦退火学习率
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=self.epoch*len(self.train_data_loader),          # 总迭代数
            lr_min=self.lr*0.01,                                       # 余弦退火最低的学习率
            warmup_t=round(self.epoch/12)*len(self.train_data_loader), # 学习率预热阶段的epoch数量
            warmup_lr_init=self.lr*0.01,                               # 学习率预热阶段的lr起始值
        )

        return optimizer, scheduler



    def myLogger(self):
        '''生成日志对象
        '''
        logger = logging.getLogger('runer')
        logger.setLevel(level=logging.DEBUG)
        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

        if self.mode == 'train':
            # 写入文件的日志
            self.log_dir = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_train")
            # 日志文件保存路径
            log_save_path = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_train.log")
        if self.mode == 'eval':
            # 写入文件的日志
            self.log_dir = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_val")
             # 日志文件保存路径
            log_save_path = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_val.log")
        if not os.path.isdir(self.log_dir):os.makedirs(self.log_dir)
        file_handler = logging.FileHandler(log_save_path, encoding="utf-8", mode="a")
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # 终端输出的日志
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger, log_save_path




    def recoardArgs(self, mode, loss=None, acc=None, mAP=None, mF1Score=None):
        '''训练/验证过程中记录变量(每个iter都会记录, 不间断)
            Args:
                :param mode: 模式(train, epoch)
                :param loss: 损失
                :param acc:  准确率

            Returns:
                None
        '''        
        if mode == 'train':
            current_lr = self.optimizer.param_groups[0]['lr']
            self.argsHistory.record('lr', current_lr)
            self.argsHistory.record('train_loss', loss)
            self.argsHistory.record('train_acc', acc)
        # 一个epoch结束后val评估结果的平均值
        if mode == 'epoch':
            self.argsHistory.record('mean_val_acc', acc)
            self.argsHistory.record('val_mAP', mAP)
            self.argsHistory.record('val_mF1Score', mF1Score)
            self.argsHistory.saveRecord()



    def printLog(self, mode, step, epoch, batch_num):
        '''训练/验证过程中打印日志
            Args:
                :param mode:       模式(train, val, epoch)
                :param step:       当前迭代到第几个batch
                :param epoch:      当前迭代到第几个epoch
                :param batch_num:  当前batch的大小
                :param loss:       当前batch的loss
                :param acc:        当前batch的准确率
                :param best_epoch: 当前最佳模型所在的epoch

            Returns:
                None
        '''        
        lr = self.optimizer.param_groups[0]['lr']
        if mode == 'train':
            # 每间隔self.log_interval个iter才打印一次
            if step % self.log_interval == 0:
                loss = self.argsHistory.args_history_dict['train_loss'][-1]
                acc = self.argsHistory.args_history_dict['train_acc'][-1]
                log = ("Epoch(train)  [%d][%d/%d]  lr: %8f  train_loss: %5f  train_acc.: %5f") % (epoch+1, step, batch_num, lr, loss, acc)
                self.logger.info(log)

        elif mode == 'epoch':
            acc_list = self.argsHistory.args_history_dict['mean_val_acc']
            mAP_list = self.argsHistory.args_history_dict['val_mAP']
            mF1Score_list = self.argsHistory.args_history_dict['val_mF1Score']
            # 找到最高准确率对应的epoch
            best_epoch = acc_list.index(max(acc_list)) + 1
            self.logger.info('=' * 100)
            log = ("Epoch  [%d]  mean_val_acc.: %.5f  mAP: %.5f  mF1Score: %.5f  best_val_epoch: %d" % (epoch+1, acc_list[-1], mAP_list[-1], mF1Score_list[-1], best_epoch))
            self.logger.info(log)
            self.logger.info('=' * 100)



    def recordTensorboardLog(self, mode, epoch, batch_num=None, step=None):
        '''训练过程中记录tensorBoard日志
            Args:
                :param mode:       模式(train, val, epoch)
                :param step:       当前迭代到第几个batch
                :param batch_num:  当前batch的大小

            Returns:
                None
        '''    
        if mode == 'train':
            step = epoch * batch_num + step
            loss = self.argsHistory.args_history_dict['train_loss'][-1]
            acc = self.argsHistory.args_history_dict['train_acc'][-1]
            self.tb_writer.add_scalar('train_loss', loss, step)
            self.tb_writer.add_scalar('train_acc', acc, step)
        if mode == 'epoch':
            acc = self.argsHistory.args_history_dict['mean_val_acc'][-1]
            mAP = self.argsHistory.args_history_dict['val_mAP'][-1]
            mF1Score = self.argsHistory.args_history_dict['val_mF1Score'][-1]
            self.tb_writer.add_scalar('mean_valid_acc', acc, epoch)
            self.tb_writer.add_scalar('valid_mAP', mAP, epoch)
            self.tb_writer.add_scalar('valid_mF1Score', mF1Score, epoch)



    def saveCkpt(self, epoch):
        '''保存权重和训练断点
            Args:
                :param epoch:        当前epoch
                :param max_acc:      当前最佳模型在验证集上的准确率
                :param mean_val_acc: 当前epoch准确率
                :param best_epoch:   当前最佳模型对应的训练epoch

            Returns:
                None
        '''  
        # checkpoint_dict能够恢复断点训练
        checkpoint_dict = {
            'epoch': epoch, 
            'model_state_dict': self.model.state_dict(), 
            'optim_state_dict': self.optimizer.state_dict(),
            'sched_state_dict': self.scheduler.state_dict()
            }
        torch.save(checkpoint_dict, os.path.join(self.log_dir, f"epoch_{epoch}.pt"))
        # 如果本次Epoch的acc最大，则保存参数(网络权重)
        acc_list = self.argsHistory.args_history_dict['mean_val_acc']
        if epoch == acc_list.index(max(acc_list)):
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best.pt'))
            self.logger.info('best checkpoint has saved !')




    def fitEpoch(self, epoch):
        '''对一个epoch进行训练的流程
        '''
        self.model.train()
        # 一个Epoch包含几轮Batch
        train_batch_num = len(self.train_data_loader)
        for step, batch in enumerate(self.train_data_loader):
            # [bs, channel, w, h] -> [bs, w*h, channel]
            with torch.no_grad():
                x = batch[0].to(self.device)
            y = batch[1].to(self.device).reshape(-1)   # 标签[batch_size, 1]
            # 前向传播
            output = self.model(x) # [batchsize, cls_num]
            # 计算loss
            loss = self.loss_func(output, y)
            # 预测结果对应置信最大的那个下标
            pre_lab = torch.argmax(output, 1)
            # 计算一个batchsize的准确率
            train_acc = torch.sum(pre_lab == y.data) / x.shape[0]
            # 记录args(lr, loss, acc)
            self.recoardArgs(mode='train', loss=loss.item(), acc=train_acc)
            # 记录tensorboard
            self.recordTensorboardLog('train', epoch, train_batch_num, step)
            # 打印日志
            self.printLog('train', step, epoch, train_batch_num) 
            # 将上一次迭代计算的梯度清零
            self.optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()       
            # 更新参数
            self.optimizer.step() 
            # 更新学习率
            self.scheduler.step(epoch * train_batch_num + step) 




    def eval(self):
        '''得到网络在验证集的真实标签true_list, 预测标签pred_list, 置信度soft_list, 给后续评估做准备
        '''
        # 记录真实标签和预测标签
        pred_list, true_list, soft_list = [], [], []
        # 验证模式
        self.model.eval()
        # 验证时无需计算梯度
        with torch.no_grad():
            print('evaluating val dataset...')
            for batch in tqdm(self.val_data_loader):
                x = batch[0].to(self.device)   # [batch_size, 3, 64, 64]
                y = batch[1].to(self.device).reshape(-1)  # [batch_size, 1]
                # 前向传播
                output = self.model(x)
                # 预测结果对应置信最大的那个下标
                pre_lab = torch.argmax(output, dim=1)
                # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
                true_list += list(y.cpu().detach())
                pred_list += list(pre_lab.cpu().detach())
                soft_list += list(np.array(output.softmax(dim=-1).cpu().detach()))

            return np.array(pred_list), np.array(true_list), np.array(soft_list)



    def trainer(self):
        '''把pytorch训练代码独自分装成一个函数
        '''
        for epoch in range(self.start_epoch, self.epoch):
            '''一个epoch的训练'''
            self.fitEpoch(epoch)
            '''一个epoch的验证'''
            self.evaler(epoch, self.log_dir)
            '''保存网络权重'''
            self.saveCkpt(epoch)
            '''打印日志(一个epoch结束)'''
            self.printLog('epoch', 0, epoch, len(self.val_data_loader))

        '''结果评估'''
        self.model.load_state_dict(torch.load(os.path.join(self.log_dir, 'best.pt')))
        # 评估(最佳权重)
        self.evaler(epoch, self.log_dir)



    def evaler(self, epoch, log_dir):
        '''把pytorch训练代码独自分装成一个函数

        Args:
            :param modelType:    模型名称(timm)
            :param DatasetDir:   数据集根目录(到images那一层, 子目录是train/valid)
            :param BatchSize:    BatchSize
            :param imgSize:      网络接受的图像输入尺寸
            :param ckptPath:     权重路径
            :param logSaveDir:   训练日志保存目录

        Returns:
            None
        '''

        # 得到网络预测结果
        # shape = [val_size,] [val_size,] [val_size, cls_num]
        predList, trueList, softList = self.eval()

        '''自定义的实现'''
        # 准确率
        acc = sum(predList==trueList) / predList.shape[0]
        self.logger.info(f'acc: {acc}')
        # 可视化混淆矩阵
        showComMatrix(trueList, predList, self.cats, self.log_dir)
        # 绘制PR曲线
        PRs = drawPRCurve(self.cats, trueList, softList, self.log_dir)
        # 计算每个类别的 AP, F1Score
        mAP, mF1Score, form = clacAP(PRs, self.cats)
        self.logger.info(f'\n{form}')
        # 记录args(epoch)
        self.recoardArgs(mode='epoch', acc=acc, mAP=mAP, mF1Score=mF1Score)
        # 绘制损失，学习率，准确率曲线
        visArgsHistory(log_dir, self.log_dir)
        # 记录tensorboard的log
        if self.mode == 'train':
            self.recordTensorboardLog('epoch', epoch)



    def tester(self, img_path, save_res_dir):
        '''把pytorch测试代码独自分装成一个函数

        Args:
            :param img_path:     测试图像路径
            :param save_res_dir: 推理结果保存目录

        Returns:
            None
        '''
        from dataloader import Transforms
        # 加载一张图片并进行预处理
        image = Image.open(img_path)
        image = np.array(image)
        tf = Transforms(imgSize = self.img_size)
        visImg = tf.visTF(image=image)['image']
        img = torch.tensor(tf.validTF(image=image)['image']).permute(2,1,0).unsqueeze(0).to(self.device)
        # 加载网络
        self.model.eval()
        # 预测
        logits = self.model(img).softmax(dim=-1).cpu().detach().numpy()[0]
        sorted_id = sorted(range(len(logits)), key=lambda k: logits[k], reverse=True)
        # 超过10类则只显示top10的类别
        logits_top_10 = logits[sorted_id[:10]]
        cats_top_10 = [self.cats[i] for i in sorted_id[:10]]

        '''CAM'''
        # CAM需要网络能反传梯度, 否则会报错
        # 要可视化网络哪一层的CAM(以mobilenetv3_large_100.ra_in1k为例, 不同的网络这部分还需更改)
        target_layers = [self.model.backbone.blocks[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        # 要关注的区域对应的类别
        targets = [ClassifierOutputTarget(sorted_id[0])]
        grayscale_cam = cam(input_tensor=img, targets=targets)[0].transpose(1,0)
        visualization = show_cam_on_image(visImg / 255., grayscale_cam, use_rgb=True)

        '''可视化预测结果'''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        # 在第一个子图中绘制图像
        ax1.set_title('image')
        ax1.axis('off')
        # ax1.imshow(image)
        ax1.imshow(visualization)
        # 在第二个子图中绘制置信度(横向)
        ax2.barh(cats_top_10, logits_top_10.reshape(-1))
        ax2.set_title('classification')
        ax2.set_xlabel('confidence')
        # 将数值最大的条块设置为不同颜色
        bar2 = ax2.patches[0]
        bar2.set_color('orange')
        # y轴上下反转，不然概率最大的在最下面
        plt.gca().invert_yaxis()
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.90)
        if not os.path.isdir(save_res_dir):os.makedirs(save_res_dir)
        plt.savefig(os.path.join(save_res_dir, 'res.jpg'), dpi=200)
        plt.clf() 











def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    return args



def import_module_by_path(module_path):
    """根据给定的完整路径动态导入模块(config.py)
    """
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



if __name__ == '__main__':
    args = getArgs()
    # 使用动态导入的模块
    config_path = args.config
    config_file = import_module_by_path(config_path)
    # 调用动态导入的模块的函数
    config = config_file.config

    runner = Runner(config['timm_model_name'], config['img_size'], config['ckpt_load_path'],config['dataset_dir'], config['epoch'], config['bs'], config['lr'], 
                    config['log_dir'], config['log_interval'], config['pretrain'], config['froze'], config['optim_type'], config['mode'], config['resume'], config['seed'])
    # 训练
    if config['mode'] == 'train':
        runner.trainer()
    # 评估
    elif config['mode'] == 'eval':
        runner.evaler(epoch=0, log_dir=config['eval_log_dir'])
    elif config['mode'] == 'test':
        runner.tester(config['img_path'], config['save_res_dir'])
    else:
        print("mode not valid. it must be 'train', 'eval' or 'test'.")





