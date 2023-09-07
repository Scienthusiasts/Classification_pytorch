import torch
import torch.nn as nn
from torchsummary import summary
# from collections import OrderedDict
import timm
'''
huggingface里的timm模型:
https://huggingface.co/timm?sort_models=downloads#models
'''



class Model(nn.Module):
    '''MobileNetv3
    '''
    def __init__(self, catNums:int, modelType:str, loadckpt=False, pretrain=True, froze=True):
        '''网络初始化

        Args:
            :param catNums:   数据集类别数
            :param modelType: 使用哪个模型(timm库里的模型)
            :param loadckpt:  是否导入模型权重
            :param pretrain:  是否用预训练模型进行初始化(是则输入权重路径)
            :param froze:     是否只训练分类头

        Returns:
            None
        '''
        super(Model, self).__init__()
        # 模型接到线性层的维度
        modelList = {
            'mobilenetv3_small_100.lamb_in1k':            1024, 
            'mobilenetv3_large_100':                      1280,
            'vit_base_patch16_224.augreg2_in21k_ft_in1k': 768, 
            'efficientnet_b5.sw_in12k_ft_in1k':           2048,
            'resnet50.a1_in1k':                           2048,
            }
        # 加载模型
        self.model = timm.create_model(modelType, pretrained=pretrain)
        # 是否只训练线性层
        if froze:
            for param in self.model.parameters():
                param.requires_grad_(False)
        # 修改分类头分类数
        baseModel = modelType.split('.')[0]
        if(baseModel in ['mobilenetv3_small_100', 'mobilenetv3_large_100', 'efficientnet_b5']):
            self.model.classifier = nn.Linear(modelList[modelType], catNums)
        if(baseModel=='vit_base_patch16_224'):
            self.model.head = nn.Linear(modelList[modelType], catNums)
        if(baseModel=='resnet50'):
            self.model.fc = nn.Linear(modelList[modelType], catNums)
        # 是否导入预训练权重
        if loadckpt: 
            self.load_state_dict(torch.load(loadckpt))

    def forward(self, x):
        '''前向传播
        '''
        x = self.model(x)
        return x
    





# for test only
if __name__ == '__main__':
    net = Model(catNums=10, modelType='mobilenetv3_small_100.lamb_in1k', pretrain=False).to('cuda')
    # net = ViT(catNums=10, modelType='vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrain=False).to('cuda')
    # 验证 1
    print(net)
    # 验证 2
    # summary(net, input_size=[(3, 224, 224)])  


