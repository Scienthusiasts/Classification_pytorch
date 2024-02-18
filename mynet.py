import torch
import torch.nn as nn
from torchsummary import summary
import timm
from utils import normalInit

'''
huggingface里的timm模型:
https://huggingface.co/timm?sort_models=downloads#models
'''



class Model(nn.Module):
    '''Backbone
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
            'mobilenetv3_large_100.ra_in1k':              1280,
            'vit_base_patch16_224.augreg2_in21k_ft_in1k': 768, 
            'efficientnet_b5.sw_in12k_ft_in1k':           2048,
            'resnetaa50d.d_in12k':                        2048, 
            'resnet50.a1_in1k':                           2048,
            'vgg16.tv_in1k':                              4096,
            }
        # 加载模型
        self.backbone = timm.create_model(modelType, pretrained=pretrain)
        # 删除原来的分类头并添加新的分类头(self.backbone就是去除了分类头的原始完整模型)
        baseModel = modelType.split('.')[0]
        if(baseModel in ['mobilenetv3_small_100', 'mobilenetv3_large_100', 'efficientnet_b5']):
            self.backbone.classifier = nn.Identity()
            self.head = nn.Linear(modelList[modelType], catNums)
        if(baseModel=='vit_base_patch16_224'):
            self.backbone.head = nn.Identity()
            self.head = nn.Linear(modelList[modelType], catNums)
        if(baseModel in ['resnet50', 'resnetaa50d']):
            self.backbone.fc = nn.Identity()
            self.head = nn.Linear(modelList[modelType], catNums)
        if(baseModel=='vgg16'):
            self.backbone.head.fc = nn.Identity()
            self.head = nn.Linear(modelList[modelType], catNums)
        # 分类头权重初始化
        normalInit(self.head, mean=0, stddev=0.01)
        # 是否导入预训练权重
        if loadckpt: 
            self.load_state_dict(torch.load(loadckpt))
        # 是否只训练线性层
        if froze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

                
    def forward(self, x):
        '''前向传播
        '''
        feat = self.backbone(x)
        out = self.head(feat)
        return out






# for test only
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Model(catNums=101, modelType='mobilenetv3_large_100.ra_in1k', pretrain=False, froze=False).to(device)

    
    # '''验证 1'''
    # print(model)
    # '''验证 2'''
    summary(model, input_size=[(3, 224, 224)])  
    # '''验证 3'''
    x = torch.rand((4, 3, 800, 800)).to(device)
    out = model(x)
    print(out.shape)


    '''导出预训练权重(去掉分类头)'''
    # exportWeight('vgg16.tv_in1k', './')


    '''导出原始权重'''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = timm.create_model('resnetaa50d.d_in12k', pretrained=False, features_only=True, out_indices=[3]).to(device)
    # # summary(model, input_size=[(3, 800, 800)])
    # print(model)
    # x = torch.rand((4, 3, 800, 800)).to(device)
    # out = model(x)
    # for o in out:
    #     print(o.shape)
    # # torch.save(model.state_dict(), './ckpt/mobilenetv3_large_100.ra_in1k.pt')



