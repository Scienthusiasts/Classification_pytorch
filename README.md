# pytorch图像分类pipeline

基于对`timm`库提供的模型进行微调实现图像分类

## 环境配置

```
conda create -n pytorch_classify python=3.9
pip install -r requirenments.txt
```

## 训练

将`config.py`的mode更改为'train'

```
python runner.py --config ./config.py
```

## 验证

将`config.py`的mode更改为'eval'

`ckpt_load_path`为训练的权重

`eval_log_dir`为训练日志保存根目录

```
python runner.py --config ./config.py
```

## 测试

将`config.py`的mode更改为'test'

`img_path`为测试图片路径

`save_res_dir`为推理结果保存路径

```
python runner.py --config ./config.py
```

## 基于`food-101`数据集的实验效果

数据集下载：[Food 101 (kaggle.com)](https://www.kaggle.com/datasets/dansbecker/food-101)

| model(`timm`)                   | param(M) | accuracy | mAP  | mF1Score | Download                                                     |
| ------------------------------- | -------- | -------- | ---- | -------- | ------------------------------------------------------------ |
| `mobilenetv3_large_100.ra_in1k` | 4.33     | 0.82     | 0.88 | 0.83     | [best.pt](https://pan.baidu.com/s/12Qu4jZbMaR-E8DZoJR1mJQ?pwd=n73v) |

训练/验证日志获取：https://pan.baidu.com/s/1I0qrhBkTimHkKQ8_taNYdg?pwd=q8la 

