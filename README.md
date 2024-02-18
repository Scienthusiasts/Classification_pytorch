# Classification_Pytorch
A toy example for Pytorch based image classification using deep models.



## 环境配置

```
conda create -n torch_classify python=3.9
pip install -r requirenments.txt
```

## 训练

在`config.py`中将mode更改为`train`

```
python runner.py --config config.py
```

## 验证

在`config.py`中将mode更改为`eval`

添加`eval_log_di`参数指向验证集图像根目录

```
python runner.py --config config.py
```

## 测试

在`config.py`中将mode更改为`test`

添加`img_path`参数指向图像路径

添加`save_res_dir`参数指向推理结果保存路径

```
python runner.py --config config.py
```



## 实验

| model                           | param(M) | accuracy | mAP  | mF1Score | download ckpt                                                |
| ------------------------------- | -------- | -------- | ---- | -------- | ------------------------------------------------------------ |
| `mobilenetv3_large_100.ra_in1k` | 4.33     | 0.82     | 0.88 | 0.83     | [best.pt](https://pan.baidu.com/s/1Oo8Y9bwgJofS3hxMHobV5Q?pwd=tit8) |

**完整训练和验证log日志获取：**https://pan.baidu.com/s/1j46UMjaSJlaS5S5JVLMh_g?pwd=kncb 
