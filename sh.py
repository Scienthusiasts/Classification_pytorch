import os
# while training:
os.system("python runner.py --model mobilenetv3_large_100.ra_in1k --dataset E:/datasets/Classification/food-101/images --bs 64 --lr 1e-4 --ckpt_save_name mobilenetv3_large_100.ra_in1k.pt --pretrain True --froze True")
# while eval:
os.system("python runner.py  --mode eval --model efficientnet_b5.sw_in12k_ft_in1k --dataset E:/datasets/Classification/food-101/images --bs 64 --ckpt_load_path log/2023-09-29-16-25-06/efficientnet_b5.sw_in12k_ft_in1k.pt")
# while testing:
os.system("python runner.py --mode test --model mobilenetv3_large_100.ra_in1k --img_path data/IN10/valid/n03180011/n03180011_475.JPEG --ckpt_load_path log/2023-09-29-14-53-34/mobilenetv3_large_100.ra_in1k.pt")
