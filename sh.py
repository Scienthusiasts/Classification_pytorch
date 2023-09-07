import os
# while training:
os.system("python runner.py --model mobilenetv3_small_100.lamb_in1k --dataset data/IN10 --bs 64 --lr 1e-4 --ckpt_name mobilenetv3.pt --pretrain True")
# while testing:
os.system("python runner.py --mode test --model mobilenetv3_small_100.lamb_in1k --img_path data/IN10/valid/n03180011/n03180011_475.JPEG --ckpt_name log/2023-09-01-14-19-44/mobilenetv3.pt")