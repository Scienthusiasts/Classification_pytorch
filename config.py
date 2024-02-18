config = dict(
    # train
    mode = 'train',
    timm_model_name = 'mobilenetv3_large_100.ra_in1k',
    img_size = 224,
    ckpt_load_path =  '',   # 'log/2024-02-14-03-04-03_train/best.pt',
    dataset_dir = 'E:/datasets/Classification/food-101/images',
    epoch = 48,
    bs = 64,
    lr = 1e-3,
    log_dir = './log',
    log_interval = 50,
    pretrain = True,
    froze = False,
    optim_type = 'adamw',
    resume = None,  # 'log/2024-02-05-21-28-59_train/epoch_9.pt',
    seed=22,
    # eval
    eval_log_dir = 'log/2024-02-14-03-04-03_train',
    # test
    # french_fries/3171053.jpg 3897130.jpg 3393816.jpg club_sandwich/3143042.jpg
    img_path = 'E:/datasets/Classification/food-101/images/valid/french_fries/3393816.jpg',
    save_res_dir = './result'
)


