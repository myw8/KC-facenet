from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## add param 
config.TRAIN.lamda = 1         # use for p net 
config.TRAIN.lamda_landmark = 1          # use for p_landmark net 
config.TRAIN.lamda_parsing = 1         # use for p_parsing  net 
config.TRAIN.lamda_c = 10    # use for D 
config.TRAIN.lamda_p = 5e-5    # use for vgg
config.TRAIN.g_gan = 0.1      # use for aotoencoder



## Adam
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 500
#config.TRAIN.n_epoch_init = 2
config.TRAIN.lr_decay_init = 0.1
config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (FSRGAN)
config.TRAIN.n_epoch = 2000
#config.TRAIN.n_epoch = 2
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = './data/DIV2K_train_HR/'
config.TRAIN.lr_img_path = './data/DIV2K_train_LR_bicubic/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = './data/DIV2K_valid_HR/'
config.VALID.lr_img_path = './data/DIV2K_valid_LR_bicubic/'


## add
config.TRAIN.batch_size = 16
config.TRAIN.bacth_size_num_temp = 145 # is = int(train_img num / batch_size)
#config.TRAIN.bacth_size_num_temp = 1 # is = int(train_img num / batch_size)
config.TRAIN.img_width = 128        # train img width,128*128
config.TRAIN.landmark_num = 388     # landmark points * 2
config.TRAIN.sorted_path = '/home/kaiyu.zhong/opt/09_srgan-tensorflow_better_temp/data/sorted.txt'
config.TRAIN.train_data_path = '/home/kaiyu.zhong/opt/09_srgan-tensorflow_better_temp/data/helen_HR/'
config.TRAIN.landmark_path = "/home/kaiyu.zhong/opt/09_srgan-tensorflow_better_temp/data/landmark_img/"
config.TRAIN.parsing_path = '/home/kaiyu.zhong/opt/09_srgan-tensorflow_better_temp/data/SmithCVPR2013_dataset_original/labels/'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
