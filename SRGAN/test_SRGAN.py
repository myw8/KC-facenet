import os, time
from datetime import datetime
import cv2
import math
import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
from os import listdir
from os.path import join
import argparse
import numpy as np
import pandas as pd
import sys
from MSE_PSNR_SSIM_new import *
from SRGAN import *
results = {'filename': [],  'mse': [], 'psnr': [], 'ssim': []}
def main(args):
    ## create folders to save result images
    tl.files.exists_or_mkdir(args.sr_data_dir)
    tl.files.exists_or_mkdir(args.lr_bicu_dir)
    tl.files.exists_or_mkdir(args.statistics_dir)

    width = args.image_size
    height = args.image_size

    sr_time = []
    #  loading model
    srgan = LHSRGAN()
    ###==============if(args.mode )======== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    if (args.mode=='TRAIN'):
        valid_hr_img_list = sorted(tl.files.load_file_list(path=args.hr_data_dir, regx='.*.png', printable=False))
        valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=args.hr_data_dir, n_threads=32)
        valid_lr_img_list = sorted(tl.files.load_file_list(path=args.lr_data_dir, regx='.*.png', printable=False))
        valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=args.lr_data_dir, n_threads=32)
        i = 0
        for im in valid_lr_imgs:
            im = imresize(im, size=[32, 32], interp='bicubic', mode=None)
            sr_image,sr_ti = srgan.LR_to_HR(im)
            sr_time.append(sr_ti)
            tl.vis.save_image(sr_image, args.sr_data_dir + '/' + valid_lr_img_list[i])
            i = i + 1

        results = more_mse_psnr(args.hr_data_dir,args.sr_data_dir)

        data_frame = pd.DataFrame(
            data={'Filename': results['filename'], 'MSE': results['mse'], 'PSNR': results['psnr'],
                  'SSIM': results['ssim'],'time':sr_time})
        data_frame.to_csv(args.statistics_dir + '/' + 'test_results_sr.csv')
    else:
        path_exp = os.path.expanduser(args.lr_data_dir)
        classes = [path for path in os.listdir(path_exp) \
                   if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            tl.files.exists_or_mkdir(args.lr_bicu_dir + '/' + class_name)
            tl.files.exists_or_mkdir(args.sr_data_dir + '/' + class_name)
            facedir = os.path.join(path_exp, class_name)
            valid_lr_img_list = sorted(tl.files.load_file_list(path=facedir, regx='.*.png', printable=False))
            valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=facedir, n_threads=4)
            k=0
            for im in valid_lr_imgs:
                im = imresize(im, size=[32, 32], interp='bicubic', mode=None)
                sr_image,sr_ti = srgan.LR_to_HR(im)
                tl.vis.save_image(sr_image, args.sr_data_dir + '/' + class_name+ '/'+ valid_lr_img_list[k])
                k=k+1
            i = i + 1
    exit()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' +
        'model should be used for classification', default='TRAIN')
    parser.add_argument('--lr_data_dir', type=str, help='Path to the input LR data directory containing face images.')
    parser.add_argument('--lr_bicu_dir', type=str,default = '.',help='Path to the target data directory ')
    parser.add_argument('--sr_data_dir', type=str, help='Path to the output HR data directory ')
    parser.add_argument('--hr_data_dir', type=str,default = '.', help='Path to the  HR data directory ')
    parser.add_argument('--statistics_dir', type=str, default = '.',help='Path to the output HR data directory ')
    parser.add_argument('--model', type=str, default='models/',
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=128)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
