import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import cv2
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    #if x.shape[3] == 1:
        #x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
    print('------------------crop_sub_imgs_fn------------')  
    #x = crop(x, wrg=128, hrg=128, is_random=is_random)
    x = imresize(x, size=[128, 128], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    print('++++++++++++++++++++downsample_fndownsample_fn++++++++++++++++++++')  
    x = imresize(x, size=[32, 32], interp='bicubic', mode=None)
    x = imresize(x, size=[128, 128], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
def downsample_64_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[64, 64], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
    
    
def downsampleto64_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    #x = imresize(x, size=[128, 128], interp='bicubic', mode=None)
    x = cv2.resize(x,(64,64))
    x = x / (255. / 2.)
    x = x - 1.
    return x