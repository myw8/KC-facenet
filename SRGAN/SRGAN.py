# -*- coding: utf8 -*-
# ! /usr/bin/python


import os, time
from datetime import datetime
import cv2

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
from scipy.misc.pilutil import *

try:
    from PIL import Image, ImageFilter
except ImportError:
    import Image
    import ImageFilter



class LHSRGAN(object):
    def __init__(self):
         ## create folders to save result images

        checkpoint_dir = os.path.dirname(__file__)+"/checkpoint"
        #
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
        self.t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
        
        self.net_g = SRGAN_g(self.t_image, is_train=False, reuse=False)

        ###========================== RESTORE G =============================###
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)
        tl.files.load_and_assign_npz(sess=self.sess, name=checkpoint_dir + '/g_srgan.npz', network=self.net_g)



    def evaluate(self, input_img):
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = self.sess.run(self.net_g.outputs, {self.t_image: [input_img]})
        tm  = time.time() - start_time
        print("took: %4.4fs" % (tm))
        #print("LR size: %s /  generated HR size: %s" % (
        #    input_img.shape, out[0].shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        #print("[*] save images")
        return out[0],tm


    def LR_to_HR(self, valid_lr_img):
        img = (valid_lr_img / 127.5) - 1
        out_img,tm = self.evaluate(img)
        im = toimage(out_img, channel_axis=2)
        im = np.array(im)
        return im, tm



