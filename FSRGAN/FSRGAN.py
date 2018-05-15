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

class LHFSRGAN(object):
    def __init__(self):
        #  loading model
        self.image_size = 128
        self.mode_dir = os.path.dirname(__file__)
        self.sess, self.net_g_test, self.t_image = self.InitModel()


    def InitModel(self):
        t_image = tf.placeholder('float32', [1,self.image_size, self.image_size,3], name='input_image')
        net_c_test = Coarse_SR(t_image, is_train=False, reuse=False)
        net_e_test = Encode(net_c_test.outputs, is_train=False, reuse=False)
        landmarks_test, parsing_maps_test,_,_,net_p_test = priorEstimation(net_c_test.outputs, is_train=False, reuse=False)

        de_image_test = tf.concat([net_e_test.outputs, landmarks_test.outputs, parsing_maps_test.outputs], 3)
        net_g_test = Decode(de_image_test, is_train=False, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)

        tl.files.load_ckpt(sess=sess, mode_name='net_c_srgan.ckpt', save_dir=self.mode_dir+'/checkpoint/checkpoint_c/',var_list=net_c_test.all_params, is_latest=True, printable=True)
        tl.files.load_ckpt(sess=sess, mode_name='net_e_srgan.ckpt', save_dir=self.mode_dir+'/checkpoint/checkpoint_e/',var_list=net_e_test.all_params, is_latest=True, printable=True)
        tl.files.load_ckpt(sess=sess, mode_name='net_p_srgan.ckpt', save_dir=self.mode_dir+'/checkpoint/checkpoint_p/',var_list=net_p_test.all_params, is_latest=True, printable=True)
        tl.files.load_ckpt(sess=sess, mode_name='net_g_srgan.ckpt', save_dir=self.mode_dir+'/checkpoint/checkpoint_g/',var_list=net_g_test.all_params, is_latest=True, printable=True)
        return sess,net_g_test,t_image

    def evaluate(self,img):

        start_time = time.time()
        out = self.sess.run(self.net_g_test.outputs, {self.t_image: [img]})
        tm = time.time() - start_time
        #print("###############took: %4.4fs" % (time.time() - start_time))
        #print('out[0]', out[0].shape)
        #print("[*] save images")
        return out[0],tm


    def LR_to_HR(self,valid_lr_img):
        #lr_img = valid_lr_img[:, :, ::-1]  # if we do not use 'tl.vis.save_image()',Comment this line of code.
        #lr_img = imresize(valid_lr_img, size=[32, 32], interp='bicubic', mode=None)
        lr_img = imresize(valid_lr_img, size=[self.image_size, self.image_size], interp='bicubic', mode=None)

        lr_img = (lr_img / 127.5) - 1
        # test img
        out_img,tm = self.evaluate(lr_img)
        im = toimage(out_img, channel_axis=2)
        #sess.close()
        im = np.array(im)
        return im,tm
        #return out_img


