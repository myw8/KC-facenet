#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def Coarse_SR(c_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("Coarse_SR", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(c_image, name='in')  # input 128*128 LR
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c0')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b0')
        temp = n

        # B residual blocks
        for i in range(3):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c0/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b0/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c01/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b01/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add,  name ='b_residual_add/%s' % i)
            n = nn
        # B residual blacks end

        net_c = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c0/m1')
        return net_c

def Encode(e_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("Encode", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(e_image, name='in')  # input 128*128
        n = Conv2d(n, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s2/c')  # 64*64
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s2/b')
        # temp = n

        # B residual blocks
        # input and output is 64*64
        for i in range(12):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add,  name ='b_residual_add/%s' % i)
            n = nn
        # B residual blacks end

        net_encoder = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m1')

        return net_encoder


def Decode(de_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("Decode", reuse=reuse) as vs:
        # decoder
        # priorEstimation network plus encoder,and then put them to decoder

        #n = tf.concat([n.outputs, landmarks.outputs, parsing_maps.outputs], 3)
        n = InputLayer(de_image, name='in_to')  # input 128*128        
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s2/cc')  # 64*64
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s2/bb1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        # de_weight=tf.get_variable('de_weight',shape=[3, 3, 64, 64],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.02))
        # shape[batch,out_height,out_width,out_channels]
        # n = tf.nn.conv2d_transpose(n, filter=de_weight, output_shape=[16,128,128,64],strides=[1, 2, 2, 1], padding='SAME')   #16 is batch_size,[batch, height, width, in_channels]
        # n = tf.nn.conv2d_transpose(n.outputs, de_weight, output_shape=[16,128,128,64],strides=[1, 2, 2, 1], padding='SAME')   #16 is batch_size,[batch, height, width, in_channels]
        # B residual blocks
        # input and output is 64*64
        for i in range(3):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c10/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/bb11/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/cc22/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/bb2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add,  name ='b_residual_add_de/%s' % i)
            n = nn
        # B residual blacks end
        net_decode = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m22')
        
        return  net_decode


def SRGAN_d(d_image, is_train=False, reuse=False):
    '''
    next code (SRGAN) use SRGAN_d2
    '''
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(d_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                            name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME',
                            name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv


def priorEstimation(p_image, is_train=False, reuse=False):
    print('$$ priorEstimation start $$')
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("priorEstimation", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(p_image, name='input_LR')
        n = Conv2d(n, 128, (7, 7), (2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, name='Conv0')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='BN0')
        temp = n

        #  Three residual blocks
        for i in range(3):
            print('$$ residual blocks cycle $$')
            nn = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='ResBlock/conv1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='ResBlock/bn1/%s' % i)
            nn = Conv2d(nn, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='ResBlock/conv2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='ResBlock/bn2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='ResBlock/add/%s' % i)
            n = nn
            print('n========', n)
        # First Call functions HG_block
        print('First Call functions HG_block')
        # layer_tmp = HG_block(n, is_train=is_train, reuse=reuse)

        print('________________________step_1: layer_in')

        # '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HG_block  LAUNCHING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        layer_size = 128 / 2 ** 0
        layer_conv1 = Conv2d(n, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                             b_init=b_init,
                             name='HG/Upconv1')
        print('################################# UpConvBlock1 DONE #################################')
        layer_bn1_1 = BatchNormLayer(layer_conv1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes1/bn1')
        layer_conv1_1 = Conv2d(layer_bn1_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/dwRes1/conv1')
        layer_bn1_2 = BatchNormLayer(layer_conv1_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes1/bn2')
        layer_conv1_2 = Conv2d(layer_bn1_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes1/conv2')
        layer_bn1_3 = BatchNormLayer(layer_conv1_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes1/bn3')
        layer_conv1_3 = Conv2d(layer_bn1_3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes1/conv3')
        layer_res_conv_up1 = Conv2d(n, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init,
                                    name='HG/dwRes1/Upconv')

        layer_out1 = ElementwiseLayer([layer_res_conv_up1, layer_conv1_3], tf.add, name='HG/dwRes1/add')

        layer_dwSamp1 = MaxPool2d(layer_out1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='HG/downSample1')
        print('################################# ResBlock1 DONE #################################')

        layer_size = 128 / 2 ** 1
        layer_conv2 = Conv2d(layer_dwSamp1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                             b_init=b_init,
                             name='HG/UpConv2')
        print('################################# UpConvBlock2 DONE #################################')
        layer_bn2_1 = BatchNormLayer(layer_dwSamp1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes2/bn1')
        layer_conv2_1 = Conv2d(layer_bn2_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/dwRes2/conv1')
        layer_bn2_2 = BatchNormLayer(layer_conv2_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes2/bn2')
        layer_conv2_2 = Conv2d(layer_bn2_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes2/conv2')
        layer_bn2_3 = BatchNormLayer(layer_conv2_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes2/bn3')
        layer_conv2_3 = Conv2d(layer_bn2_3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes2/conv3')
        layer_res_conv_up2 = Conv2d(layer_dwSamp1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init,
                                    name='HG/dwRes2/Upconv')

        layer_out2 = ElementwiseLayer([layer_res_conv_up2, layer_conv2_3], tf.add, name='HG/dwRes2/add')

        layer_dwSamp2 = MaxPool2d(layer_out2, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='HG/downSample2')
        print('################################# ResBlock2 DONE #################################')

        layer_size = 128 / 2 ** 2
        layer_conv3 = Conv2d(layer_dwSamp2, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                             b_init=b_init,
                             name='HG/UpConv3')
        print('################################# UpConvBlock3 DONE #################################')
        layer_bn3_1 = BatchNormLayer(layer_dwSamp2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes3/bn1')
        layer_conv3_1 = Conv2d(layer_bn3_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/dwRes3/conv1')
        layer_bn3_2 = BatchNormLayer(layer_conv3_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes3/bn2')
        layer_conv3_2 = Conv2d(layer_bn3_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes3/conv2')
        layer_bn3_3 = BatchNormLayer(layer_conv3_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes3/bn3')
        layer_conv3_3 = Conv2d(layer_bn3_3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes3/conv3')
        layer_res_conv_up3 = Conv2d(layer_dwSamp2, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init,
                                    name='HG/dwRes3/Upconv')

        layer_out3 = ElementwiseLayer([layer_res_conv_up3, layer_conv3_3], tf.add, name='HG/dwRes3/add')

        layer_dwSamp3 = MaxPool2d(layer_out3, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='HG/downSample3')
        print('################################# ResBlock3 DONE #################################')

        layer_size = 128 / 2 ** 3
        layer_conv4 = Conv2d(layer_dwSamp3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                             b_init=b_init,
                             name='HG/UpConv4')
        print('################################# UpConvBlock4 DONE #################################')
        layer_bn4_1 = BatchNormLayer(layer_dwSamp3, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes4/bn1')
        layer_conv4_1 = Conv2d(layer_bn4_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/dwRes4/conv1')
        layer_bn4_2 = BatchNormLayer(layer_conv4_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes4/bn2')
        layer_conv4_2 = Conv2d(layer_bn4_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes4/conv2')
        layer_bn4_3 = BatchNormLayer(layer_conv4_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/dwRes4/bn3')
        layer_conv4_3 = Conv2d(layer_bn4_3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name='HG/dwRes4/conv3')
        layer_res_conv_up4 = Conv2d(layer_dwSamp3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init,
                                    name='HG/dwRes4/Upconv')

        layer_out4 = ElementwiseLayer([layer_res_conv_up4, layer_conv4_3], tf.add, name='HG/dwRes4/add')

        layer_dwSamp4 = MaxPool2d(layer_out4, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='HG/downSample4')
        print('################################# ResBlock5 DONE #################################')

        layer_size = 128 / 2 ** 4
        layer_bn5_1 = BatchNormLayer(layer_dwSamp4, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes5/bn1')
        layer_conv5_1 = Conv2d(layer_bn5_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes5/conv1')
        layer_bn5_2 = BatchNormLayer(layer_conv5_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes5/bn2')
        layer_conv5_2 = Conv2d(layer_bn5_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes5/conv2')
        layer_bn5_3 = BatchNormLayer(layer_conv5_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes5/bn3')
        layer_conv5_3 = Conv2d(layer_bn5_3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes5/conv3')
        layer_res_conv_up5 = Conv2d(layer_dwSamp4, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='HG/MidRes5/Upconv')
        layer_out5 = ElementwiseLayer([layer_res_conv_up5, layer_conv5_3], tf.add, name='HG/MidRes5/add')
        print('################################# ResBlock6 DONE #################################')

        layer_size = 128 / 2 ** 4
        layer_bn6_1 = BatchNormLayer(layer_out5, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes6/bn1')
        layer_conv6_1 = Conv2d(layer_bn6_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes6/conv1')
        layer_bn6_2 = BatchNormLayer(layer_conv6_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes6/bn2')
        layer_conv6_2 = Conv2d(layer_bn6_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes6/conv2')
        layer_bn6_3 = BatchNormLayer(layer_conv6_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes6/bn3')
        layer_conv6_3 = Conv2d(layer_bn6_3, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes6/conv3')
        layer_res_conv_up6 = Conv2d(layer_out5, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='HG/MidRes6/Upconv')
        layer_out6 = ElementwiseLayer([layer_res_conv_up6, layer_conv6_3], tf.add, name='HG/MidRes6/add')
        print('################################# ResBlock7 DONE #################################')

        layer_size = 128 / 2 ** 4
        layer_bn7_1 = BatchNormLayer(layer_out6, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes7/bn1')
        layer_conv7_1 = Conv2d(layer_bn7_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes7/conv1')
        layer_bn7_2 = BatchNormLayer(layer_conv7_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes7/bn2')
        layer_conv7_2 = Conv2d(layer_bn7_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes7/conv2')
        layer_bn7_3 = BatchNormLayer(layer_conv7_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                     name='HG/MidRes7/bn3')
        layer_conv7_3 = Conv2d(layer_bn7_3, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='HG/MidRes7/conv3')
        layer_res_conv_up7 = Conv2d(layer_out6, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='HG/MidRes7/Upconv')
        layer_out7 = ElementwiseLayer([layer_res_conv_up7, layer_conv7_3], tf.add, name='HG/MidRes7/add')

        layer_upSample0 = SubpixelConv2d(layer_out7, scale=2, n_out_channel=None, act=tf.nn.relu,
                                         name='HG/upSample0')
        print('################################# ResBlock8 DONE #################################')

        layer_upSample1 = SubpixelConv2d(layer_upSample0, scale=2, n_out_channel=None, act=tf.nn.relu,
                                         name='HG/upSample1')
        layer_add1 = ElementwiseLayer([layer_conv4, layer_upSample0], tf.add, name='HG/add1')
        print('################################# AddBlock1 DONE #################################')
        layer_size = 128 / 2 ** 3
        layer_bn11_1 = BatchNormLayer(layer_add1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes1/bn1')
        layer_conv11_1 = Conv2d(layer_bn11_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes1/conv1')
        layer_bn11_2 = BatchNormLayer(layer_conv11_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes1/bn2')
        layer_conv11_2 = Conv2d(layer_bn11_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes1/conv2')
        layer_bn11_3 = BatchNormLayer(layer_conv11_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes1/bn3')
        layer_conv11_3 = Conv2d(layer_bn11_3, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes1/conv3')
        layer_res_conv_up11 = Conv2d(layer_add1, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME',
                                     W_init=w_init, b_init=b_init, name='HG/upRes1/upConv')
        layer_out11 = ElementwiseLayer([layer_res_conv_up11, layer_conv11_3], tf.add, name='HG/upRes1/add')
        print('################################# ResBlock11 DONE #################################')

        layer_upSample2 = SubpixelConv2d(layer_out11, scale=2, n_out_channel=None, act=tf.nn.relu,
                                         name='HG/upSample2')
        layer_add2 = ElementwiseLayer([layer_conv3, layer_upSample2], tf.add, name='HG/add2')
        print('################################# AddBlock1 DONE #################################')
        layer_size = 128 / 2 ** 2
        layer_bn12_1 = BatchNormLayer(layer_add2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes2/bn1')
        layer_conv12_1 = Conv2d(layer_bn12_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes2/conv1')
        layer_bn12_2 = BatchNormLayer(layer_conv12_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes2/bn2')
        layer_conv12_2 = Conv2d(layer_bn12_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes2/conv2')
        layer_bn12_3 = BatchNormLayer(layer_conv12_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes2/bn3')
        layer_conv12_3 = Conv2d(layer_bn12_3, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes2/conv3')
        layer_res_conv_up12 = Conv2d(layer_add2, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME',
                                     W_init=w_init, b_init=b_init, name='HG/upRes2/upConv')
        layer_out12 = ElementwiseLayer([layer_res_conv_up12, layer_conv12_3], tf.add, name='HG/upRes2/add')
        print('################################# ResBlock12 DONE #################################')

        layer_upSample3 = SubpixelConv2d(layer_out12, scale=2, n_out_channel=None, act=tf.nn.relu,
                                         name='HG/upSample3')
        layer_add3 = ElementwiseLayer([layer_conv2, layer_upSample3], tf.add, name='HG/add3')
        print('################################# AddBlock1 DONE #################################')
        layer_size = 128 / 2 ** 1
        layer_bn13_1 = BatchNormLayer(layer_add3, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes3/bn1')
        layer_conv13_1 = Conv2d(layer_bn13_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes3/conv1')
        layer_bn13_2 = BatchNormLayer(layer_conv13_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes3/bn2')
        layer_conv13_2 = Conv2d(layer_bn13_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes3/conv2')
        layer_bn13_3 = BatchNormLayer(layer_conv13_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes3/bn3')
        layer_conv13_3 = Conv2d(layer_bn13_3, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes3/conv3')
        layer_res_conv_up13 = Conv2d(layer_add3, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME',
                                     W_init=w_init, b_init=b_init, name='HG/upRes3/upConv')
        layer_out13 = ElementwiseLayer([layer_res_conv_up13, layer_conv13_3], tf.add, name='HG/upRes3/add')
        print('################################# ResBlock13 DONE #################################')

        layer_upSample4 = SubpixelConv2d(layer_out13, scale=2, n_out_channel=None, act=tf.nn.relu,
                                         name='HG/upSample4')
        layer_add4 = ElementwiseLayer([layer_conv1, layer_upSample4], tf.add, name='HG/add4')
        print('################################# AddBlock4 DONE #################################')
        layer_size = 128 / 2 ** 0
        layer_bn14_1 = BatchNormLayer(layer_add4, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes4/bn1')
        layer_conv14_1 = Conv2d(layer_bn14_1, layer_size, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes4/conv1')
        layer_bn14_2 = BatchNormLayer(layer_conv14_1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes4/bn2')
        layer_conv14_2 = Conv2d(layer_bn14_2, layer_size, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes4/conv2')
        layer_bn14_3 = BatchNormLayer(layer_conv14_2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init,
                                      name='HG/upRes4/bn3')
        layer_conv14_3 = Conv2d(layer_bn14_3, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                                b_init=b_init, name='HG/upRes4/conv3')
        layer_res_conv_up14 = Conv2d(layer_add4, layer_size * 8, (1, 1), (1, 1), act=None, padding='SAME',
                                     W_init=w_init, b_init=b_init, name='HG/upRes4/upConv')
        layer_out14 = ElementwiseLayer([layer_res_conv_up14, layer_conv14_3], tf.add, name='HG/upRes4/add')
        print('################################# ResBlock14 DONE #################################')

        print(
            '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HG_block  DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # return layer_out14

        # print('layer_tmp=========', layer_tmp)
        # n = conv2(layer_out14, 128, 2, is_train=is_train, reuse=reuse, name='conv')

        n = Conv2d(layer_out14, 128, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                   name='fConv1')
        n = Conv2d(n, 128, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='fConv2')

        # second Call functions HG_block
        print('second Call functions HG_block')
        # layer_tmp = HG_block(n, is_train=is_train, reuse=True)

        # n = conv2(n, 128, 1, is_train=is_train, reuse=reuse, name='conv2')
        # landmarks = conv2(n, 64, 1, is_train=is_train, reuse=reuse, name='conv3')
        # parsing_maps = conv2(n, 64, 1, is_train=is_train, reuse=reuse, name='conv4')

        n = Conv2d(n, 128, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='fConv3')
        landmarks = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                           name='landmarkConv')
        parsing_maps = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                              name='parsingConv')



       
        ## add a conv ,last output [16,64,64,1] for P_loss1
        landmarks_loss = Conv2d(landmarks, 1, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,name='landmarkLossConv')
        
        ## add conv ,last output 11 parising maps[16,64,64,11] for P_loss2
        parsing_maps_loss = Conv2d(parsing_maps, 11, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   name='parisingLossConv')

        ## just for return all the p net
        ## combine   landmarks net and  parsing_maps net
        p_net = Conv2d(parsing_maps_loss, 1, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   name='pNetConv1')
        p_net = ElementwiseLayer([p_net, landmarks_loss], tf.add,  name ='pNetConv2')


        '''
        ## add pool and FC*2 ,last output points for P_loss1
        network = MaxPool2d(landmarks, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='landPool')
        network = FlattenLayer(network, name='flattenLayer0')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='Dense1')
        landmarks_loss = DenseLayer(network, n_units=388, act=tf.nn.relu,
                                    name='Dense2')  # points num is 68,so n_unitis=68*2

        ## add convsd ,last output 11 parising maps for P_loss2
        parsing_maps_loss = Conv2d(parsing_maps, 11, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   name='parisingLossConv')

        ## just for return all the p net
        ## combine   landmarks net and  parsing_maps net
        p_net = MaxPool2d(parsing_maps_loss, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='parisingPool')
        p_net = FlattenLayer(p_net, name='flattenLayer1')
        p_net = DenseLayer(p_net, n_units=388, act=tf.nn.relu, name='Dense3')
        p_net = ElementwiseLayer([p_net, landmarks_loss], tf.add, name='addLayer/jks')
        p_net = DenseLayer(p_net, n_units=1, act=tf.nn.relu, name='Dense4')
        '''

        return landmarks, parsing_maps, landmarks_loss, parsing_maps_loss, p_net

