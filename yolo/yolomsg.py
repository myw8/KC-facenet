#!/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
import json
import time
import tensorlayer as tl
import base64
import random
class YoloMsg(object):
    def __init__(self):
        #self.dict_yolo = {}
        self.list_yolo = []

    def hasYoloMsg(self):
        return len(self.list_yolo)

    def PopMsg(self):
        if(len(self.list_yolo)):
            return self.list_yolo.pop()
    def InsertYoloMsg(self,facedir):
        img_list = sorted(tl.files.load_file_list(path=facedir, regx='.*.png', printable=False))
        imgs = tl.vis.read_images(img_list, path=facedir, n_threads=4)
        dict_yolo = {}
        id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '_' + str(np.random.randint(0, 9999)).zfill(4)
        #print(id)
        im = imgs[0]
        h = im.shape[0]
        w = im.shape[1]
        x1 = random.randint(0, 1080)
        y1 = random.randint(0, 1080)
        x2 = random.randint(0, 1080)
        y2 = random.randint(0, 1080)
        imcode = base64.b64encode(im)
        faceinfo = [[x1, y1, w, h], imcode, [x2, y2, w, h], imcode]
        dict_yolo[id] = [faceinfo]
        for i in range(len(imgs)-1):
            im = imgs[i+1]
            imcode = base64.b64encode(im)
            h = im.shape[0]
            w = im.shape[1]
            x1 = random.randint(0,1080)
            y1 = random.randint(0,1080)
            x2 = random.randint(0,1080)
            y2 = random.randint(0,1080)
            faceinfo = [[x1,y1,w,h],imcode,[x2,y2,w,h],imcode]
            dict_yolo[id].append(faceinfo)
        self.list_yolo.append(dict_yolo)