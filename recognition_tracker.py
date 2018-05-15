# -*- coding: utf8 -*-
# ! /usr/bin/python
from FSRGAN.FSRGAN import LHFSRGAN
from SRGAN.SRGAN import LHSRGAN
import tensorlayer as tl
from multiprocessing import Process,Queue,Lock
import sys
from src import face
from scipy import misc
from face_subscriber import FaceSubscriber
from face_publisher import FacePublisher
import sys,signal
from facemsg import FaceMsg
import base64
import io
import os
import random
import time
import numpy as np

def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs
    global publish
    publish.stop()
    subs.stop()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    subs = FaceSubscriber("tcp://127.0.0.1:12347")
    subs.start()
    publish = FacePublisher("tcp://127.0.0.1:12354")
    publish.start()
    face_recognition = face.Recognition()
    # srgan = LHSRGAN()
    srgan = LHFSRGAN()
    thed = 0.6
    facemsg = publish.facemsg

    path_exp = os.path.expanduser('./samples')
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)

    while True:
        i = random.randint(0, nrof_classes - 1)
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        img_list = sorted(tl.files.load_file_list(path=facedir, regx='.*.png', printable=False))
        imgs = tl.vis.read_images(img_list, path=facedir, n_threads=4)
        id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '_' + str(np.random.randint(0, 9999)).zfill(4)
        # print(id)
        dict_face = {}

        for i in range(len(imgs)):
            imgface = imgs[i]
            h = imgface.shape[0]
            w = imgface.shape[1]
            x1 = random.randint(0, 1080)
            y1 = random.randint(0, 1080)
            flag = 0
            if (h< 80 and w < 80):
                imgface, _ = srgan.LR_to_HR(imgface)
                flag = 1
            if imgface.ndim < 2:
                print('Unable to align ')
            faces = face_recognition.identify(imgface)
            if(len(faces)>0):
                if(faces[0].confidence<thed):
                    faces[0].name = 'unknown'
                faceinfo = [[x1, y1, w, h], str(faces[0].embedding), faces[0].name, flag]
            else:
                faceinfo = [[x1, y1, w, h], str(np.zeros((128,))), 'unable', flag]
            if i == 0:
                dict_face[id] = [faceinfo]
            else:
                dict_face[id].append(faceinfo)

        facemsg.insertListFace(dict_face)
        time.sleep(0.2)

    subs.join()
    publish.join()







