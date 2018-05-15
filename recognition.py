# -*- coding: utf8 -*-
# ! /usr/bin/python
from FSRGAN.FSRGAN import LHFSRGAN
from SRGAN.SRGAN import LHSRGAN
import tensorlayer as tl
from multiprocessing import Process,Queue,Lock
import sys
from src import face

from face_subscriber import FaceSubscriber
from face_publisher import FacePublisher
import sys,signal
import numpy as np
import cv2
import time


def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs
    global publish
    publish.stop()
    subs.stop()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    face_recognition = face.Recognition()
    # srgan = LHSRGAN()
    srgan = LHFSRGAN()
    thed = 0.6

    subs = FaceSubscriber("tcp://127.0.0.1:12341")
    subs.setDaemon(True)

    publish = FacePublisher("tcp://127.0.0.1:12354")
    publish.setDaemon(True)
    publish.start()
    subs.start()
    while True:
        while (len(subs.msg_list) > 0):
            msg = subs.msg_list.pop()
            for id, value in msg.items():
                dict_face = {}
                dict_face[id] = []
                #item_tm = time.time()
                for i in range(len(value)):
                    ##########################################
                    D_ID = 0
                    D_HEAD_AXIS = 1
                    D_W = 2
                    D_H = 3
                    D_HEAD = 2
                    person = msg[id][i]
                    # print(person)
                    pid = str(person[D_ID])
                    w = int(person[D_HEAD_AXIS][D_W])
                    h = int(person[D_HEAD_AXIS][D_H])
                    #print(pid,"@@@@@@@@@@w:%d h:%d"%(w, h))
                    data = person[D_HEAD]
                    arr = np.array(data)
                    arr = arr.reshape(-1)
                    img = arr.reshape(h, w, 3)
                    img = np.uint8(img)
                    faceimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
                    #face rgb
                    #########################################
                    flag = 0
                    #print(faceimg.shape[0], faceimg.shape[1])
                    #start_time = time.time()
                    if (faceimg.shape[0] < 80 and faceimg.shape[1] < 80):
                        faceimg, _ = srgan.LR_to_HR(faceimg)
                        flag = 1
                    if faceimg.ndim < 2:
                        print('Unable to align ')
                    faces = face_recognition.identify(faceimg)
                    if (len(faces) > 0):
                        if (faces[0].confidence < thed):
                            faces[0].name = 'unknown'
                        faceinfo = [pid, faces[0].name, flag]
                    else:
                        faceinfo = [pid, 'unable', flag]
                    dict_face[id].append(faceinfo)
                    #print('111111111111111one face img cost time:%f' % (time.time() - start_time))
            publish.insertListFace(dict_face)
            #print('222222222222one face img cost time:%f' % (time.time() - start_time))

        #time.sleep(0.02)
    subs.join()
    publish.join()







