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
import numpy as np
from PIL import Image
import cv2
import time
import threading
import Queue

class ThreadPool(object):  #创建线程池类

    def __init__(self, max_num=20):  #创建一个最大长度为20的队列
        self.queue = Queue.Queue(max_num)  #创建一个队列
        for i in range(max_num):  #循环把线程对象加入到队列中
            self.queue.put(threading.Thread)  #把线程的类名放进去，执行完这个Queue

    def get_thread(self):  #定义方法从队列里获取线程
        return self.queue.get()  #在队列中获取值

    def add_thread(self):  #线程执行完任务后，在队列里添加线程
        self.queue.put(threading.Thread)




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

    #publish = FacePublisher("tcp://127.0.0.1:12354")
    #publish.setDaemon(True)
    #facemsg = publish.facemsg

    pool = ThreadPool(1)
    #publish.start()
    subs.start()


    def yoloMsgProc(p, msg):
        # global facemsg
        global  face_recognition
        for id, value in msg.items():
            dict_face = {}
            dict_face[id] = []
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
                print(pid, "w:%d h:%d" % (w, h))
                data = person[D_HEAD]
                arr = np.array(data)
                arr = arr.reshape(-1)
                img = arr.reshape(h, w, 3)
                img = np.uint8(img)
                faceimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
                # face rgb
                #########################################
                flag = 0
                print(faceimg.shape[0], faceimg.shape[1])
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
        # facemsg.insertListFace(dict_face)
        p.add_thread()

    while True:
        if (len(subs.msg_list) > 0):
            msg = subs.msg_list.pop()
            #yoloMsgProc(msg)
            thread = pool.get_thread()  # 线程池10个线程，每一次循环拿走一个拿到类名，没有就等待
            t = thread(target=yoloMsgProc, args=(pool, msg))  # 创建线程；  线程执行func函数的这个任务；args是给函数传入参数
            t.setDaemon(True)
            t.start()  # 激活线程
        time.sleep(0.02)
    subs.join()
    publish.join()







