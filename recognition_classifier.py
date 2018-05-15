# -*- coding: utf8 -*-
# ! /usr/bin/python
import sys,signal

from multiprocessing import Queue,Lock


from multiprocessing import Process

import time
import os

sys.path.append('./src/')
from src import classifier
from easydict import EasyDict as edict

class FaceClassifer(Process):
    def __init__(self,q):
        Process.__init__(self)
        #threading.Thread.__init__(self)
        self.thread_state = False
        self.q = q


    def stop(self):
        self.thread_state = False
        print("FaceRecognition %d Process stop!"%(os.getpid()))

        #signal.alarm(1) #break select method



    def run(self):
        self.q.put(os.getpid())
        self.thread_state = True
        while self.thread_state:
            print('process:{} run'.format(os.getpid()))
            time.sleep(5)


def sigint_handler(signum,frame):
    print("main-thread exit")
    global face_proc

    for i in range(len(face_proc)):
        face_proc[i].stop()

    sys.exit()

def get_classes(path):
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    return classes
def KNN_train(arg):
    classifier.main(arg)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    thed = 0.6
    procnum = 2
    load_mode_finish_q = Queue()


    yololock = Lock()
    facelock = Lock()
    face_proc = []
    for i in range(procnum):
        face_proc.append(FaceClassifer(load_mode_finish_q))
    for i in range(procnum):
        face_proc[i].start()
        print('face_proc[{}]={}'.format(i,face_proc[i].pid))

    while load_mode_finish_q.qsize()<procnum:
        #print(load_mode_finish_q.get())
        time.sleep(1)



    for i in range(len(face_proc)):
        load_mode_finish_q.get()

    config = edict()
    config.model = 'models/20180504-215624'
    config.mode = 'TRAIN'
    config.data_dir = '/opt/yanhong.jia/classes'
    config.classifier = 'KNN'
    config.classifier_filename = 'models/KNN.pkl'
    config.use_split_dataset = 0
    config.test_data_dir = ''
    config.batch_size = 100
    config.image_size = 160
    config.seed = 666
    config.min_nrof_images_per_class = 5
    config.nrof_train_images_per_class = 5
    knn_classes = get_classes('/opt/yanhong.jia/classes')
    while True:
        time.sleep(1)
        classes = get_classes('/opt/yanhong.jia/classes')

        if(len(classes) != len(knn_classes)):
            #start classifier train
            print('start train KNN.....')
            knn_train = Process(target=KNN_train, args=(config,))
            knn_train.start()
            knn_train.join()
            print('finish train KNN.....')
            for i in range(len(face_proc)):
                face_proc[i].stop()
                print('restart face_proc[i].pid.....')
                face_proc[i].start()
                print('face_proc[{}]={} restart finish'.format(i, face_proc[i].pid))
                print(''.format(load_mode_finish_q.get()))


        for i in range(len(classes)):
            if(classes[i] != knn_classes[i]):
                # start classifier train
                print('start strain KNN.....')
                knn_train = Process(target=KNN_train, args=(config,))
                knn_train.start()
                knn_train.join()
                print('finish train KNN.....')
                for i in range(len(face_proc)):
                    print('restart face_proc[i].pid.....')
                    face_proc[i].stop()
                    face_proc[i].start()
                    print('face_proc[{}]={} restart finish'.format(i, face_proc[i].pid))
                    print(''.format(load_mode_finish_q.get()))
                break


    for i in range(len(face_proc)):
        face_proc[i].join()











