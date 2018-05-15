#!/usr/bin/env python
# -*- coding: utf-8 -*-
from publisher import Publisher
import sys,signal
import time
import os
import random
def sigint_handler(signum,frame):
    print("main-thread exit")
    global publish
    publish.stop()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    publish = Publisher("tcp://127.0.0.1:12347")
    publish.start()


    path_exp = os.path.expanduser('../samples')
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)

    while True:
        i = random.randint(0,nrof_classes-1)
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        publish.yolomsg.InsertYoloMsg(facedir)
        time.sleep(1)

    publish.join()