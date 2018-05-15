#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tracker_subscriber import TrackerSubscriber
import sys,signal
def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs
    subs.stop()
    sys.exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    subs = TrackerSubscriber("tcp://127.0.0.1:812354")
    subs.start()
    subs.join()