import zmq, json, socket, signal
import threading
import time
import sys

class Subscriber(threading.Thread):
    def __init__(self, addr):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SNDBUF, 20000 * 1024)
        self.socket.connect(addr)
        self.socket.setsockopt(zmq.SUBSCRIBE, '')
        self.thread_state = True
        self.poller = zmq.Poller()

        self.poller.register(self.socket, zmq.POLLIN)
        self.msg_list = []

    def stop(self):
        print("zmq recv thread stop!")
        self.thread_state = False
        self.socket.close()
        signal.alarm(1)  # break select method

    def run(self):
        while self.thread_state:
            # socks = dict(self.poller.poll())
            # if socks.get(self.socket) == zmq.POLLIN:
            msg_cli = self.socket.recv_json()
            recv = json.loads(msg_cli)
            print(recv)
            #self.msg_list.append(recv)
            # print(recv)  TypeError(repr(o) + " is not JSON serializable")
            time.sleep(0.2)


def sigint_handler(signum, frame):
    print("main-thread exit")
    global subs
    subs.stop()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    subs = Subscriber("tcp://192.168.20.40:12354")
    subs.start()
    subs.join()