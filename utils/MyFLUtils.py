import zmq
import time
import random

from paddle_fl.paddle_fl.core import FLScheduler


def recv_and_parse_kv(socket):
    message = socket.recv()
    group = message.decode().split("\t")
    if group[0] == "alive":
        return group[0], "0"
    else:
        return group[0], group[1]


WORKER_EP = "WORKER_EP"
SERVER_EP = "SERVER_EP"


class MFLScheduler(FLScheduler):
    def __init__(self, worker_num, server_num, port=9091, socket=None):
        super().__init__(worker_num, server_num, port=9091, socket=None)

    def start_fl_training_with_round(self, round=0):
        for i in range(round):
            random.shuffle(self.fl_workers)
            worker_dict = {}
            for worker in self.fl_workers[:self.sample_worker_num]:
                worker_dict[worker] = 0

            ready_workers = []
            all_ready_to_train = False
            while not all_ready_to_train:
                key, value = recv_and_parse_kv(self.socket)
                if key == "JOIN":
                    if value in worker_dict:
                        if worker_dict[value] == 0:
                            ready_workers.append(value)
                            worker_dict[value] = 1
                            self.socket.send_string("ACCEPT\t0")
                            continue
                    else:
                        if value not in ready_workers:
                            ready_workers.append(value)
                self.socket.send_string("REJECT\t0")
                if len(ready_workers) == len(self.fl_workers):
                    all_ready_to_train = True

            all_finish_training = False
            finish_training_dict = {}
            while not all_finish_training:
                key, value = recv_and_parse_kv(self.socket)
                if key == "FINISH":
                    finish_training_dict[value] = 1
                    self.socket.send_string("WAIT\t0")
                else:
                    self.socket.send_string("REJECT\t0")
                if len(finish_training_dict) == len(worker_dict):
                    all_finish_training = True
                # print(len(finish_training_dict), finish_training_dict)
            print('round {} triained'.format(i))
            time.sleep(5)
