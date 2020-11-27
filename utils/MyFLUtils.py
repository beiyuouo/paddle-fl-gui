import logging

import zmq
import time
import random

from paddle import fluid
from paddle_fl.paddle_fl.core import FLScheduler, FLWorkerAgent
from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainer, FedAvgTrainer


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

    def ep_to_idx(self, ep=None):
        if ep == None:
            self.ep_dict = {}
            for idx, i in enumerate(self.fl_workers):
                self.ep_dict[i] = idx
        else:
            return self.ep_dict[ep]

    def start_fl_training_with_round(self, round=0, label=None):
        self.ep_to_idx()
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
                    if label is not None:
                        self.change_label(label[self.ep_to_idx(value)], i, round)
                    self.socket.send_string("WAIT\t0")
                else:
                    self.socket.send_string("REJECT\t0")
                if len(finish_training_dict) == len(worker_dict):
                    all_finish_training = True
                # print(len(finish_training_dict), finish_training_dict)
            print('round {} triained'.format(i))
            # self.change_label(label[-1], i, round)
            time.sleep(5)

    def change_label(self, labels, idx, rounds):
        name_label, pro_bar, pro_label, state_label = labels
        prog = (idx+1)/rounds
        pro_bar.setValue(prog*100)
        pro_label.setText('{} %'.format(prog*100))


class MFLWorkerAgent(FLWorkerAgent):
    def __init__(self, scheduler_ep, current_ep):
        super(MFLWorkerAgent, self).__init__(scheduler_ep, current_ep)

    def finish_training(self):
        self.socket.send_string("FINISH\t{}".format(self.current_ep))
        key, value = recv_and_parse_kv(self.socket)
        if key == "WAIT":
            time.sleep(3)
            return True
        return False


class MFLTrainer(FLTrainer):
    def __init__(self):
        self._logger = logging.getLogger("FLTrainer")

    def start(self, place):
        # current_ep = "to be added"
        self.agent = MFLWorkerAgent(self._scheduler_ep, self._current_ep)
        self.agent.connect_scheduler()
        self.exe = fluid.Executor(place)
        self.exe.run(self._startup_program)


class MFedAvgTrainer(FedAvgTrainer):
    def __init__(self):
        super(MFedAvgTrainer, self).__init__()
        pass

    def start(self, place):
        # current_ep = "to be added"
        self.agent = MFLWorkerAgent(self._scheduler_ep, self._current_ep)
        self.agent.connect_scheduler()
        self.exe = fluid.Executor(place)
        self.exe.run(self._startup_program)

    def run_with_epoch(self, reader, feeder, fetch, num_epoch):
        self._logger.debug("begin to run recv program")
        self.exe.run(self._recv_program)
        epoch = 0
        loss_list = []
        for i in range(num_epoch):
            loss_i = 0
            for data in reader():
                loss_i += self.exe.run(self._main_program,
                                       feed=feeder.feed(data),
                                       fetch_list=fetch)
            loss_list.append(loss_i)
            self.cur_step += 1
            epoch += 1
        self._logger.debug("begin to run send program")
        self.exe.run(self._send_program)
        return loss_list
