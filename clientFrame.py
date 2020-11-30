import sys
from time import sleep
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import yaml

from testFrame import TestFrame
from utils.MyFLUtils import MFLTrainerFactory
from utils.reader import *

import threading

import matplotlib.pyplot as plt


class ClientFrame(QWidget):
    def __init__(self, id):
        super(ClientFrame, self).__init__()
        self.id = id
        self.config_file = 'config/config_client.yaml'
        self.yamlstream = open(self.config_file)
        self.config = yaml.load(self.yamlstream)
        print(self.config)
        # print(self.config['parameter']['lr'])
        self.initGUI()

    def initGUI(self):
        self.resize(800, 500)

        self.paramTab = QTableWidget(self)
        self.paramTab.resize(300, 300)
        self.paramTab.move(50, 50)
        self.paramTab.setColumnCount(2)
        self.paramTab.setColumnWidth(0, 100)
        self.paramTab.setColumnWidth(1, 180)
        self.paramTab.setHorizontalHeaderLabels(['param', 'value'])
        paramList = []
        for it in self.config['parameter']:
            # print(it)
            paramList.append([it, str(self.config['parameter'][it])])
        paramList.append(['ip:port', '{}:{}'.format(self.config['server']['ip'], self.config['server']['port'])])
        paramList.append(
            ['scheduler_ip:port', '{}:{}'.format(self.config['scheduler']['ip'], self.config['scheduler']['port'])])

        self.paramTab.setRowCount(len(paramList))
        for i, (it0, it1) in enumerate(paramList):
            self.paramTab.setItem(i, 0, QTableWidgetItem(it0))
            self.paramTab.setItem(i, 1, QTableWidgetItem(it1))
            # print(it0, it1)

        self.connectBtn = QPushButton('connect', self)
        # self.connectBtn.resize()
        self.connectBtn.move(150, 400)
        self.connectBtn.clicked.connect(self.connect_server)

        self.lossLabel = QLabel('loss', self)
        self.lossLabel.resize(300, 300)
        self.lossLabel.move(450, 50)

        self.processLabel = QLabel('noconnect', self)
        self.processLabel.resize(100, 20)
        # self.processLabel.setAlignment(AlignRight)
        self.processLabel.move(600, 470)

        self.testBtn = QPushButton('test', self)
        self.testBtn.move(550, 400)
        self.testBtn.clicked.connect(self.open_test_frame)
        self.update_loss_label()

        self.setWindowTitle('Client {}'.format(self.id))
        self.show()

    def open_test_frame(self):
        self.testframe = TestFrame(self.config)
        self.testframe.show()

    def update_loss_label(self):
        jpg = QtGui.QPixmap('loss_temp_{}.jpg'.format(self.id)).scaled(self.lossLabel.width(), self.lossLabel.height())
        self.lossLabel.setPixmap(jpg)

    def connect_server(self):
        from paddle import fluid
        from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
        from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
        import numpy as np
        import sys

        import logging
        self.processLabel.setText('connecting')
        trainer_id = self.id
        job_path = self.config['path']['job_path']
        job = FLRunTimeJob()
        job.load_trainer_job(job_path, trainer_id)
        job._scheduler_ep = '{}:{}'.format(self.config['scheduler']['ip'], self.config['scheduler']['port'])
        # print(job._trainer_send_program)

        self.trainer = MFLTrainerFactory().create_fl_trainer(job)
        use_cuda = False
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        self.trainer._current_ep = '{}:{}'.format(self.config['client']['ip'],
                                                  int(self.config['client']['port']) + self.id)
        print('prepared ok')
        self.trainer.start(place=place)
        self.trainer._logger.setLevel(logging.DEBUG)
        print('connected ok')
        self.processLabel.setText('connected')
        self.trainThread = threading.Thread(target=self.train)
        self.processLabel.setText('training')
        self.trainThread.start()
        self.trainThread.join()
        self.processLabel.setText('finished')

    def train(self):
        output_folder = self.config['path']['output_path']
        step_i = 0
        print(id(self.lossLabel))
        # self.lossThread = threading.Thread(target=self.update_loss_label)
        # self.lossThread.start()
        loss_list = np.array([])

        if self.config['parameter']['model'] == 'resnet':
            reader = data_loader(self.config['path']['data/data/'], batch_size=1)
        else:
            reader = mreader(self.id)

        while not self.trainer.stop():
            step_i += 1
            print("batch %d start train" % step_i)
            loss_list = np.concatenate((loss_list,
                                        self.trainer.run_with_epoch(reader, [],
                                                                    int(self.config['parameter']['epochs']),
                                                                    self.id)))
            plt.plot(range(0, len(loss_list)), loss_list)
            plt.legend(['train_loss'], loc='upper left')
            plt.savefig('loss_temp_{}.jpg'.format(self.id))
            self.update_loss_label()
            if self.id == 0:
                print("start saving model")
                self.trainer.save_inference_program(output_folder)
            if step_i > int(self.config['parameter']['round']):
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClientFrame(int(sys.argv[1]))
    sys.exit(app.exec_())
