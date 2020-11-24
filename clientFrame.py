import sys

from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import yaml
from utils.testFrame import *


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

        self.setWindowTitle('Client {}'.format(self.id))
        self.show()

    def open_test_frame(self):
        self.testframe = TestFrame()
        self.testframe.show()

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

        trainer = FLTrainerFactory().create_fl_trainer(job)
        use_cuda = False
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        trainer._current_ep = '{}:{}'.format(self.config['client']['ip'], int(self.config['client']['port']) + self.id)
        print('prepared ok')
        trainer.start(place=place)
        trainer._logger.setLevel(logging.DEBUG)
        print('connected ok')
        self.processLabel.setText('connected')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClientFrame(int(sys.argv[1]))
    sys.exit(app.exec_())