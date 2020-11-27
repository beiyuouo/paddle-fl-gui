import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import yaml
from utils.testFrame import *

from serverControlFrame import *


class ServerFrame(QWidget):
    def __init__(self, id=0):
        super(ServerFrame, self).__init__()
        self.id = id
        self.config_file = 'config/config_server.yaml'
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

        '''
        self.clientTab = QTableWidget(self)
        self.clientTab.resize(300, 300)
        self.clientTab.move(450, 50)
        self.clientTab.setColumnCount(2)
        self.clientTab.setColumnWidth(0, 145)
        self.clientTab.setColumnWidth(1, 145)

        self.paramTab.setHorizontalHeaderLabels(['client_id', 'ip:port'])
        '''

        self.model_path_label = QLabel('model save path:', self)
        self.model_path_text = QLineEdit(self.config['path']['models_save_path'], self)
        self.model_path_btn = QPushButton('...', self)
        self.model_path_label.move(450, 70)
        self.model_path_text.resize(275, 20)
        self.model_path_text.move(450, 100)
        self.model_path_btn.resize(20, 20)
        self.model_path_btn.move(730, 100)

        self.job_path_label = QLabel('job save path:', self)
        self.job_path_text = QLineEdit(self.config['path']['job_path'], self)
        self.job_path_btn = QPushButton('...', self)
        self.job_path_label.move(450, 170)
        self.job_path_text.resize(275, 20)
        self.job_path_text.move(450, 200)
        self.job_path_btn.resize(20, 20)
        self.job_path_btn.move(730, 200)




        self.startBtn = QPushButton('start', self)
        self.startBtn.move(550, 400)
        self.startBtn.clicked.connect(self.open_scf)

        # self.stopBtn = QPushButton('stop', self)
        # self.stopBtn.move(600, 400)

        self.setWindowTitle('Server {}'.format(self.id))
        self.show()

    def open_scf(self):
        self.scf = ServerControlFrame(self.config, self.id)
        self.scf.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ServerFrame()
    sys.exit(app.exec_())
