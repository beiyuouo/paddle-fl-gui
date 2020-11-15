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
        self.paramTab.setHorizontalHeaderLabels(['param', 'value'])
        paramList = []
        for it in self.config['parameter']:
            # print(it)
            paramList.append([it, str(self.config['parameter'][it])])
        paramList.append(['ip:port', '{}:{}'.format(self.config['server']['ip'], self.config['server']['port'])])
        self.paramTab.setRowCount(len(paramList))
        for i, (it0, it1) in enumerate(paramList):
            self.paramTab.setItem(i, 0, QTableWidgetItem(it0))
            self.paramTab.setItem(i, 1, QTableWidgetItem(it1))
            # print(it0, it1)

        self.connectBtn = QPushButton('connect', self)
        # self.connectBtn.resize()
        self.connectBtn.move(150, 400)

        self.lossLabel = QLabel('loss', self)
        self.lossLabel.resize(300, 300)
        self.lossLabel.move(450, 50)

        self.processLabel = QLabel('noconnect', self)
        self.processLabel.resize(100, 20)
        self.processLabel.move(600, 470)

        self.testBtn = QPushButton('test', self)
        self.testBtn.move(550, 400)
        self.testBtn.clicked.connect(self.opentestwin)

        self.setWindowTitle('Client {}'.format(self.id))
        self.show()

    def opentestwin(self):
        self.testframe = TestFrame()
        self.testframe.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClientFrame(int(sys.argv[1]))
    sys.exit(app.exec_())
