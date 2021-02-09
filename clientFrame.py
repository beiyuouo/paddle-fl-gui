import sys
from time import sleep
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import yaml

from testFrame import TestFrame
from utils.MyFLUtils import MFLTrainerFactory
from utils.reader import *

import threading

import matplotlib.pyplot as plt
from style import FramelessWindow, CircleProgressBar, language
import qtawesome as qta


class ClientFrame(QWidget):
    def __init__(self, id, lang='cn'):
        super(ClientFrame, self).__init__()
        self.lang = lang
        self.id = id
        self.config_file = 'config/config_client.yaml'
        self.yamlstream = open(self.config_file)
        self.config = yaml.load(self.yamlstream)
        print(self.config)
        # print(self.config['parameter']['lr'])
        self.loadQSS()
        self.initGUI()
        self.translateAll()


    def clickedChinese(self):
        self.lang = "cn"
        self.translateAll()

    def clickedEnglish(self):
        self.lang = "en"
        self.translateAll()

    def translateAll(self):
        self.paramTab.setHorizontalHeaderLabels([language[self.lang]['param'], language[self.lang]['value']])
        self.processLabel.setText(language[self.lang]['noconnect'])
        self.connectBtn.setText(language[self.lang]['connect'])
        self.testBtn.setText(language[self.lang]['test'])

    def loadQSS(self):
        """ 加载QSS """
        file = 'qss/style/main.qss'
        with open(file, 'rt', encoding='utf8') as f:
            styleSheet = f.read()
        self.setStyleSheet(styleSheet)
        f.close()
        
    def initGUI(self):
        self.resize(800, 500)

        self.paramTab = QTableWidget(self)
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QtCore.Qt.gray) # 阴影颜色
        self.paramTab.setGraphicsEffect(effect_shadow)

        self.paramTab.resize(350, 300)
        self.paramTab.move(50, 50)
        self.paramTab.setColumnCount(2)
        self.paramTab.setColumnWidth(0, 150)
        self.paramTab.setColumnWidth(1, 150)
        self.paramTab.setHorizontalHeaderLabels(['param', 'value'])
        tabWidgetFont = self.paramTab.horizontalHeader().font()
        tabWidgetFont.setBold(True);
        self.paramTab.horizontalHeader().setFont(tabWidgetFont)
        self.paramTab.horizontalHeader().setStretchLastSection(True);
        self.paramTab.horizontalHeader().resizeSection(0, 170) #设置表头第一列的宽度为150
        self.paramTab.horizontalHeader().resizeSection(1, 170) #设置表头第一列的宽度为150
        self.paramTab.verticalHeader().setVisible(False)
        paramList = []
        for it in self.config['parameter']:
            # print(it)
            paramList.append([it, str(self.config['parameter'][it])])
        paramList.append(['ip:port', '{}:{}'.format(self.config['server']['ip'], self.config['server']['port'])])
        paramList.append(
            ['scheduler_ip:port', '{}:{}'.format(self.config['scheduler']['ip'], self.config['scheduler']['port'])])

        self.paramTab.setRowCount(len(paramList))
        for i, (it0, it1) in enumerate(paramList):
            backcolor = QColor(255,255,255)
            if i % 2 == 1:
                backcolor = QColor(225,225,255)
            item = QTableWidgetItem(it0)
            item.setBackground(backcolor)
            self.paramTab.setItem(i, 0, item)
            item = QTableWidgetItem(it1)
            item.setBackground(backcolor)
            self.paramTab.setItem(i, 1, item)
            # print(it0, it1)

        self.connectBtnIcon1 = qta.icon('fa.plug', scale_factor = 1, color='white')
        self.connectBtn = QPushButton(self.connectBtnIcon1, language[self.lang]['connect'], self, objectName='btnSuccess2')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.connectBtn.setGraphicsEffect(effect_shadow)
        self.connectBtn.resize(150, 40)
        self.connectBtn.move(150, 400)
        # self.connectBtn.resize()
        self.connectBtn.clicked.connect(self.connect_server)

        self.lossLabel = QLabel('loss', self)
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QtCore.Qt.gray) # 阴影颜色
        self.lossLabel.setGraphicsEffect(effect_shadow)
        self.lossLabel.resize(300, 300)
        self.lossLabel.move(450, 50)

        self.processLabel = QLabel(language[self.lang]['noconnect'], self)
        self.processLabel.resize(100, 20)
        # self.processLabel.setAlignment(AlignRight)
        self.processLabel.move(600, 460)

        self.testBtnIcon1 = qta.icon('fa.pencil', scale_factor = 1, color='white')
        self.testBtn = QPushButton(self.testBtnIcon1, language[self.lang]['test'], self, objectName='btnSuccess2')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.testBtn.setGraphicsEffect(effect_shadow)
        self.testBtn.resize(150, 40)
        #self.testBtn = QPushButton('test', self, objectName='btnInfo')
        self.testBtn.move(550, 400)
        self.testBtn.clicked.connect(self.open_test_frame)
        self.update_loss_label()

        self.setWindowTitle('Client {}'.format(self.id))
        #self.show()

    def open_test_frame(self):
        self.testframe = FramelessWindow()
        frame = TestFrame(self.config)
        self.testframe.setWindowTitle('TestFrame')
        self.testframe.setWindowIcon(QIcon('icon.png'))
        self.testframe.setFixedSize(QSize(600,480))  #因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
        self.testframe.setWidget(frame)  # 把自己的窗口添加进来
        self.testframe.titleBar.clickedChinese.connect(frame.clickedChinese)
        self.testframe.titleBar.clickedEnglish.connect(frame.clickedEnglish)
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
            reader = mreader

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
    # ex = ClientFrame(int(sys.argv[1]))
    ex = ClientFrame(0)
    mainWnd = FramelessWindow()
    mainWnd.setWindowTitle('Client {}'.format(0))   #sys.argv[1]
    mainWnd.setWindowIcon(QIcon('icon.png'))
    mainWnd.setFixedSize(QSize(800,550))  #因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
    mainWnd.setWidget(ex)  # 把自己的窗口添加进来
    mainWnd.titleBar.clickedChinese.connect(ex.clickedChinese)
    mainWnd.titleBar.clickedEnglish.connect(ex.clickedEnglish)
    mainWnd.show()
    sys.exit(app.exec_())
