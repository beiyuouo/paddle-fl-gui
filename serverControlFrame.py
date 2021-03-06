import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
import yaml

from model import *
from testFrame import TestFrame
from utils.MyFLUtils import MFLScheduler

import numpy as np
from style import FramelessWindow, CircleProgressBar, language
import qtawesome as qta

class ServerControlFrame(QWidget):
    def __init__(self, config, id, lang='cn'):
        super(ServerControlFrame, self).__init__()
        self.lang = lang
        self.config = config
        self.id = id
        self.connected_agent = 0
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
        self.clientGroup.setTitle(language[self.lang]['Client ProgressBar'])
        if self.processLabel.text() == language['cn']['noconnect']:
            self.processLabel.setText(language[self.lang]['noconnect'])
        if self.processLabel.text() == language['en']['noconnect']:
            self.processLabel.setText(language[self.lang]['noconnect'])
        if self.processLabel.text() == language['cn']['training']:
            self.processLabel.setText(language[self.lang]['training'])
        if self.processLabel.text() == language['en']['training']:
            self.processLabel.setText(language[self.lang]['training'])
        if self.processLabel.text() == language['cn']['finished']:
            self.processLabel.setText(language[self.lang]['finished'])
        if self.processLabel.text() == language['en']['finished']:
            self.processLabel.setText(language[self.lang]['finished'])
        self.initBtn.setToolTip(language[self.lang]['init env'])
        self.startBtn.setToolTip(language[self.lang]['start'])
        self.stopBtn.setToolTip(language[self.lang]['stop'])
        self.testBtn.setToolTip(language[self.lang]['test'])
        for grouplist in self.clientGroupList:
            if grouplist[3].text() == language['cn']['disconnected']:
                grouplist[3].setText(language[self.lang]['disconnected'])
            if grouplist[3].text() == language['en']['disconnected']:
                grouplist[3].setText(language[self.lang]['disconnected'])
            if grouplist[3].text() == language['cn']['connected']:
                grouplist[3].setText(language[self.lang]['connected'])
            if grouplist[3].text() == language['en']['connected']:
                grouplist[3].setText(language[self.lang]['connected'])

    def loadQSS(self):
        """ 加载QSS """
        file = 'qss/style/main.qss'
        with open(file, 'rt', encoding='utf8') as f:
            styleSheet = f.read()
        self.setStyleSheet(styleSheet)
        f.close()

    def initGUI(self):
        self.resize(800, 500)
        self.clientGroup = QGroupBox('Client ProgressBar', self)
        self.clientGroup.resize(350, 400)
        self.clientGroup.move(25, 25)
        self.clientGroupList = []
        self.cgvBox = QVBoxLayout()
        self.cgvBox.setSpacing(20)
        # print(self.config['parameter']['num_users'])
        for i in range(int(self.config['parameter']['num_users'])):
            cliSubgroup = QWidget(objectName='subgroup')
            cliLabel = QLabel('Client {}'.format(i))
            clivBox = QVBoxLayout()
            clihBox = QHBoxLayout()
            """

        layout.addWidget(CircleProgressBar(self))
        layout.addWidget(CircleProgressBar(
            self, color=QColor(255, 0, 0), clockwise=False))
        layout.addWidget(CircleProgressBar(self, styleSheet="" "
            qproperty-color: rgb(0, 255, 0);
        "" "))
            """
            
            cliProBar = QProgressBar()
            cliProBar.setTextVisible(False)
            cliProBar.setMaximum(100)
            cliProBar.setValue(40)
            cliProLabel = QLabel('{} %'.format(0))
            cliStateLabel = QLabel('Disconnected')
            clihBox.addWidget(cliLabel)
            clihBox.addWidget(cliStateLabel)
            clivBox.addLayout(clihBox)
            clivBox.addWidget(cliProBar)
            clivBox.addWidget(cliProLabel)
            cliSubgroup.setLayout(clivBox)
            effect_shadow = QGraphicsDropShadowEffect(self)
            effect_shadow.setOffset(3,3) # 偏移
            effect_shadow.setBlurRadius(10) # 阴影半径
            effect_shadow.setColor(QColor(38, 78, 200, 127)) # 阴影颜色
            cliSubgroup.setGraphicsEffect(effect_shadow)
            self.cgvBox.addWidget(cliSubgroup)
            self.clientGroupList.append([cliLabel, cliProBar, cliProLabel, cliStateLabel])

        # self.serGroupList = []
        # serLabel = QLabel('server')
        # serProBar = QProgressBar()
        # serProLabel = QLabel('{} %'.format(0))
        # self.cgvBox.addWidget(serLabel)
        # self.cgvBox.addWidget(serProBar)
        # self.cgvBox.addWidget(serProLabel)
        # self.clientGroupList.append()
        self.clientGroup.setLayout(self.cgvBox)

        self.processLabel0 = QLabel(chr(0xf00d), self)
        self.processLabel0.resize(20, 20)
        self.processLabel0.move(590, 480)
        self.processLabel0.setStyleSheet("color:red")
        self.processLabel0.setFont(qta.font('fa', 20))
        self.processLabel = QLabel(language[self.lang]['noconnect'], self)
        self.processLabel.resize(175, 20)
        self.processLabel.move(610, 480)
        
        self.initBtn = QPushButton(chr(0xf112), self, objectName='btnSuccess')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.initBtn.setGraphicsEffect(effect_shadow)
        self.initBtn.setToolTip(language[self.lang]['init env'])
        self.initBtn.setFont(qta.font('fa', 30))
        #self.initBtn = QPushButton('init env', self, objectName='btnSuccess')
        self.initBtn.resize(70, 70)
        self.initBtn.move(35, 450)
        self.initBtn.clicked.connect(self.init_env)

        self.startBtn = QPushButton(chr(0xf04b), self, objectName='btnSuccess')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.startBtn.setGraphicsEffect(effect_shadow)
        self.startBtn.setFont(qta.font('fa', 30))
        self.startBtn.setToolTip(language[self.lang]['start'])
        #self.startBtn = QPushButton('start', self, objectName='btnSuccess')
        self.startBtn.resize(70, 70)
        self.startBtn.move(160, 450)
        self.startBtn.setDisabled(True)
        # train thread
        self.trainThread = threading.Thread(target=self.start_train)
        self.startBtn.clicked.connect(self.trainThread.start)

        self.stopBtn = QPushButton(chr(0xf04d), self, objectName='btnSuccess')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.stopBtn.setGraphicsEffect(effect_shadow)
        self.stopBtn.setFont(qta.font('fa', 30))
        self.stopBtn.setToolTip(language[self.lang]['stop'])
        #self.stopBtn = QPushButton('stop', self, objectName='btnSuccess')
        self.stopBtn.resize(70, 70)
        self.stopBtn.move(285, 450)
        self.stopBtn.setDisabled(True)
        self.stopBtn.clicked.connect(self.stop_train)

        self.testBtn = QPushButton(chr(0xf040), self, objectName='btnPrimary')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 78, 200, 127)) # 阴影颜色
        self.testBtn.setGraphicsEffect(effect_shadow)
        self.testBtn.setToolTip(language[self.lang]['test'])
        self.testBtn.setFont(qta.font('fa', 30))
        #self.testBtn = QPushButton('test', self, objectName='btnInfo')
        self.testBtn.resize(70, 70)
        self.testBtn.move(600, 400)
        self.testBtn.clicked.connect(self.open_test_frame)

        self.lossLabel = QLabel(self)
        self.lossLabel.resize(300, 300)
        self.lossLabel.move(425, 25)
        self.contrLabel = QLabel(self)
        self.contrLabel.resize(300, 80)
        self.contrLabel.move(425, 345)

        self.setWindowTitle('ServerControlFrame {}'.format(self.id))
        #self.show()

    def open_test_frame(self):
        self.testframe = FramelessWindow()
        frame = TestFrame(self.config, lang=self.lang)
        self.testframe.setWindowTitle('TestFrame')
        self.testframe.setWindowIcon(QIcon('icon.png'))
        self.testframe.setFixedSize(QSize(600,480))  #因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
        self.testframe.setWidget(frame)  # 把自己的窗口添加进来
        self.testframe.titleBar.clickedChinese.connect(frame.clickedChinese)
        self.testframe.titleBar.clickedEnglish.connect(frame.clickedEnglish)
        self.testframe.show()

    def init_env(self):
        from paddle_fl.paddle_fl.core.scheduler.agent_master import FLScheduler
        import paddle.fluid as fluid
        # import paddle_fl as fl
        from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
        from paddle_fl.paddle_fl.core.strategy.fl_distribute_transpiler import FLDistributeTranspiler
        from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory, FedAvgStrategy

        if self.config['parameter']['model'] == 'resnet':
            model = ResNet18()
            # inputs = np.array([np.zeros((3, 224, 224)).astype('float32')]).astype('float32')
            inputs = fluid.layers.data(name='x', shape=[1, 3, 224, 224], dtype='float32')
            labels = np.array([0]).astype('float32').reshape(-1, 1)
            labels = fluid.layers.data(name='label', shape=[1, 1], dtype='float32')
            model.resnet(inputs, labels)
        else:
            inputs = [fluid.layers.data(
                name=str(slot_id), shape=[5],
                dtype="float32")
                for slot_id in range(3)]
            label = fluid.layers.data(
                name="label",
                shape=[1],
                dtype='int64')
            model = Model()
            model.mlp(inputs, label)

        job_generator = JobGenerator()
        optimizer = fluid.optimizer.SGD(learning_rate=0.1)
        job_generator.set_optimizer(optimizer)
        job_generator.set_losses([model.loss])
        job_generator.set_startup_program(model.startup_program)
        job_generator.set_infer_feed_and_target_names(
            [x.name for x in inputs], [model.predict.name])

        build_strategy = FLStrategyFactory()
        build_strategy.fed_avg = True
        build_strategy.inner_step = 1

        strategy = build_strategy.create_fl_strategy()

        endpoints = ['{}:{}'.format(self.config['server']['ip'], self.config['server']['port'])]
        output = self.config['path']['job_path']
        job_generator.generate_fl_job(
            strategy, server_endpoints=endpoints, worker_num=int(self.config['parameter']['num_users']), output=output)

        QMessageBox.information(self, self, language[self.lang]['compile ok'], self, language[self.lang]['compile env done'], QMessageBox.Ok)
        print('finish!')

        self.worker_num = int(self.config['parameter']['num_users'])

        self.server_num = 1
        # Define the number of worker/server and the port for scheduler
        self.scheduler = MFLScheduler(self.worker_num, self.server_num, port=int(self.config['scheduler']['port']))
        self.scheduler.set_sample_worker_num(self.worker_num)
        # self.scheduler.set_sample_worker_num(max(1, int(float(self.config['parameter']['frac']) * self.worker_num)))

        import paddle_fl as fl
        import paddle.fluid as fluid
        from paddle_fl.paddle_fl.core.server.fl_server import FLServer
        from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
        self.server = FLServer()
        server_id = self.id
        job_path = self.config['path']['job_path']
        print(job_path)
        self.job = FLRunTimeJob()
        self.job.load_server_job(job_path, server_id)
        self.job._scheduler_ep = '{}:{}'.format(self.config['scheduler']['ip'], self.config['scheduler']['port'])
        print(self.job._scheduler_ep)
        self.server.set_server_job(self.job)
        self.server._current_ep = '{}:{}'.format(self.config['server']['ip'], self.config['server']['port'])
        print(self.server._current_ep)

        self.processLabel.setText(language[self.lang]['waiting for agents'])
        # self.server.start()
        self.initThread = threading.Thread(target=self.scheduler.init_env)
        self.waitThread = threading.Thread(target=self.wait_agent)
        self.servThread = threading.Thread(target=self.server.start)
        self.initThread.start()
        self.servThread.start()
        self.waitThread.start()

        # self.servThread.join()
        # self.waitThread.join()

        print("init env done.")

        # scheduler.start_fl_training()

    def wait_agent(self):
        # print('?????????')
        cli_set = set([])
        # print(threading.activeCount())
        while self.connected_agent < self.worker_num:
            # print('{} ? {}'.format(self.connected_agent, self.worker_num))
            # print(self.scheduler.fl_workers)
            # print(self.scheduler.fl_servers)
            if self.connected_agent != len(self.scheduler.fl_workers):
                # print(self.scheduler.fl_workers, cli_set)
                new_cli = set(self.scheduler.fl_workers) - cli_set
                self.clientGroupList[self.connected_agent][3].setText(language[self.lang]['connected'])
                # print(new_cli)
                new_cli = list(new_cli)
                print('get client {} connection'.format(new_cli))
                cli_set = set(self.scheduler.fl_workers)
                self.connected_agent = len(self.scheduler.fl_workers)
        QMessageBox.information(self, language[self.lang]['init env'], language[self.lang]['init env done'], QMessageBox.Ok)
        self.startBtn.setDisabled(False)

    def start_train(self):
        self.stopBtn.setDisabled(False)
        self.startBtn.setDisabled(True)
        self.processLabel.setText(language[self.lang]['training'])
        self.scheduler.start_fl_training_with_round(int(self.config['parameter']['round']),
                                                    label=self.clientGroupList)
        print('train ok!')
        self.stopBtn.setDisabled(True)
        self.startBtn.setDisabled(False)
        self.processLabel.setText(language[self.lang]['finished'])
        QMessageBox.information(self, language[self.lang]['training progress'], language[self.lang]['The model has been trained!!'], QMessageBox.Ok)

    def stop_train(self):
        print(self.trainThread.is_alive())
        # self.stopBtn.setDisabled(True)
        # self.startBtn.setDisabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    yamlstream = open('config/config_server.yaml')
    ex = ServerControlFrame(yaml.load(yamlstream), 0)
    mainWnd = FramelessWindow()
    mainWnd.setWindowTitle('ServerControlFrame {}'.format(0))
    mainWnd.setWindowIcon(QIcon('icon.png'))
    mainWnd.setFixedSize(QSize(800,600))  #因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
    mainWnd.setWidget(ex)  # 把自己的窗口添加进来
    mainWnd.titleBar.clickedChinese.connect(ex.clickedChinese)
    mainWnd.titleBar.clickedEnglish.connect(ex.clickedEnglish)
    mainWnd.show()
    sys.exit(app.exec_())
