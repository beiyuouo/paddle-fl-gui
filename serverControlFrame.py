import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import yaml
from utils.testFrame import *


class ServerControlFrame(QWidget):
    def __init__(self, config, id):
        super(ServerControlFrame, self).__init__()
        self.config = config
        self.id = id
        self.initGUI()

    def initGUI(self):
        self.resize(800, 500)
        self.clientGroup = QGroupBox('Client ProgressBar', self)
        self.clientGroup.resize(350, 350)
        self.clientGroup.move(25, 25)
        self.clientGroupList = []
        self.cgvBox = QVBoxLayout()
        # print(self.config['parameter']['num_users'])
        for i in range(int(self.config['parameter']['num_users'])):
            cliLabel = QLabel('client {}'.format(i))
            cliProBar = QProgressBar()
            cliProLabel = QLabel('{} %'.format(0))
            self.cgvBox.addWidget(cliLabel)
            self.cgvBox.addWidget(cliProBar)
            self.cgvBox.addWidget(cliProLabel)
            self.clientGroupList.append([cliLabel, cliProBar, cliProLabel])
        self.serGroupList = []
        serLabel = QLabel('server')
        serProBar = QProgressBar()
        serProLabel = QLabel('{} %'.format(0))
        self.cgvBox.addWidget(serLabel)
        self.cgvBox.addWidget(serProBar)
        self.cgvBox.addWidget(serProLabel)
        self.clientGroup.setLayout(self.cgvBox)

        self.initBtn = QPushButton('init env', self)
        self.initBtn.resize(100, 25)
        self.initBtn.move(25, 400)
        self.initBtn.clicked.connect(self.initenv)

        self.startBtn = QPushButton('start', self)
        self.startBtn.resize(100, 25)
        self.startBtn.move(150, 400)

        self.stopBtn = QPushButton('stop', self)
        self.stopBtn.resize(100, 25)
        self.stopBtn.move(275, 400)

        self.testBtn = QPushButton('test', self)
        self.testBtn.resize(100, 25)
        self.testBtn.move(550, 450)
        self.testBtn.clicked.connect(self.opentestwin)

        self.lossLabel = QLabel(self)
        self.lossLabel.resize(300, 300)
        self.lossLabel.move(425, 25)
        self.contrLabel = QLabel(self)
        self.contrLabel.resize(300, 80)
        self.contrLabel.move(425, 345)

        self.setWindowTitle('ServerControlFrame {}'.format(self.id))
        self.show()

    def opentestwin(self):
        self.testframe = TestFrame()
        self.testframe.show()

    def initenv(self):
        from paddle_fl.paddle_fl.core.scheduler.agent_master import FLScheduler
        import paddle.fluid as fluid
        import paddle_fl as fl
        from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
        from paddle_fl.paddle_fl.core.strategy.fl_distribute_transpiler import FLDistributeTranspiler
        from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory, FedAvgStrategy

        class Model(object):
            def __init__(self):
                pass

            def mlp(self, inputs, label, hidden_size=128):
                self.concat = fluid.layers.concat(inputs, axis=1)
                self.fc1 = fluid.layers.fc(input=self.concat, size=256, act='relu')
                self.fc2 = fluid.layers.fc(input=self.fc1, size=128, act='relu')
                self.predict = fluid.layers.fc(input=self.fc2, size=2, act='softmax')
                self.sum_cost = fluid.layers.cross_entropy(input=self.predict, label=label)
                self.accuracy = fluid.layers.accuracy(input=self.predict, label=label)
                self.loss = fluid.layers.reduce_mean(self.sum_cost)
                self.startup_program = fluid.default_startup_program()

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

        endpoints = ["127.0.0.1:8181"]
        output = self.config['path']['fl_job_config']
        job_generator.generate_fl_job(
            strategy, server_endpoints=endpoints, worker_num=int(self.config['parameter']['num_users']), output=output)

        QMessageBox.information(self, 'compile ok', 'compile env done', QMessageBox.Ok)
        print('finish!')

        self.worker_num = int(self.config['parameter']['num_users'])
        self.server_num = 1
        # Define the number of worker/server and the port for scheduler
        self.scheduler = FLScheduler(self.worker_num, self.server_num, port=self.config['scheduler']['port'])
        self.scheduler.set_sample_worker_num(self.worker_num)
        self.scheduler.init_env()
        QMessageBox.information(self, 'init env', 'init env done', QMessageBox.Ok)
        print("init env done.")

        # scheduler.start_fl_training()

    def starttrain(self):
        self.scheduler.start_fl_training()

    def stoptrain(self):
        pass
        # self.scheduler.stop()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    yamlstream = open('config/config_server.yaml')
    ex = ServerControlFrame(yaml.load(yamlstream), 1)
    sys.exit(app.exec_())
