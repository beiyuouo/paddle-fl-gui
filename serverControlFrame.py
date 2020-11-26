import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import threading
import yaml
from utils.testFrame import *


class ServerControlFrame(QWidget):
    def __init__(self, config, id):
        super(ServerControlFrame, self).__init__()
        self.config = config
        self.id = id
        self.connected_agent = 0
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
            cliStateLabel = QLabel('disconnected')
            self.cgvBox.addWidget(cliLabel)
            self.cgvBox.addWidget(cliProBar)
            self.cgvBox.addWidget(cliProLabel)
            self.cgvBox.addWidget(cliStateLabel)
            self.clientGroupList.append([cliLabel, cliProBar, cliProLabel, cliStateLabel])

        # self.serGroupList = []
        serLabel = QLabel('server')
        serProBar = QProgressBar()
        serProLabel = QLabel('{} %'.format(0))
        self.cgvBox.addWidget(serLabel)
        self.cgvBox.addWidget(serProBar)
        self.cgvBox.addWidget(serProLabel)
        self.clientGroup.setLayout(self.cgvBox)

        self.processLabel = QLabel('noconnect', self)
        self.processLabel.resize(175, 20)
        self.processLabel.move(600, 470)

        self.initBtn = QPushButton('init env', self)
        self.initBtn.resize(100, 25)
        self.initBtn.move(25, 400)
        self.initBtn.clicked.connect(self.init_env)

        self.startBtn = QPushButton('start', self)
        self.startBtn.resize(100, 25)
        self.startBtn.move(150, 400)
        self.startBtn.setDisabled(True)
        self.startBtn.clicked.connect(self.start_train)

        self.stopBtn = QPushButton('stop', self)
        self.stopBtn.resize(100, 25)
        self.stopBtn.move(275, 400)
        self.stopBtn.setDisabled(True)

        self.testBtn = QPushButton('test', self)
        self.testBtn.resize(100, 25)
        self.testBtn.move(550, 450)
        self.testBtn.clicked.connect(self.open_test_frame)

        self.lossLabel = QLabel(self)
        self.lossLabel.resize(300, 300)
        self.lossLabel.move(425, 25)
        self.contrLabel = QLabel(self)
        self.contrLabel.resize(300, 80)
        self.contrLabel.move(425, 345)

        self.setWindowTitle('ServerControlFrame {}'.format(self.id))
        self.show()

    def open_test_frame(self):
        self.testframe = TestFrame()
        self.testframe.show()

    def init_env(self):
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

        endpoints = ['{}:{}'.format(self.config['server']['ip'], self.config['server']['port'])]
        output = self.config['path']['job_path']
        job_generator.generate_fl_job(
            strategy, server_endpoints=endpoints, worker_num=int(self.config['parameter']['num_users']), output=output)

        QMessageBox.information(self, 'compile ok', 'compile env done', QMessageBox.Ok)
        print('finish!')

        self.worker_num = int(self.config['parameter']['num_users'])
        self.server_num = 1
        # Define the number of worker/server and the port for scheduler
        self.scheduler = FLScheduler(self.worker_num, self.server_num, port=int(self.config['scheduler']['port']))
        self.scheduler.set_sample_worker_num(self.worker_num)

        # self.listenThread = threading.Thread(target=self.scheduler.init_env)
        # self.listenThread.start()
        # self.listenThread.join()

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

        self.processLabel.setText('waiting for agents')
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
        print('?????????')
        cli_set = set([])
        print(threading.activeCount())
        while self.connected_agent < self.worker_num:
            # print('{} ? {}'.format(self.connected_agent, self.worker_num))
            # print(self.scheduler.fl_workers)
            # print(self.scheduler.fl_servers)
            if self.connected_agent != len(self.scheduler.fl_workers):
                print(self.scheduler.fl_workers, cli_set)
                new_cli = set(self.scheduler.fl_workers) - cli_set
                self.clientGroupList[self.connected_agent][3].setText('connected')
                print(new_cli)
                new_cli = list(new_cli)
                print('get client {} connection'.format(new_cli))
                cli_set = set(self.scheduler.fl_workers)
                self.connected_agent = len(self.scheduler.fl_workers)
        QMessageBox.information(self, 'init env', 'init env done', QMessageBox.Ok)
        self.startBtn.setDisabled(False)

    def start_train(self):
        self.scheduler.start_fl_training()

    def stop_train(self):
        pass
        # self.scheduler.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    yamlstream = open('config/config_server.yaml')
    ex = ServerControlFrame(yaml.load(yamlstream), 0)
    sys.exit(app.exec_())
