import os
import sys

import yaml
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

# import interpretdl as it

import utils.InterpretDL.interpretdl as it
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
import numpy as np

from utils.models.resnet18.model_with_code.model import x2paddle_net

from datetime import datetime

from utils.testInterpreter import MyGradCAMInterpreter, plot_bounding_box

last_layer_name = 'x2paddle_188.tmp_0'
model_path = 'utils/models/resnet18/model_with_code'
test_result = 0
test_prob = 0
test_dic = {0: 'pneumonia', 1: 'normal', 2: 'COVID-19'}


def paddle_model(data):
    import os
    inputs, outputs = x2paddle_net(input=data)
    ops = fluid.default_main_program().global_block().ops
    used_vars = list()
    for op in ops:
        used_vars += op.input_arg_names
    tmp = list()
    for input in inputs:
        if isinstance(input, list):
            for ipt in input:
                if ipt.name not in used_vars:
                    continue
                tmp.append(ipt)
        else:
            if input.name not in used_vars:
                continue
            tmp.append(input)
    inputs = tmp
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    outputs = outputs[0]
    probs = fluid.layers.softmax(outputs, axis=-1)
    # print(probs[0])
    # print(np.argmax(probs[0]))
    # global test_result, test_prob
    # test_result = np.argmax(probs[0])
    # test_prob = probs[0][test_result]
    # print(outputs.shape)
    return probs


class TestFrame(QWidget):
    def __init__(self, config):
        super().__init__()
        print('called')
        self.initGUI()
        self.config = config
        self.modelPath = model_path

    def initGUI(self):
        self.resize(600, 400)
        self.choosepicLabel = QLabel("选择图片", self)
        self.choosepicLabel.resize(200, 200)
        self.choosepicBtn = QPushButton("选择图片", self)
        self.choosepicBtn.clicked.connect(self.openimage)

        self.processBtn = QPushButton("=>", self)
        self.imgPath = ''
        self.processBtn.clicked.connect(self.processGradCAM)
        self.processBtn.resize(50, 30)

        self.resultpicLabel = QLabel("", self)
        self.resultpicLabel.resize(200, 200)
        self.resultLabel = QLabel("结果", self)
        self.resultLabel.resize(200, 80)

        self.choosepicLabel.move(50, 100)
        self.choosepicBtn.move(100, 350)
        self.processBtn.move(275, 175)
        self.resultpicLabel.move(350, 100)
        self.resultLabel.move(350, 280)

        self.reportBtn = QPushButton("generate report", self)
        self.reportBtn.resize(200, 30)
        self.reportBtn.move(350, 350)
        self.reportBtn.clicked.connect(self.generate_report)

        self.setWindowTitle('TestFrame')
        self.show()

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "Image File(*.jpg *.png *.bmp)")
        self.picPath = imgName
        jpg = QtGui.QPixmap(imgName).scaled(self.choosepicLabel.width(), self.choosepicLabel.height())
        self.choosepicLabel.setPixmap(jpg)

    def processGradCAM(self):
        img_path = self.picPath
        # print(self.picPath)
        sg = MyGradCAMInterpreter(paddle_model, self.modelPath, use_cuda=False,
                                  model_input_shape=[3, 224, 224])
        self.resultPath = 'reports/result_{:02d}_{:02d}_{:02d}_{:02d}'.format(
            datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
        # print(self.resultPath)
        gradients, labels, output, img0 = sg.interpret(img_path, thresholds=float(self.config['test']['thresholds']),
                                                       visual=True, target_layer_name=last_layer_name,
                                                       save_path=self.resultPath)
        # print(labels, output)
        for label, out in zip(labels, output):
            print('label: {}, prop: {}'.format(label, out))

        plot_bounding_box(labels, gradients, img0, self.resultPath)

        self.model_output = output
        self.model_labels = labels

        test_result = labels[0][0]
        test_prob = output[0][test_result]

        jpg = QtGui.QPixmap(os.path.join(self.resultPath, '{}_with_bbox.jpg'.format(test_result))).scaled(
            self.resultpicLabel.width(), self.resultpicLabel.height())
        self.resultpicLabel.setPixmap(jpg)
        # print(test_prob * 100)
        str = 'result: {},\nprobability: {:.2f}%'.format(test_dic[test_result], test_prob * 100)
        print(str)
        self.resultLabel.setText(str)

    def generate_report(self):
        import webbrowser
        gen_html = os.path.join(self.resultPath, "report.html")
        # 打开文件，准备写入
        f = open(gen_html, 'w')

        # 准备相关变量
        str1 = 'my name is :'
        str2 = '--MichaelAn--'

        # 写入HTML界面中
        message_f = """
        <html>
        <head></head>
        <body style="text-align:center;margin-left:auto;margin-right:auto;">
        """

        message_m = ""
        for i in range(len(self.model_labels)):
            message_m += "<img src={} width='400' height='400'></img>\n".format('{}_with_bbox.jpg'.format(i))
            message_m += "<br>\n"
            message_m += "<p>\n"
            message_m += 'label: {}, prop: {}\n'.format(test_dic[self.model_labels[i][0]], self.model_output[0][i])
            message_m += "</p>\n"
            message_m += "<br>\n"

        message_b = """
        </body>
        </html>
        """

        message = message_f + message_m + message_b

        # 写入文件
        f.write(message)
        # 关闭文件
        f.close()

        # 运行完自动在网页中显示
        webbrowser.open(gen_html, new=1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    yamlstream = open('config/config_client.yaml')
    testFrame = TestFrame(yaml.load(yamlstream))
    sys.exit(app.exec_())
