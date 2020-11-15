import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

# import interpretdl as it
import InterpretDL.interpretdl as it
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
import numpy as np

from models.resnet18.model_with_code.model import x2paddle_net

from datetime import datetime

last_layer_name = 'x2paddle_188.tmp_0'
model_path = 'models/resnet18/model_with_code'
test_result = 0
test_prob = 0
test_dic = {0: 'normal', 1: 'pun', 2: 'covid19'}


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
    def __init__(self):
        super().__init__()
        self.initGUI()
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
        self.resultLabel.resize(100, 50)

        self.choosepicLabel.move(50, 100)
        self.choosepicBtn.move(100, 350)
        self.processBtn.move(275, 175)
        self.resultpicLabel.move(350, 100)
        self.resultLabel.move(400, 300)
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
        sg = it.GradCAMInterpreter(paddle_model, self.modelPath, use_cuda=False,
                                   model_input_shape=[3, 224, 224])
        self.resultPath = 'result_{}_{}_{}_{}.jpg'.format(
            datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
        # print(self.resultPath)
        gradients, labels, output = sg.interpret(img_path, visual=True, target_layer_name=last_layer_name, save_path=self.resultPath)
        # print(labels, output)
        test_result = labels[0][0]
        test_prob = output[0][test_result]

        jpg = QtGui.QPixmap(self.resultPath).scaled(self.resultpicLabel.width(), self.resultpicLabel.height())
        self.resultpicLabel.setPixmap(jpg)
        # print(test_prob * 100)
        str = 'result: {}, probability: {:.2f}%'.format(test_dic[test_result], test_prob * 100)
        # print(str)
        self.resultLabel.setText(str)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    testFrame = TestFrame()
    sys.exit(app.exec_())
