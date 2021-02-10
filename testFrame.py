import os
import sys

import yaml
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtGui, QtCore

# import interpretdl as it

import utils.InterpretDL.interpretdl as it
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
import numpy as np

from utils.models.resnet18.model_with_code.model import x2paddle_net

from datetime import datetime

from utils.testInterpreter import MyGradCAMInterpreter, plot_bounding_box
from style import FramelessWindow, CircleProgressBar, language
import qtawesome as qta

last_layer_name = 'x2paddle_188.tmp_0'
model_path = 'utils/models/resnet18/model_with_code'
test_result = 0
test_prob = 0
test_dic = {0: '普通肺炎', 1: '正常', 2: '新冠肺炎COVID-19'}
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
    def __init__(self, config, lang='cn'):
        super().__init__()
        print('called')
        self.lang = lang
        self.loadQSS()
        self.initGUI()
        self.config = config
        self.modelPath = model_path
        self.translateAll()

    def clickedChinese(self):
        self.lang = "cn"
        self.translateAll()

    def clickedEnglish(self):
        self.lang = "en"
        self.translateAll()

    def translateAll(self):
        self.choosepicLabel.setText(language[self.lang]['select image'])
        self.choosepicBtn.setText(language[self.lang]['select image'])
        self.resultLabel.setText(language[self.lang]['result'])
        self.reportBtn.setText(language[self.lang]['generate'])

    def loadQSS(self):
        """ 加载QSS """
        file = 'qss/style/main.qss'
        with open(file, 'rt', encoding='utf8') as f:
            styleSheet = f.read()
        self.setStyleSheet(styleSheet)
        f.close()

    def initGUI(self):
        self.resize(600, 400)
        self.choosepicLabel = QLabel("选择图片", self)
        self.choosepicLabel.resize(200, 200)

        self.choosepicBtnIcon1 = qta.icon('fa.image', scale_factor=1, color='white')
        self.choosepicBtn = QPushButton(self.choosepicBtnIcon1, "选择图片", self, objectName='btnPrimary2')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3, 3)  # 偏移
        effect_shadow.setBlurRadius(10)  # 阴影半径
        effect_shadow.setColor(QColor(38, 78, 200, 127))  # 阴影颜色
        self.choosepicBtn.setGraphicsEffect(effect_shadow)
        self.choosepicBtn.clicked.connect(self.openimage)

        self.processBtn = QPushButton(chr(0xf061), self, objectName='btnArrow')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3, 3)  # 偏移
        effect_shadow.setBlurRadius(10)  # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127))  # 阴影颜色
        self.processBtn.setGraphicsEffect(effect_shadow)
        self.processBtn.setFont(qta.font('fa', 20))
        self.processBtn.resize(50, 50)
        self.imgPath = ''
        self.processBtn.clicked.connect(self.processGradCAM)

        self.resultpicLabel = QLabel("", self)
        self.resultpicLabel.resize(200, 200)
        self.resultLabel = QLabel("结果", self)
        self.resultLabel.resize(200, 80)

        self.choosepicLabel.move(50, 100)
        self.choosepicBtn.move(100, 350)
        self.processBtn.move(275, 175)
        self.resultpicLabel.move(350, 100)
        self.resultLabel.move(350, 280)

        self.reportBtnIcon1 = qta.icon('fa.sign-out', scale_factor=1, color='white')
        self.reportBtn = QPushButton(self.reportBtnIcon1, "生成报告", self, objectName='btnSuccess2')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3, 3)  # 偏移
        effect_shadow.setBlurRadius(10)  # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127))  # 阴影颜色
        self.reportBtn.setGraphicsEffect(effect_shadow)
        self.reportBtn.resize(200, 30)
        self.reportBtn.move(350, 350)
        self.reportBtn.clicked.connect(self.generate_report)

        self.setWindowTitle('TestFrame')
        # self.show()

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "Image File(*.jpg *.png *.bmp)")
        self.picPath = imgName
        jpg = QtGui.QPixmap(imgName).scaled(self.choosepicLabel.width(), self.choosepicLabel.height())
        self.choosepicLabel.setPixmap(jpg)
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3, 3)  # 偏移
        effect_shadow.setBlurRadius(10)  # 阴影半径
        effect_shadow.setColor(QColor(38, 78, 200, 127))  # 阴影颜色
        self.choosepicLabel.setGraphicsEffect(effect_shadow)

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

        jpg = QtGui.QPixmap(os.path.join(self.resultPath, '{}_with_bbox.jpg'.format(0))).scaled(
            self.resultpicLabel.width(), self.resultpicLabel.height())
        self.resultpicLabel.setPixmap(jpg)
        # print(test_prob * 100)
        str = language[self.lang]['result'] + ': {},\n'.format(test_dic[test_result]) + language[self.lang][
            'probability'] + ': {:.2f}%'.format(test_prob * 100)
        print(str)
        self.resultLabel.setText(str)

    def generate_report_cn(self):
        message_f = """
                <html>
        <title>CT检查报告单</title>
        <head>
            <style type="text/css">
                td {
                    width: 25%;
                }
            </style>
        </head>
        <body style="text-align: center; width: 50% margin: 0 auto;">
        <div style="font-size: 30px"><p style="margin: 0 auto;">海 南 省 X X 医 院</p></div>
        <div style="font-size: 25px;">
            <p style="margin: 0 auto;">C T 检 查 报 告 单</p>
            <p style="font-size: 10px; margin: 0 auto;"> CT号: 20201229xxxxxxxx</p>
        </div>
        <hr style="height: 3px; width: 50%; border: none; border-top: 3px solid #555555; margin: 5 auto 0;" />
        <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 0 auto;" />
        <table style="border: 0; text-align: center; margin: 0 auto;">
          <tr>
            <td>姓名：小阿昆</td>
            <td>性别：不告诉你</td>
            <td>年龄：20</td>
            <td>摄片日期：2020-12-30</td>
          </tr>
          <tr>
            <td>科别：心胸外科</td>
            <td>床号：</td>
            <td>住院号：</td>
            <td>报告日期：2020-12-30</td>
          </tr>
        </table>
        <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 10 auto 0;" />
        <div style="text-align: left; margin-left: 30%;">
            <p><b>检验项目：</b>胸腔</p>
            <p><b>扫描方式：</b> 平扫、三维重组</p>
            <p><b>影像表现：</b></p>
                """

        message_m = ""
        for i in range(len(self.model_labels)):
            message_m += "<img src={} width='400' height='400'></img>\n".format('{}_with_bbox.jpg'.format(i))
            message_m += "<br>\n"
            message_m += "<p>"
            message_m += '该处可能存在{}病症，置信度：{:.2f}%'.format(test_dic[self.model_labels[i][0]],
                                                         self.model_output[0][i] * 100)
            message_m += "</p>"
            message_m += "<br>\n"

        message_b = """
        </div>
        <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 0 auto;" />
        <table style="border: 0; text-align: center; margin: 0 auto;">
          <tr>
            <td style="width: 15%;">报告医生：</td>
            <td style="width: 15%;">审核医生：</td>
          </tr>
        </table>
        <p>*注：本报告仅供临床医师参考，不做他用，影像科医生签字后有效。</p>
        <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 0 auto;" />
        <hr style="height: 3px; width: 50%; border: none; border-top: 3px solid #555555; margin: 5 auto 0;" />


        </body>
        </html>
                """

        message = message_f + message_m + message_b

        return message

    def generated_report_en(self):
        message_f = """
                        <html>
                <title>Chest CT Findings Related to COVID19 Report</title>
                <head>
                    <style type="text/css">
                        td {
                            width: 25%;
                        }
                    </style>
                </head>
                <body style="text-align: center; width: 50% margin: 0 auto;">
                <div style="font-size: 30px"><p style="margin: 0 auto;">Hospital</p></div>
                <div style="font-size: 25px;">
                    <p style="margin: 0 auto;">Chest CT Findings Report</p>
                    <p style="font-size: 10px; margin: 0 auto;"> No. 20201229xxxxxxxx</p>
                </div>
                <hr style="height: 3px; width: 50%; border: none; border-top: 3px solid #555555; margin: 5 auto 0;" />
                <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 0 auto;" />
                <table style="border: 0; text-align: center; margin: 0 auto;">
                  <tr>
                    <td>Name: Alan</td>
                    <td>Gender: -</td>
                    <td>Age: 20</td>
                    <td>Date: 2020-12-30</td>
                  </tr>
                  <tr>
                    <td>Clinic:</td>
                    <td>Room no.: </td>
                    <td>Bed no.: </td>
                    <td>Report Date: 2020-12-30</td>
                  </tr>
                </table>
                <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 10 auto 0;" />
                <div style="text-align: left; margin-left: 30%;">
                    <p><b>Scan position: </b>Chest</p>
                    <p><b>Method:</b> Flat sweep, Three-dimensional reconstruction</p>
                    <p><b>Imaging Findings:</b></p>
                        """

        message_m = ""
        for i in range(len(self.model_labels)):
            message_m += "<img src={} width='400' height='400'></img>\n".format('{}_with_bbox.jpg'.format(i))
            message_m += "<br>\n"
            message_m += "<p>"
            message_m += 'Symptom {} may exist here, with a confidence of :{:.2f}%'.format(
                test_dic[self.model_labels[i][0]],self.model_output[0][i] * 100)
            message_m += "</p>"
            message_m += "<br>\n"

        message_b = """
                </div>
                <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 0 auto;" />
                <table style="border: 0; text-align: center; margin: 0 auto;">
                  <tr>
                    <td style="width: 15%;">Report Dr.:</td>
                    <td style="width: 15%;">Audit Dr.:</td>
                  </tr>
                </table>
                <p>*Note: This report is for the reference of clinicians only, not for other purposes, it is valid <br>
                after the imaging doctor signs.</p>
                <hr style="height: 3px; width: 50%; border: none; border-top: 1px solid #555555; margin: 0 auto;" />
                <hr style="height: 3px; width: 50%; border: none; border-top: 3px solid #555555; margin: 5 auto 0;" />


                </body>
                </html>
                        """

        message = message_f + message_m + message_b
        return message

    def generate_report(self):
        import webbrowser
        gen_html = os.path.join(self.resultPath, "report.html")
        # 打开文件，准备写入
        f = open(gen_html, 'w')

        # 准备相关变量
        str1 = 'my name is :'
        str2 = '--MichaelAn--'

        # 写入HTML界面中
        if self.lang == "cn":
            message = self.generate_report_cn()
        else:
            message = self.generated_report_en()

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
    mainWnd = FramelessWindow()
    mainWnd.setWindowTitle('TestFrame')
    mainWnd.setWindowIcon(QIcon('icon.png'))
    mainWnd.setFixedSize(QSize(600, 480))  # 因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
    mainWnd.setWidget(testFrame)  # 把自己的窗口添加进来
    mainWnd.titleBar.clickedChinese.connect(testFrame.clickedChinese)
    mainWnd.titleBar.clickedEnglish.connect(testFrame.clickedEnglish)
    mainWnd.show()
    sys.exit(app.exec_())
