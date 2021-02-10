import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import yaml

from serverControlFrame import *
from style import FramelessWindow, CircleProgressBar, language
import qtawesome as qta


class ServerFrame(QWidget):
    def __init__(self, id=0, lang='cn'):
        super(ServerFrame, self).__init__()
        self.id = id
        self.lang = lang
        self.config_file = 'config/config_server.yaml'
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
        self.model_path_label.setText(language[self.lang]['model save path'])
        self.job_path_label.setText(language[self.lang]['job save path'])
        self.startBtn.setText(language[self.lang]['start'])
        
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
        effect_shadow.setColor(QColor(130, 130, 130, 127)) # 阴影颜色
        self.paramTab.setGraphicsEffect(effect_shadow)
        self.paramTab.resize(350, 350)
        self.paramTab.move(50, 50)
        self.paramTab.setColumnCount(2)
        self.paramTab.setColumnWidth(0, 100)
        self.paramTab.setColumnWidth(1, 180)
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
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(100, 100, 100, 127)) # 阴影颜色
        self.model_path_text.setGraphicsEffect(effect_shadow)
        self.model_path_btn = QPushButton('...', self, objectName='btnSuccess3')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.model_path_btn.setGraphicsEffect(effect_shadow)
        self.model_path_label.resize(200, 30)
        self.model_path_label.move(450, 70)
        self.model_path_text.resize(275, 30)
        self.model_path_text.move(450, 100)
        self.model_path_btn.resize(50, 30)
        self.model_path_btn.move(730, 100)

        self.job_path_label = QLabel('job save path:', self)
        self.job_path_text = QLineEdit(self.config['path']['job_path'], self)
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.job_path_text.setGraphicsEffect(effect_shadow)
        self.job_path_btn = QPushButton('...', self, objectName='btnSuccess3')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 200, 78, 127)) # 阴影颜色
        self.job_path_btn.setGraphicsEffect(effect_shadow)

        self.job_path_label.resize(200, 30)
        self.job_path_label.move(450, 170)
        self.job_path_text.resize(275, 30)
        self.job_path_text.move(450, 200)
        self.job_path_btn.resize(50, 30)
        self.job_path_btn.move(730, 200)

        self.startBtnIcon1 = qta.icon('fa.play', scale_factor = 1, color='white')
        self.startBtn = QPushButton(self.startBtnIcon1, "Start", self, objectName='btnPrimary2')
        effect_shadow = QGraphicsDropShadowEffect(self)
        effect_shadow.setOffset(3,3) # 偏移
        effect_shadow.setBlurRadius(10) # 阴影半径
        effect_shadow.setColor(QColor(38, 78, 200, 127)) # 阴影颜色
        self.startBtn.setGraphicsEffect(effect_shadow)
        self.startBtn.resize(150, 40)
        #self.startBtn = QPushButton('start', self)
        self.startBtn.move(550, 400)
        self.startBtn.clicked.connect(self.open_scf)

        # self.stopBtn = QPushButton('stop', self)
        # self.stopBtn.move(600, 400)

        self.setWindowTitle('Server {}'.format(self.id))
        #self.show()

    def open_scf(self):
        self.scf = FramelessWindow()
        frame = ServerControlFrame(self.config, self.id, lang=self.lang)
        self.scf.setWindowTitle('ServerControlFrame {}'.format(self.id))
        self.scf.setWindowIcon(QIcon('icon.png'))
        self.scf.setFixedSize(QSize(800,600))  #因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
        self.scf.setWidget(frame)  # 把自己的窗口添加进来
        self.scf.titleBar.clickedChinese.connect(frame.clickedChinese)
        self.scf.titleBar.clickedEnglish.connect(frame.clickedEnglish)
        self.scf.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ServerFrame()
    mainWnd = FramelessWindow()
    mainWnd.setWindowTitle('Server {}'.format(0))   #sys.argv[1]
    mainWnd.setWindowIcon(QIcon('icon.png'))
    mainWnd.setFixedSize(QSize(800,600))  #因为这里固定了大小，所以窗口的大小没有办法任意调整，想要使resizeWidget函数生效的话要把这里去掉，自己调节布局和窗口大小
    mainWnd.setWidget(ex)  # 把自己的窗口添加进来
    mainWnd.titleBar.clickedChinese.connect(ex.clickedChinese)
    mainWnd.titleBar.clickedEnglish.connect(ex.clickedEnglish)
    mainWnd.show()
    sys.exit(app.exec_())
