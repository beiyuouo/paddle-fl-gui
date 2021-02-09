from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import qtawesome as qta

language = {
    'cn':
        {
            'param': '参数',
            'value': '值',
            'connect': '连接',
            'test': '测试',
            'start': '开始',
            'model save path': '模型保存位置',
            'job save path': '职业保存位置',
            'noconnect': '没有连接',
            'select image': '选择图片',
            'generate': '生成报告',
            'result': '结果',
            'disconnected': '已断开',
            'init env': '初始化环境',
            'stop': '停止',
            'Client ProgressBar': 'Client进度条',
            'probability': '概率',
            'connected': '已连接',
            'training': '训练中',
            'finished': '一结束',
            'init env done': '环境初始化成功',
            'training progress': '训练进度',
            'The model has been trained!!': '模型已经被训练完成!!',
            'waiting for agents': '等待中...',
            'compile ok':'编译成功',
            'compile env done':'编译完成',
        },
    'en':
        {
            'param': 'Param',
            'value': 'Value',
            'connect': 'Connect',
            'test': 'Test',
            'start': 'Start',
            'model save path': 'Model save path',
            'job save path': 'Job save path',
            'noconnect': 'No connect',
            'select image': 'Select',
            'generate': 'Generate Result',
            'result': 'Result',
            'disconnected': 'Disconnected',
            'init env': 'Init Environment',
            'stop': 'Stop',
            'Client ProgressBar': 'Client ProgressBar',
            'probability': 'Probability',
            'connected': 'Connected',
            'training': 'Training',
            'finished': 'Finished',
            'init env done': 'Init environment done',
            'training progress': 'Training progress',
            'The model has been trained!!': 'The model has been trained!!',
            'waiting for agents': 'waiting for agents',
            'compile ok':'compile ok',
            'compile env done':'compile env done',
            
        },
    }

StyleSheet = """
/*最小化最大化关闭按钮通用默认背景*/
#buttonEnglish,#buttonChinese {
    border: none;
    color:black;
}
#buttonEnglish:hover,#buttonChinese:hover {
    color:grey;
}
#buttonEnglish:pressed,#buttonChinese:pressed {
    color:black;
}
#buttonMinimum,#buttonMaximum,#buttonClose {
    border: none;
}
#buttonClose,#buttonMaximum,#buttonMinimum{
    color:black;
}
/*悬停*/
#buttonMinimum:hover,#buttonMaximum:hover {
    color: grey;
}
#buttonClose:hover {
    color: grey;
}
/*鼠标按下不放*/
#buttonMinimum:pressed,#buttonMaximum:pressed {
    color:black;
}
#buttonClose:pressed {
    color: grey;

}
"""

class TitleBar(QWidget):

    # 窗口最小化信号
    windowMinimumed = pyqtSignal()
    # 窗口最大化信号
    windowMaximumed = pyqtSignal()
    # 窗口还原信号
    windowNormaled = pyqtSignal()
    # 窗口关闭信号
    windowClosed = pyqtSignal()

    # 窗口移动
    windowMoved = pyqtSignal(QPoint)
    
    clickedChinese = pyqtSignal()
    clickedEnglish = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(TitleBar, self).__init__(*args, **kwargs)
        self.setStyleSheet(StyleSheet)
        self.mPos = None
        self.iconSize = 20  # 图标的默认大小

        # 布局
        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)
        # 空白
        self.tempLabel = QLabel(self)
        self.tempLabel.setStyleSheet("color:black")
        self.tempLabel.setMargin(3)
        self.tempLabel.setText("  ")
        layout.addWidget(self.tempLabel)
        # 窗口图标
        self.iconLabel = QLabel(self)
        self.iconLabel.setMargin(3)
#         self.iconLabel.setScaledContents(True)
        # 窗口标题
        self.titleLabel = QLabel(self)
        self.titleLabel.setStyleSheet("color:black")
        self.titleLabel.setMargin(2)
        # 中间伸缩条
        layout.addSpacerItem(QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addWidget(self.iconLabel)
        layout.addWidget(self.titleLabel)
        # 中间伸缩条
        layout.addSpacerItem(QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        # 利用Webdings字体来显示图标
        font = self.font() or QFont()
        font.setFamily('Webdings')
        # 中文按钮
        self.buttonChinese = QPushButton(
            '中文', self, clicked=self.clickedChinese.emit, objectName='buttonChinese')
        layout.addWidget(self.buttonChinese)
        # 英文按钮
        self.buttonEnglish = QPushButton(
            'English', self, clicked=self.clickedEnglish.emit, objectName='buttonEnglish')
        layout.addWidget(self.buttonEnglish)
        # 最小化按钮
        self.buttonMinimum = QPushButton(
            '0', self, clicked=self.windowMinimumed.emit, font=font, objectName='buttonMinimum')
        layout.addWidget(self.buttonMinimum)
        # 最大化/还原按钮
        #self.buttonMaximum = QPushButton(
        #    '1', self, clicked=self.showMaximized, font=font, objectName='buttonMaximum')
        #layout.addWidget(self.buttonMaximum)
        # 关闭按钮
        self.buttonClose = QPushButton(
            'r', self, clicked=self.windowClosed.emit, font=font, objectName='buttonClose')
        layout.addWidget(self.buttonClose)
        # 初始高度
        self.setHeight()
        """
    def showMaximized(self):
        if self.buttonMaximum.text() == '1':
            # 最大化
            self.buttonMaximum.setText('2')
            self.windowMaximumed.emit()
        else:  # 还原
            self.buttonMaximum.setText('1')
            self.windowNormaled.emit()
        """
    def setHeight(self, height=50):
        """设置标题栏高度"""
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        # 设置右边按钮的大小
        self.buttonMinimum.setMinimumSize(height - 20, height)
        self.buttonMinimum.setMaximumSize(height - 20, height)
        self.buttonChinese.setMinimumSize(height + 20, height)
        self.buttonEnglish.setMinimumSize(height + 40, height)
        self.buttonChinese.setMaximumSize(height + 20, height)
        self.buttonEnglish.setMaximumSize(height + 40, height)
        """
        self.buttonMaximum.setMinimumSize(height, height)
        self.buttonMaximum.setMaximumSize(height, height)
        """
        self.buttonClose.setMinimumSize(height, height)
        self.buttonClose.setMaximumSize(height, height)

    def setTitle(self, title):
        """设置标题"""
        self.titleLabel.setText(title)

    def setIcon(self, icon):
        """设置图标"""
        self.iconLabel.setPixmap(icon.pixmap(self.iconSize, self.iconSize))

    def setIconSize(self, size):
        """设置图标大小"""
        self.iconSize = size

    def enterEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super(TitleBar, self).enterEvent(event)

    def mouseDoubleClickEvent(self, event):
        super(TitleBar, self).mouseDoubleClickEvent(event)
        self.showMaximized()

    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.LeftButton:
            self.mPos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        '''鼠标弹起事件'''
        self.mPos = None
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.mPos:
            self.windowMoved.emit(self.mapToGlobal(event.pos() - self.mPos))
        event.accept()

# 枚举左上右下以及四个定点
Left, Top, Right, Bottom, LeftTop, RightTop, LeftBottom, RightBottom = range(8)



class FramelessWindow(QWidget):

    # 四周边距
    Margins = 5

    def __init__(self, *args, **kwargs):
        super(FramelessWindow, self).__init__(*args, **kwargs)
        self.border_width = 8
        self.lang = "cn"
        #palette1 = QtGui.QPalette()
        #palette1.setBrush(self.backgroundRole(), QtGui.QBrush(
        #    QtGui.QPixmap('log0.jpg')))  # 设置登录背景图片
        #self.setPalette(palette1)
        #self.setAutoFillBackground(True)
        self.setGeometry(300, 300, 250, 150)
        self._pressed = False
        self.Direction = None
        # 无边框
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window) # 隐藏边框
        # 鼠标跟踪
        self.setMouseTracking(True)
        # 布局
        layout = QVBoxLayout(self, spacing=0)
        layout.setContentsMargins(0,0,0,0)
        # 标题栏
        self.titleBar = TitleBar(self)
        layout.addWidget(self.titleBar)
        # 信号槽
        self.titleBar.clickedChinese.connect(self.clickedChinese)
        self.titleBar.clickedEnglish.connect(self.clickedEnglish)
        self.titleBar.windowMinimumed.connect(self.showMinimized)
        self.titleBar.windowMaximumed.connect(self.showMaximized)
        self.titleBar.windowNormaled.connect(self.showNormal)
        self.titleBar.windowClosed.connect(self.close)
        self.titleBar.windowMoved.connect(self.move)
        self.windowTitleChanged.connect(self.titleBar.setTitle)
        self.windowIconChanged.connect(self.titleBar.setIcon)

    #def setTitleBarHeight(self, height=38):
    def setTitleBarHeight(self, height=50):
        """设置标题栏高度"""
        self.titleBar.setHeight(height)

    def setIconSize(self, size):
        """设置图标的大小"""
        self.titleBar.setIconSize(size)

    def setWidget(self, widget):
        """设置自己的控件"""
        if hasattr(self, '_widget'):
            return
        self._widget = widget
        # 设置默认背景颜色,否则由于受到父窗口的影响导致透明
        #self._widget.setAutoFillBackground(True)
        self._widget.installEventFilter(self)
        self.layout().addWidget(self._widget)

    def move(self, pos):
        if self.windowState() == Qt.WindowMaximized or self.windowState() == Qt.WindowFullScreen:
            # 最大化或者全屏则不允许移动
            return
        super(FramelessWindow, self).move(pos)

    def clickedChinese(self):
        self.lang = "cn"
        self.translateAll()

    def clickedEnglish(self):
        self.lang = "en"
        self.translateAll()

    def translateAll(self):
        pass

    def showMaximized(self):
        """最大化,要去除上下左右边界,如果不去除则边框地方会有空隙"""
        super(FramelessWindow, self).showMaximized()
        self.layout().setContentsMargins(0, 0, 0, 0)

    def showNormal(self):
        """还原,要保留上下左右边界,否则没有边框无法调整"""
        super(FramelessWindow, self).showNormal()
        self.layout().setContentsMargins(0, 0, 0, 0)

    def eventFilter(self, obj, event):
        """事件过滤器,用于解决鼠标进入其它控件后还原为标准鼠标样式"""
        if isinstance(event, QEnterEvent):
            self.setCursor(Qt.ArrowCursor)
        return super(FramelessWindow, self).eventFilter(obj, event)

    def paintEvent(self, event):
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)

        pat = QPainter(self)
        pat.setRenderHint(pat.Antialiasing)
        pat.fillPath(path, QBrush(Qt.white))

        color = QColor(192, 192, 192, 50)

        for i in range(10):
            i_path = QPainterPath()
            i_path.setFillRule(Qt.WindingFill)
            ref = QRectF(10-i, 10-i, self.width()-(10-i)*2, self.height()-(10-i)*2)
            # i_path.addRect(ref)
            i_path.addRoundedRect(ref, self.border_width, self.border_width)
            color.setAlpha(150 - i**0.5*50)
            pat.setPen(color)
            pat.drawPath(i_path)

        # 圆角
        pat2 = QPainter(self)
        pat2.setRenderHint(pat2.Antialiasing)  # 抗锯齿
        pat2.setBrush(Qt.white)
        pat2.setPen(Qt.transparent)

        rect = self.rect()
        rect.setLeft(9)
        rect.setTop(9)
        rect.setWidth(rect.width()-9)
        rect.setHeight(rect.height()-9)
        pat2.drawRoundedRect(rect, 4, 4)


    def mousePressEvent(self, event):  ##事件开始
        if event.button() == QtCore.Qt.LeftButton:
            self.Move = True  ##设定bool为True
            self.Point = event.globalPos() - self.pos()  ##记录起始点坐标
            event.accept()

    def mouseMoveEvent(self, QMouseEvent):  ##移动时间
        if QtCore.Qt.LeftButton and self.Move:  ##切记这里的条件不能写死，只要判断move和鼠标执行即可！
            self.move(QMouseEvent.globalPos() - self.Point)  ##移动到鼠标到达的坐标点！
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):  ##结束事件 
        self.Move = False


class CircleProgressBar(QWidget):

    Color = QColor(24, 189, 155)  # 圆圈颜色
    Clockwise = True  # 顺时针还是逆时针
    Delta = 36

    def __init__(self, *args, color=None, clockwise=True, **kwargs):
        super(CircleProgressBar, self).__init__(*args, **kwargs)
        self.angle = 0
        self.Clockwise = clockwise
        if color:
            self.Color = color
        self._timer = QTimer(self, timeout=self.update)
        self._timer.start(100)

    def paintEvent(self, event):
        super(CircleProgressBar, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        side = min(self.width(), self.height())
        painter.scale(side / 100.0, side / 100.0)
        painter.rotate(self.angle)
        painter.save()
        painter.setPen(Qt.NoPen)
        color = self.Color.toRgb()
        for i in range(11):
            color.setAlphaF(1.0 * i / 10)
            painter.setBrush(color)
            painter.drawEllipse(30, -10, 20, 20)
            painter.rotate(36)
        painter.restore()
        self.angle += self.Delta if self.Clockwise else -self.Delta
        self.angle %= 360

    @pyqtProperty(QColor)
    def color(self) -> QColor:
        return self.Color

    @color.setter
    def color(self, color: QColor):
        if self.Color != color:
            self.Color = color
            self.update()

    @pyqtProperty(bool)
    def clockwise(self) -> bool:
        return self.Clockwise

    @clockwise.setter
    def clockwise(self, clockwise: bool):
        if self.Clockwise != clockwise:
            self.Clockwise = clockwise
            self.update()

    @pyqtProperty(int)
    def delta(self) -> int:
        return self.Delta

    @delta.setter
    def delta(self, delta: int):
        if self.delta != delta:
            self.delta = delta
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(100, 100)

