"""
FloatingProgress — 绘图区中央悬浮进度指示器

半透明悬浮层，内含任务文字 + 进度条，用于在同步计算期间
于绘图区中央显示忙态或真实迭代百分比。
"""

from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame,
)


class FloatingProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._host = None
        self._hide_timer = None
        self.setAttribute(Qt.WA_TranslucentBackground)

        frame = QFrame(self)
        frame.setObjectName("fpFrame")
        frame.setAttribute(Qt.WA_StyledBackground, True)

        f_layout = QVBoxLayout(frame)
        f_layout.setContentsMargins(16, 10, 16, 12)
        f_layout.setSpacing(8)

        self.label = QLabel("正在计算…")
        self.label.setObjectName("fpLabel")
        self.label.setAlignment(Qt.AlignCenter)
        f_layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setObjectName("fpBar")
        self.bar.setFixedHeight(14)
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        f_layout.addWidget(self.bar)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        self.hide()

    def attach_to(self, host):
        """挂到绘图区宿主，随宿主缩放重居中"""
        self.setParent(host)
        if self._host is not None and self._host is not host:
            self._host.removeEventFilter(self)
        self._host = host
        host.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._host and event.type() == QEvent.Resize:
            self._recenter()
        return super().eventFilter(obj, event)

    def _recenter(self):
        if self._host is None or not self.isVisible():
            return
        self.adjustSize()
        self.move((self._host.width() - self.width()) // 2,
                  (self._host.height() - self.height()) // 2)
        self.raise_()

    def show_busy(self, text):
        self.label.setText(text)
        self.bar.setRange(0, 0)
        self._show()

    def show_progress(self, text, maximum):
        self.label.setText(text)
        self.bar.setRange(0, max(1, int(maximum)))
        self.bar.setValue(0)
        self._show()

    def set_progress(self, value):
        self.bar.setValue(int(value))

    def finish(self, text="完成"):
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self.label.setText(text)
        if self._hide_timer is not None:
            self._hide_timer.stop()
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._hide_timer.start(400)

    def _show(self):
        self.show()
        self.raise_()
        self._recenter()
