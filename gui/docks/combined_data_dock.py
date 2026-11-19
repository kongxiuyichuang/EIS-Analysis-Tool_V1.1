"""
数据导入 + 频率筛选 合并面板（左右排列 1:1）
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PyQt5.QtCore import Qt
from gui.docks.data_import_dock import DataImportDock
from gui.docks.filter_dock import FilterDock


class CombinedDataDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        self.data_dock_widget = DataImportDock()
        self.filter_dock_widget = FilterDock()

        splitter.addWidget(self.data_dock_widget)
        splitter.addWidget(self.filter_dock_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(6)

        layout.addWidget(splitter)
