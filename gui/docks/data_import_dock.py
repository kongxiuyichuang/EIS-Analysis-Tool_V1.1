"""
数据导入面板
"""

import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class DataImportDock(QWidget):
    data_imported = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_data = None
        self.filtered_data = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.btn_import = QPushButton("📂 导入数据 (csv/txt/xlsx)")
        self.btn_import.clicked.connect(self._load_data)
        layout.addWidget(self.btn_import)

        self.info_label = QLabel("未导入数据")
        self.info_label.setStyleSheet("color: #616161; font-size: 9pt;")
        layout.addWidget(self.info_label)

        layout.addStretch()

    def _load_data(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择 EIS 数据文件",
            "",
            "支持的文件 (*.csv *.txt *.xlsx *.xls);;CSV (*.csv);;TXT (*.txt);;Excel (*.xlsx *.xls)"
        )
        if not fp:
            return

        try:
            ext = os.path.splitext(fp)[1].lower()
            if ext in ('.xlsx', '.xls'):
                df = pd.read_excel(fp)
            else:
                df = pd.read_csv(fp, sep=None, engine='python')

            data = df.iloc[:, :3].values.astype(float)
            data = data[np.argsort(data[:, 0])[::-1]]

            self.raw_data = data
            self.filtered_data = data.copy()

            n = len(data)
            f_min = np.min(data[:, 0])
            f_max = np.max(data[:, 0])
            self.info_label.setText(f"已加载: {os.path.basename(fp)}\n{n} 个数据点\n频率范围: {f_min:.2g} - {f_max:.2g} Hz")

            self.data_imported.emit(self.raw_data)

        except Exception as e:
            QMessageBox.warning(self, "导入错误", f"文件读取失败:\n{e}")
