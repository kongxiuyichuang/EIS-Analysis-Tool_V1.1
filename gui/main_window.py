"""
主窗口：QMainWindow + QTabWidget 4 标签页布局
"""

import os
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QMessageBox
from PyQt5.QtCore import Qt

from gui import resource_path
from gui.tabs.data_tab import DataTab
from gui.tabs.fit_tab import FitTab
from gui.tabs.drt_dct_tab import DRTDCTTab
from gui.tabs.simulation_tab import SimulationTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIS Analysis Tool V1.1 ")
        self.resize(2560, 1500)
        style_path = resource_path('resources/styles.qss')
        if os.path.exists(style_path):
            with open(style_path, encoding='utf-8') as f:
                self.setStyleSheet(f.read())

        self.raw_data = None
        self.filtered_data = None
        self._kk_result = None

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)

        self.tab_data = DataTab(self)
        self.tab_fit = FitTab(self)
        self.tab_drt = DRTDCTTab(self)
        self.tab_sim = SimulationTab(self)

        tabs.addTab(self.tab_data, "① 数据 (Data)")
        tabs.addTab(self.tab_fit, "② 拟合 (Fit)")
        tabs.addTab(self.tab_drt, "③ 时间/电容/相位/扩散分布 (DRT-DCT-DOP-DDT)")
        tabs.addTab(self.tab_sim, "④ 仿真 (Simulation)")

        self.setCentralWidget(tabs)

        self._connect_signals()

    def _connect_signals(self):
        self.tab_data.data_loaded.connect(self._on_data_loaded)
        self.tab_data.filter_changed.connect(self._on_filter_changed)

    def _on_data_loaded(self, raw):
        self.raw_data = raw
        self.filtered_data = raw.copy()
        self.tab_data.update_plots(raw, self.filtered_data)
        self.tab_drt.update_data_info(
            np.min(raw[:, 0]), np.max(raw[:, 0]), len(raw))

    def _on_filter_changed(self, filtered):
        self.filtered_data = filtered
        self.tab_data.update_plots(self.raw_data, filtered)
        if filtered is not None:
            self.tab_drt.update_data_info(
                np.min(filtered[:, 0]), np.max(filtered[:, 0]), len(filtered))
