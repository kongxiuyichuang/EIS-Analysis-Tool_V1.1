"""
频率筛选面板
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class FilterDock(QWidget):
    filter_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        f_min_row = QHBoxLayout()
        f_min_row.addWidget(QLabel("f_min:"))
        self.f_min_spin = QDoubleSpinBox()
        self.f_min_spin.setDecimals(1)
        self.f_min_spin.setRange(-10, 10)
        self.f_min_spin.setValue(-3)
        self.f_min_spin.setSuffix(" log10(Hz)")
        f_min_row.addWidget(self.f_min_spin)
        layout.addLayout(f_min_row)

        f_max_row = QHBoxLayout()
        f_max_row.addWidget(QLabel("f_max:"))
        self.f_max_spin = QDoubleSpinBox()
        self.f_max_spin.setDecimals(1)
        self.f_max_spin.setRange(-10, 10)
        self.f_max_spin.setValue(6)
        self.f_max_spin.setSuffix(" log10(Hz)")
        f_max_row.addWidget(self.f_max_spin)
        layout.addLayout(f_max_row)

        self.f_min_spin.valueChanged.connect(self._emit)
        self.f_max_spin.valueChanged.connect(self._emit)

    def _emit(self):
        if self.f_min_spin.value() > self.f_max_spin.value():
            self.f_max_spin.setValue(self.f_min_spin.value())
        self.filter_changed.emit()

    def set_range_from_data(self, data):
        if data is None or len(data) == 0:
            return
        f_min_log = np.log10(np.min(data[:, 0]))
        f_max_log = np.log10(np.max(data[:, 0]))
        pad = (f_max_log - f_min_log) * 0.05
        self.f_min_spin.setRange(f_min_log - pad, f_max_log + pad)
        self.f_max_spin.setRange(f_min_log - pad, f_max_log + pad)
        self.f_min_spin.setValue(f_min_log - pad)
        self.f_max_spin.setValue(f_max_log + pad)

    def get_freq_range(self):
        return 10**self.f_min_spin.value(), 10**self.f_max_spin.value()
