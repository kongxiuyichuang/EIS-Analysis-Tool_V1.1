"""
K-K 校验控制面板
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class KKDock(QWidget):
    kk_started = pyqtSignal(int, float, bool, bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.btn_run = QPushButton("▶ 运行 K-K 校验")
        self.btn_run.setObjectName("btnKK")
        self.btn_run.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold; border: none;")
        self.btn_run.clicked.connect(self._on_run)
        layout.addWidget(self.btn_run)

        grp = QGroupBox("参数设置")
        grp_layout = QVBoxLayout(grp)

        # 优化算法复选框
        self.cb_optimized = QCheckBox("auto-optimized")
        self.cb_optimized.setChecked(True)
        self.cb_optimized.stateChanged.connect(self._on_optimized_toggle)
        grp_layout.addWidget(self.cb_optimized)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("最大 RC:"))
        self.max_RC_spin = QSpinBox()
        self.max_RC_spin.setRange(3, 200)
        self.max_RC_spin.setValue(50)
        row1.addWidget(self.max_RC_spin)
        row1.addStretch()
        grp_layout.addLayout(row1)

        self.mu_row = QHBoxLayout()
        self.mu_row.addWidget(QLabel("μ 阈值:"))
        self.mu_cutoff_spin = QDoubleSpinBox()
        self.mu_cutoff_spin.setRange(0.1, 0.99)
        self.mu_cutoff_spin.setSingleStep(0.05)
        self.mu_cutoff_spin.setValue(0.85)
        self.mu_row.addWidget(self.mu_cutoff_spin)
        self.mu_row.addStretch()
        grp_layout.addLayout(self.mu_row)

        # 电容/电感勾选
        self.cb_cap = QCheckBox("串联电容 C")
        self.cb_cap.setChecked(False)
        grp_layout.addWidget(self.cb_cap)

        self.cb_ind = QCheckBox("串联电感 L")
        self.cb_ind.setChecked(True)
        grp_layout.addWidget(self.cb_ind)

        layout.addWidget(grp)

        self.label_status = QLabel("")
        self.label_status.setStyleSheet("color: #616161; font-size: 9pt;")
        layout.addWidget(self.label_status)

        layout.addStretch()
        self._on_optimized_toggle()

    def _on_optimized_toggle(self):
        use_opt = self.cb_optimized.isChecked()
        # μ 阈值仅在非优化模式使用
        for i in range(self.mu_row.count()):
            w = self.mu_row.itemAt(i).widget()
            if w is not None:
                w.setVisible(not use_opt)

    def _on_run(self):
        max_RC = self.max_RC_spin.value()
        mu_cutoff = self.mu_cutoff_spin.value()
        add_cap = self.cb_cap.isChecked()
        add_ind = self.cb_ind.isChecked()
        use_optimized = self.cb_optimized.isChecked()
        self.label_status.setText("运行中...")
        self.kk_started.emit(max_RC, mu_cutoff, add_cap, add_ind, use_optimized)

    def set_result(self, pseudo_chisqr: float, num_RC: int, mu: float):
        self.label_status.setText(
            f"K-K 完成 | χ²={pseudo_chisqr:.4e} | RC={num_RC} | μ={mu:.3f}")
        self.label_status.setStyleSheet("color: #2e7d32; font-size: 9pt;")

    def set_error(self, msg: str):
        self.label_status.setText(f"错误: {msg}")
        self.label_status.setStyleSheet("color: #c62828; font-size: 9pt;")
