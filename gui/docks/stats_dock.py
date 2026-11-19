"""
统计信息面板
"""

import os
import numpy as np
import webbrowser
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from gui import resource_path


DOC_PATH = resource_path('docs/document_V1.1.html')


class StatsDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 9))
        self.text_edit.setMinimumHeight(100)
        layout.addWidget(self.text_edit)

        btn_row = QHBoxLayout()
        self.btn_copy = QPushButton("📋 复制统计信息")
        self.btn_copy.clicked.connect(self._copy_text)
        btn_row.addWidget(self.btn_copy)

        self.btn_doc = QPushButton("📄 说明文档")
        self.btn_doc.clicked.connect(self._open_doc)
        btn_row.addWidget(self.btn_doc)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def set_text(self, text: str):
        self.text_edit.setPlainText(text)

    def append_text(self, text: str):
        self.text_edit.append(text)

    def clear(self):
        self.text_edit.clear()

    def set_kk_text(self, result):
        lines = []
        lines.append("=" * 48)
        lines.append("  K-K 校验结果")
        lines.append("=" * 48)
        lines.append(f"  测试类型   : {result.test}")
        lines.append(f"  RC 数量    : {result.num_RC}")
        lines.append(f"  μ 值       : {result.mu:.4f}")
        lines.append(f"  伪卡方 χ²  : {result.pseudo_chisqr:.6e}")
        lines.append(f"  R₀ (Ω)     : {result.series_R:.6e}")
        if result.add_ind:
            lines.append(f"  L  (H)     : {result.series_L:.6e}")
        if result.add_cap:
            lines.append(f"  C⁻¹ (F⁻¹)  : {1.0/result.series_C:.6e}" if result.series_C != 0
                         else f"  C  (F)     : N/A")
        lines.append(f"  时间常数   : {result.time_constants[0]:.4e} ~ "
                     f"{result.time_constants[-1]:.4e} s")
        lines.append("-" * 48)
        lines.append(f"  残差统计 (归一化于 |Z|):")
        lines.append(f"  Δ'%  max = {100*np.max(np.abs(result.residuals_real)):.4f}")
        lines.append(f"  Δ''% max = {100*np.max(np.abs(result.residuals_imag)):.4f}")
        lines.append("=" * 48)
        self.set_text("\n".join(lines))

    def _copy_text(self):
        QApplication.clipboard().setText(self.text_edit.toPlainText())

    def _open_doc(self):
        if os.path.exists(DOC_PATH):
            webbrowser.open(DOC_PATH)
