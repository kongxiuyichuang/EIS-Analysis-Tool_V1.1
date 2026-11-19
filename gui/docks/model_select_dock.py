"""
模型选择面板 + 模型电路图显示
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QCheckBox, QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from core.models.base import MAIN_MODELS, BOUNDARY_CONDITIONS, ZC_ZERO_SUPPORT
from gui import resource_path

IMG_DIR = resource_path('piture')


def _get_image_path(main_type: str, boundary: str) -> str:
    if main_type == 'Custom':
        return os.path.join(IMG_DIR, 'Custom.png')
    if main_type == 'Universal-type':
        return os.path.join(IMG_DIR, 'universal-type.png')
    type_map = {'Z-type': 'Z type', 'T-type': 'T type', 'E-type': 'E type'}
    filename = f"{type_map[main_type]}_{boundary}.png"
    return os.path.join(IMG_DIR, filename)


class ModelSelectDock(QWidget):
    model_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        grp = QGroupBox("主模型")
        grp_layout = QVBoxLayout(grp)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("类型:"))
        self.main_type_combo = QComboBox()
        self.main_type_combo.addItems(MAIN_MODELS)
        type_row.addWidget(self.main_type_combo)
        type_row.addStretch()
        grp_layout.addLayout(type_row)

        bc_row = QHBoxLayout()
        self.bc_label = QLabel("边界条件:")
        bc_row.addWidget(self.bc_label)
        self.boundary_combo = QComboBox()
        bc_row.addWidget(self.boundary_combo)
        bc_row.addStretch()
        grp_layout.addLayout(bc_row)

        self.zC_zero_cb = QCheckBox("zC = 0 (简化)")
        self.zC_zero_cb.setEnabled(False)
        grp_layout.addWidget(self.zC_zero_cb)

        self.mode_label = QLabel("界面模式:")
        grp_layout.addWidget(self.mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["非法拉第模式 (Non-Faradaic)", "法拉第模式 (Faradaic)"])
        grp_layout.addWidget(self.mode_combo)

        left_layout.addWidget(grp)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: white; border: 1px solid #e0e0e0; border-radius: 4px;")
        self.image_label.setMinimumWidth(120)
        right_layout.addWidget(self.image_label)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setHandleWidth(6)

        layout.addWidget(splitter)

        self._update_boundary()
        self._update_image()

    def _connect_signals(self):
        self.main_type_combo.currentTextChanged.connect(self._on_type_changed)
        self.boundary_combo.currentTextChanged.connect(self._on_boundary_changed)
        self.zC_zero_cb.stateChanged.connect(self._on_zc_changed)
        self.mode_combo.currentIndexChanged.connect(self._emit_changed)

    def _on_type_changed(self, main_type):
        is_custom = (main_type == 'Custom')
        self.bc_label.setVisible(not is_custom)
        self.boundary_combo.setVisible(not is_custom)
        self.zC_zero_cb.setVisible(not is_custom)
        self.mode_label.setVisible(not is_custom)
        self.mode_combo.setVisible(not is_custom)
        if not is_custom:
            self._update_boundary()
            self._update_zc_enabled()
        self._update_image()
        self._emit_changed()

    def _on_boundary_changed(self, bc):
        self._update_zc_enabled()
        self._update_image()
        self._emit_changed()

    def _on_zc_changed(self):
        self._emit_changed()

    def _update_boundary(self):
        main_type = self.main_type_combo.currentText()
        boundaries = BOUNDARY_CONDITIONS.get(main_type, [])
        self.boundary_combo.clear()
        self.boundary_combo.addItems(boundaries)

    def _update_zc_enabled(self):
        main_type = self.main_type_combo.currentText()
        bc = self.boundary_combo.currentText() if self.boundary_combo.isVisible() else ''
        supported = ZC_ZERO_SUPPORT.get(main_type, [])
        self.zC_zero_cb.setEnabled(bc in supported)

    def _update_image(self):
        main_type = self.main_type_combo.currentText()
        boundary = self.boundary_combo.currentText() if main_type != 'Custom' else ''
        path = _get_image_path(main_type, boundary)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                400, 260,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.setText(f"图片未找到:\n{os.path.basename(path)}")

    def _emit_changed(self):
        self.model_changed.emit()

    def get_config(self) -> dict:
        return {
            'main_type': self.main_type_combo.currentText(),
            'boundary': self.boundary_combo.currentText() if self.boundary_combo.isVisible() else '',
            'zC_zero': self.zC_zero_cb.isChecked() if self.zC_zero_cb.isVisible() else False,
            'mode': 'faradaic' if self.mode_combo.isVisible() and self.mode_combo.currentIndex() == 1 else 'non_faradaic',
        }
