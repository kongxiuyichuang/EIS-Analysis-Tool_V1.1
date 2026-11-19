"""
元件定义面板
"""

import os
import webbrowser

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QDialog,
    QTextBrowser, QVBoxLayout as QVBoxLayoutDlg
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication
from gui import resource_path
from gui.widgets.component_editor import ComponentEditor
from core.models.base import PRESET_TEMPLATES


ZLIM_COMPONENTS = ['zA', 'zB', 'zC', 'zP', 'zQ']
UNIVERSAL_COMPONENTS = ['zA', 'zB', 'zC', 'ZV', 'ZW', 'ZX', 'ZY']
ALL_COMPONENT_NAMES = ['zA', 'zB', 'zC', 'zP', 'zQ', 'ZV', 'ZW', 'ZX', 'ZY']

DOC_DIR = resource_path('docs/The Circuit Description Code (CDC) explained.html')


class ComponentDock(QWidget):
    components_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editors = {}
        self._loading = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.btn_preset = QPushButton("📋 应用预设模板")
        self.btn_preset.clicked.connect(self._apply_preset)
        layout.addWidget(self.btn_preset)

        self.btn_help = QPushButton("📖 元件说明")
        self.btn_help.clicked.connect(self._show_help)
        layout.addWidget(self.btn_help)

        for name in ALL_COMPONENT_NAMES:
            editor = ComponentEditor(name)
            editor.text_changed.connect(self._on_editor_changed)
            self._editors[name] = editor
            layout.addWidget(editor)

        self.custom_editor = ComponentEditor('circuit')
        self.custom_editor.input.setPlaceholderText('e.g. R1-(R2Wo1C1)')
        self.custom_editor.text_changed.connect(self._on_editor_changed)
        self.custom_editor.setVisible(False)
        layout.addWidget(self.custom_editor)

        layout.addStretch()
        self.update_config('Z-type', 'P-Q', False)

    def _on_editor_changed(self):
        if not self._loading:
            self.components_changed.emit()

    def load_preset(self, mode: str):
        self._loading = True
        template = PRESET_TEMPLATES.get(mode, PRESET_TEMPLATES['non_faradaic'])
        for name, val in template.items():
            if name in self._editors:
                self._editors[name].setText(val)
        self._loading = False
        self.components_changed.emit()

    def _apply_preset(self):
        window = self.window()
        mode = 'non_faradaic'
        if hasattr(window, 'model_dock_widget'):
            config = window.model_dock_widget.get_config()
            mode = config.get('mode', 'non_faradaic')
        self.load_preset(mode)
        self.components_changed.emit()

    def _show_help(self):
        if os.path.exists(DOC_DIR):
            webbrowser.open(DOC_DIR)
        # dialog = QDialog(self.window())
        # dialog.setWindowTitle("元件符号说明")
        # dialog.resize(640, 520)
        # viewer = QTextBrowser(dialog)
        
        # if os.path.exists(DOC_DIR):
            
        #     with open(DOC_DIR, encoding='utf-8') as f:
        #         viewer.setHtml(f.read())
        # else:
        #     viewer.setPlainText(f"帮助文件未找到:\n{DOC_DIR}")
        # layout = QVBoxLayoutDlg(dialog)
        # layout.addWidget(viewer)
        # dialog.exec()

    def update_config(self, main_type: str, boundary: str, zC_zero: bool):
        """控制各编辑器显隐：主模型类型 + 边界条件 + zC=0"""
        is_custom = (main_type == 'Custom')
        self.btn_preset.setVisible(not is_custom)
        self.custom_editor.setVisible(is_custom)

        for name in ALL_COMPONENT_NAMES:
            self._editors[name].setVisible(not is_custom)

        if is_custom:
            QApplication.processEvents()
            return

        if main_type == 'Universal-type':
            self._editors['zP'].setVisible(False)
            self._editors['zQ'].setVisible(False)
        else:
            for name in ['ZV', 'ZW', 'ZX', 'ZY']:
                self._editors[name].setVisible(False)

            if zC_zero:
                self._editors['zC'].setVisible(False)

            if boundary == 'P-P symmetric':
                self._editors['zQ'].setVisible(False)
            elif boundary == 'open-Q':
                self._editors['zP'].setVisible(False)
            elif boundary == 'open-open':
                self._editors['zP'].setVisible(False)
                self._editors['zQ'].setVisible(False)

        QApplication.processEvents()

    def get_component_text(self, name: str) -> str:
        editor = self._editors.get(name)
        return editor.text().strip() if editor else ""

    def set_component_text(self, name: str, text: str):
        editor = self._editors.get(name)
        if editor:
            editor.setText(text)

    def get_all_texts(self) -> dict:
        result = {}
        for name, ed in self._editors.items():
            if ed.isVisible():
                result[name] = ed.text().strip()
        if self.custom_editor.isVisible():
            result['custom'] = self.custom_editor.text().strip()
        return result
