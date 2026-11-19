"""
组件字符串编辑器控件
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QToolTip
)
from PyQt5.QtCore import Qt, pyqtSignal
from core.component_parser import parse, ParseError, validate_types


class ComponentEditor(QWidget):
    text_changed = pyqtSignal()

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self._setup_ui()
        self._valid = True

    def _setup_ui(self):
        self.setStyleSheet("""
            ComponentEditor { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 4px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self.name_label = QLabel(self.label)
        self.name_label.setStyleSheet("font-weight: bold; color: #1a237e; font-size: 10pt;")
        header.addWidget(self.name_label)
        header.addStretch()
        self.status_label = QLabel("✓")
        self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g. R1-(R2W1)")
        self.input.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.input)

    def _on_text_changed(self):
        self._validate()
        if self._valid:
            self.text_changed.emit()

    def _validate(self):
        text = self.text().strip()
        if not text:
            self._valid = True
            self.input.setStyleSheet("border: 1px solid #e0e0e0;")
            self.status_label.setText("")
            return
        try:
            node = parse(text)
            validate_types(node)
            self._valid = True
            self.input.setStyleSheet("border: 1px solid #2e7d32;")
            self.status_label.setText("✓")
            self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        except ParseError as e:
            self._valid = False
            self.input.setStyleSheet("border: 1px solid #c62828;")
            self.status_label.setText("✗")
            self.status_label.setStyleSheet("color: #c62828; font-weight: bold;")
            self.input.setToolTip(str(e))

    def text(self) -> str:
        return self.input.text()

    def setText(self, text: str):
        self.input.setText(text)

    def is_valid(self) -> bool:
        if not self.text().strip():
            return True
        return self._valid
