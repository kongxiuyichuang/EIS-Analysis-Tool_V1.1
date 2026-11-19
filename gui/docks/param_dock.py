"""
参数配置面板
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from gui.widgets.param_table import ParamTable


class ParamDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.param_table = ParamTable()
        layout.addWidget(self.param_table, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_fit = QPushButton("▶ 运行拟合")
        self.btn_fit.setObjectName("btnFit")
        btn_row.addWidget(self.btn_fit)

        self.btn_export = QPushButton("📊 导出结果")
        self.btn_export.setObjectName("btnExport")
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

    def update_from_model(self, model):
        names = model.get_all_parameter_names()
        defaults = model.get_parameter_defaults()
        slot_map = model.get_parameter_slots()
        self.param_table.set_parameters(names, defaults, slot_map)

    def get_fit_params_data(self) -> dict:
        return self.param_table.get_data()

    def update_fitted_values(self, param_dict: dict):
        self.param_table.update_values(param_dict)
