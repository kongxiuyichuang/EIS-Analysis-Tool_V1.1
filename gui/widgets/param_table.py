"""
动态参数表格控件
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt
import math


HEADERS = ['参数名', '下界', '初始值', '上界', '固定', '单位']


def _get_unit(param_name: str, slot: str = '') -> str:
    if param_name == 'R0':
        return 'Ω'
    if param_name == 'L':
        return 'cm'
    if param_name.endswith('_p'):
        return '1'
    if param_name.endswith('_L'):
        return 'H'
    if param_name.endswith('_tau'):
        return 's'
    if slot in ('zA', 'zC'):
        if param_name.endswith('_Aw'):
            return '(Ω·s^(1/2))/cm'
        if param_name.endswith('_R') or param_name.endswith('_Rd'):
            return 'Ω/cm'
        if param_name.endswith('_C'):
            return '(F·s^(p-1))/cm'
    elif slot == 'zB':
        if param_name.endswith('_Aw'):
            return 'Ω·cm·s^(1/2)'
        if param_name.endswith('_R') or param_name.endswith('_Rd'):
            return 'Ω·cm'
        if param_name.endswith('_C'):
            return '(F·s^(p-1))/cm'
    else:
        if param_name.endswith('_Aw'):
            return 'Ω·s^(1/2)'
        if param_name.endswith('_R') or param_name.endswith('_Rd'):
            return 'Ω'
        if param_name.endswith('_C'):
            return 'F·s^(p-1)'
    if param_name.endswith('_dC'):
        return 'F'
    if param_name.endswith('_a') or param_name.endswith('_b') or param_name.endswith('_n'):
        return ''
    if param_name.endswith('_Y'):
        return 'S·s^n'
    if param_name.endswith('_k'):
        return 's⁻¹'
    return ''


def _sci_fmt(v: float) -> str:
    if not math.isfinite(v):
        return str(v)
    s = f"{v:.3e}"
    mantissa, exp = s.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    exp = exp.lstrip('+')
    # Strip leading zeros from exponent (but keep the sign)
    sign = ''
    if exp.startswith('-'):
        sign = '-'
        exp = exp[1:]
    exp = exp.lstrip('0') or '0'
    return f"{mantissa}e{sign}{exp}"


class ParamTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
            header.resizeSection(i, 90)
        header.setStretchLastSection(False)

        layout.addWidget(self.table)

    def set_parameters(self, param_names: list, defaults: dict, slot_map: dict = None):
        self.table.setRowCount(0)
        self.param_names = param_names
        self.defaults = defaults
        self._slot_map = slot_map or {}

        for name in param_names:
            row = self.table.rowCount()
            self.table.insertRow(row)

            default_val = defaults.get(name, 1.0)

            item_name = QTableWidgetItem(name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_name)

            if name == 'R0':
                lb, init, ub = 1e-4, default_val, 10.0
            elif name == 'L':
                lb, init, ub = 1e-6, default_val, 1.0
            elif name.endswith('_R'):
                lb, init, ub = 1e-3, default_val, 1e6
            elif name.endswith('_C'):
                lb, init, ub = 1e-9, default_val, 1.0
            elif name.endswith('_p'):
                base = name[:-2]
                if base.rstrip('0123456789') in ('Wo', 'Ws'):
                    lb, init, ub = 0, default_val, 0.5
                else:
                    lb, init, ub = 0.5, default_val, 1.0
            elif name.endswith('_L'):
                lb, init, ub = 1e-9, default_val, 1.0
            elif name.endswith('_Rd'):
                lb, init, ub = 1e-6, default_val, 1e6
            elif name.endswith('_tau'):
                lb, init, ub = 1, default_val, 5000.0
            elif name.endswith('_dC'):
                lb, init, ub = 1e-12, default_val, 1.0
            elif name.endswith('_Y'):
                lb, init, ub = 1e-12, default_val, 1e6
            elif name.endswith('_k'):
                lb, init, ub = 1e-12, default_val, 1e6
            elif name.endswith('_a') or name.endswith('_b') or name.endswith('_n'):
                lb, init, ub = 0.0, default_val, 1.0
            elif name.endswith('_Aw'):
                lb, init, ub = 1e-6, default_val, 1e6
            else:
                lb, init, ub = 1e-6, default_val, 1e6

            self.table.setItem(row, 1, QTableWidgetItem(_sci_fmt(lb)))
            self.table.setItem(row, 2, QTableWidgetItem(_sci_fmt(init)))
            self.table.setItem(row, 3, QTableWidgetItem(_sci_fmt(ub)))

            fixed_item = QTableWidgetItem("")
            fixed_item.setFlags(fixed_item.flags() | Qt.ItemIsUserCheckable)
            fixed_item.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 4, fixed_item)

            unit_item = QTableWidgetItem(_get_unit(name, self._slot_map.get(name, '')))
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 5, unit_item)

    def get_data(self) -> dict:
        data = {'names': [], 'initial': [], 'lower': [], 'upper': [], 'fixed': []}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            try:
                lb = float(self.table.item(row, 1).text())
                iv = float(self.table.item(row, 2).text())
                ub = float(self.table.item(row, 3).text())
            except (ValueError, TypeError):
                continue
            fixed = self.table.item(row, 4).checkState() == Qt.Checked
            data['names'].append(name)
            data['lower'].append(lb)
            data['initial'].append(iv)
            data['upper'].append(ub)
            data['fixed'].append(fixed)
        return data

    def update_values(self, param_dict: dict):
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            if name in param_dict:
                self.table.item(row, 2).setText(_sci_fmt(param_dict[name]))
