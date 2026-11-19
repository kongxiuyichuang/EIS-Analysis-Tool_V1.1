"""
Tab 4: Simulation — 模型选择 + 元件编辑 + 参数 + 阻抗计算 + 可视化
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QGroupBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QDoubleSpinBox, QFileDialog, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

from gui.docks.model_select_dock import ModelSelectDock
from gui.docks.component_dock import ComponentDock
from gui.widgets.param_table import _sci_fmt
from core.models import create_model, TLMComponentNames

COLORS = {'line': '#1565c0', 'grid': '#e0e0e0'}


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 4), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='white', layout='constrained')
        super().__init__(self.fig)


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


class SimParamTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["参数名", "值", "单位"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def set_parameters(self, names, defaults, slot_map=None):
        if slot_map is None:
            slot_map = {}
        self.table.setRowCount(len(names))
        for i, name in enumerate(sorted(names)):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            val = defaults.get(name, 0.5)
            item_val = QTableWidgetItem(_sci_fmt(val))
            self.table.setItem(i, 1, item_val)
            slot = slot_map.get(name, '')
            unit_item = QTableWidgetItem(_get_unit(name, slot))
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 2, unit_item)

    def get_values(self):
        result = {}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            try:
                result[name] = float(self.table.item(row, 1).text())
            except (ValueError, TypeError):
                result[name] = 0.5
        return result


class SimulationTab(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._model = None
        self._setup_ui()
        self._connect_signals()

    def _apply_components(self):
        if self._model is None:
            return
        texts = self.comp_dock.get_all_texts()
        if 'custom' in texts:
            self._model.set_custom_expression(texts['custom'])
        else:
            for name in TLMComponentNames:
                text = texts.get(name, '')
                self._model.set_component(name, text)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.setHandleWidth(8)

        self.model_dock = ModelSelectDock()
        v_splitter.addWidget(self._make_group("模型选择", self.model_dock))

        self.comp_dock = ComponentDock()
        self.param_table = SimParamTable()
        cp_container = QWidget()
        cp_layout = QVBoxLayout(cp_container)
        cp_layout.setContentsMargins(0, 0, 0, 0)
        cp_splitter = QSplitter(Qt.Horizontal)
        cp_splitter.addWidget(self.comp_dock)
        cp_splitter.addWidget(self.param_table)
        cp_splitter.setStretchFactor(0, 1)
        cp_splitter.setStretchFactor(1, 3)
        cp_splitter.setHandleWidth(8)
        cp_layout.addWidget(cp_splitter)
        v_splitter.addWidget(self._make_group("元件编辑 + 参数设置", cp_container))

        freq_grp = QGroupBox("频率扫描")
        freq_layout = QHBoxLayout(freq_grp)
        freq_layout.addWidget(QLabel("f_min:"))
        self.f_min_edit = QLineEdit("1e-3")
        self.f_min_edit.setValidator(QDoubleValidator(1e-5, 1e6, 10))
        self.f_min_edit.setFixedWidth(100)
        freq_layout.addWidget(self.f_min_edit)
        freq_layout.addWidget(QLabel("f_max:"))
        self.f_max_edit = QLineEdit("1e5")
        self.f_max_edit.setValidator(QDoubleValidator(1e-5, 1e6, 10))
        self.f_max_edit.setFixedWidth(100)
        freq_layout.addWidget(self.f_max_edit)
        freq_layout.addWidget(QLabel("点/dec:"))
        self.ppd_spin = QDoubleSpinBox()
        self.ppd_spin.setRange(5, 200)
        self.ppd_spin.setValue(50)
        self.ppd_spin.setDecimals(0)
        freq_layout.addWidget(self.ppd_spin)
        v_splitter.addWidget(freq_grp)

        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        self.btn_calc = QPushButton("▶ 计算")
        self.btn_calc.setObjectName("btnFit")
        self.btn_export = QPushButton("💾 导出 CSV")
        self.btn_export.setObjectName("btnExport")
        btn_layout.addWidget(self.btn_calc)
        btn_layout.addWidget(self.btn_export)
        v_splitter.addWidget(btn_widget)

        v_splitter.setSizes([250, 500, 100, 40])

        left_layout.addWidget(v_splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_sim = PlotCanvas()
        toolbar = NavigationToolbar2QT(self.canvas_sim, right)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas_sim)

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setHandleWidth(8)

        layout.addWidget(splitter)

    def _make_group(self, title, widget):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)
        return box

    def _connect_signals(self):
        self.model_dock.model_changed.connect(self._on_model_changed)
        self.comp_dock.components_changed.connect(self._on_components_changed)
        self.btn_calc.clicked.connect(self._run_simulation)
        self.btn_export.clicked.connect(self._export_csv)
        self.comp_dock.btn_preset.clicked.disconnect()
        self.comp_dock.btn_preset.clicked.connect(self._apply_preset)

    def _apply_preset(self):
        config = self.model_dock.get_config()
        mode = config.get('mode', 'non_faradaic')
        self.comp_dock.load_preset(mode)
        self._sync_components()
        self._refresh_params()

    def _on_components_changed(self):
        self._sync_components()
        self._refresh_params()

    def _sync_components(self):
        if self._model is None:
            return
        self._apply_components()

    def _refresh_params(self):
        if self._model is None:
            return
        names = self._model.get_all_parameter_names()
        defaults = self._model.get_parameter_defaults()
        slot_map = self._model.get_parameter_slots()
        self.param_table.set_parameters(names, defaults, slot_map)

    def _on_model_changed(self):
        config = self.model_dock.get_config()
        self._model = create_model(config['main_type'], config['boundary'], config['zC_zero'])
        self.comp_dock.update_config(config['main_type'], config['boundary'], config['zC_zero'])
        if config['main_type'] != 'Custom':
            mode = config.get('mode', 'non_faradaic')
            self.comp_dock.load_preset(mode)
        self._sync_components()
        self._refresh_params()

    def _make_f_scan(self):
        f_min = float(self.f_min_edit.text())
        f_max = float(self.f_max_edit.text())
        ppd = int(self.ppd_spin.value())
        n_decades = np.log10(f_max / f_min)
        n_pts = int(n_decades * ppd) + 1
        return np.logspace(np.log10(f_min), np.log10(f_max), n_pts)

    def _run_simulation(self):
        if self._model is None:
            self._on_model_changed()
            if self._model is None:
                return
        self._apply_components()
        params = self.param_table.get_values()
        f_scan = self._make_f_scan()
        Z_sim = self._model.compute_Z(f_scan, params)
        self._draw_sim(f_scan, Z_sim)
        self._last_f = f_scan
        self._last_Z = Z_sim

    def _draw_sim(self, f, Z):
        self.canvas_sim.fig.clear()
        gs = self.canvas_sim.fig.add_gridspec(2, 2)

        ax1 = self.canvas_sim.fig.add_subplot(gs[0, 0])
        ax1.plot(np.real(Z), -np.imag(Z), '-o', color=COLORS['line'], lw=2)
        ax1.set_xlabel("Z' (Ω)"); ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.set_aspect('equal', 'datalim')

        ax2 = self.canvas_sim.fig.add_subplot(gs[0, 1])
        ax2.loglog(f, np.abs(Z), '-', color=COLORS['line'], lw=2)
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode |Z| vs f", fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')

        ax3 = self.canvas_sim.fig.add_subplot(gs[1, 0])
        phase = -np.arctan2(np.imag(Z), np.real(Z)) * 180.0 / np.pi
        ax3.semilogx(f, phase, '-', color=COLORS['line'], lw=2)
        ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Phase (°)")
        ax3.set_title("Phase vs f", fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ax4 = self.canvas_sim.fig.add_subplot(gs[1, 1])
        ln1 = ax4.semilogx(f, np.real(Z), '-', color='#1565c0', lw=2, label="Z'")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Z' (Ω)", color='#1565c0')
        ax4.tick_params(axis='y', labelcolor='#1565c0')
        ax4.grid(True, alpha=0.3)
        ax4_r = ax4.twinx()
        ln2 = ax4_r.semilogx(f, -np.imag(Z), '--', color='#c62828', lw=2, label="-Z''")
        ax4_r.set_ylabel("-Z'' (Ω)", color='#c62828')
        ax4_r.tick_params(axis='y', labelcolor='#c62828')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax4.legend(lns, labs, loc='best')
        ax4.set_title("Z' and -Z'' vs f", fontweight='bold')

        self.canvas_sim.draw()

    def _export_csv(self):
        if not hasattr(self, '_last_Z'):
            QMessageBox.warning(self, "无数据", "请先运行计算")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "保存模拟阻抗谱", "", "Excel (*.xlsx)")
        if not fp:
            return
        try:
            import pandas as pd
            from gui.widgets.param_table import _get_unit
            with pd.ExcelWriter(fp, engine='openpyxl') as writer:
                df_sim = pd.DataFrame({
                    "f(Hz)": self._last_f,
                    "Real(ohm)": np.real(self._last_Z),
                    "Imag(ohm)": np.imag(self._last_Z),
                })
                df_sim.to_excel(writer, sheet_name="simulation_data", index=False)

                config = self.model_dock.get_config()
                info_rows = [
                    ["模型类型", config['main_type']],
                    ["边界条件", config['boundary']],
                    ["模式", config.get('mode', '')],
                    ["zC = 0", str(config.get('zC_zero', False))],
                    ["", ""],
                    ["频率扫描", f"{float(self.f_min_edit.text()):.4e} ~ {float(self.f_max_edit.text()):.4e} Hz"],
                    ["点/dec", str(int(self.ppd_spin.value()))],
                    ["", ""],
                    ["支路", "元件表达式"],
                ]
                texts = self.comp_dock.get_all_texts()
                for name, txt in texts.items():
                    info_rows.append([name, txt])

                df_info = pd.DataFrame(info_rows[1:], columns=info_rows[0])
                df_info.to_excel(writer, sheet_name="param_value", index=False, startrow=0)

                slot_map = self._model.get_parameter_slots() if self._model else {}
                param_values = self.param_table.get_values()
                param_rows = []
                for name, val in param_values.items():
                    slot = slot_map.get(name, '')
                    param_rows.append([slot, name, val, _get_unit(name, slot)])
                df_params = pd.DataFrame(param_rows, columns=["Slot", "参数名", "值", "单位"])
                df_params.to_excel(writer, sheet_name="param_value", index=False, startrow=len(info_rows) + 2)
            QMessageBox.information(self, "导出成功", f"结果已保存至:\n{fp}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
