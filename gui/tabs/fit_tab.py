"""
Tab 2: Fit — 电路拟合 + 拟合结果可视化 + 统计
"""

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QGroupBox, QMessageBox, QFileDialog, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal

from gui.docks.model_select_dock import ModelSelectDock
from gui.docks.component_dock import ComponentDock
from gui.docks.param_dock import ParamDock
from gui.docks.stats_dock import StatsDock
from gui.widgets.floating_progress import FloatingProgress
from core.models import create_model, TLMComponentNames
from core.fitting_engine import FittingEngine, FitParams

COLORS = {
    'raw_marker': '#bdbdbd', 'sel_marker': '#2962ff',
    'fit_line': '#c62828', 'res_re': '#2e7d32', 'res_im': '#f57f17',
}


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 4), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='white', layout='constrained')
        super().__init__(self.fig)


class FitTab(QWidget):
    model_changed = pyqtSignal(object)
    fitting_started = pyqtSignal()

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._model = None
        self._fit_result = None
        self._setup_ui()
        self._connect_signals()
        self._init_state()

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
        self.param_dock = ParamDock()
        cp_container = QWidget()
        cp_layout = QVBoxLayout(cp_container)
        cp_layout.setContentsMargins(0, 0, 0, 0)
        cp_splitter = QSplitter(Qt.Horizontal)
        cp_splitter.addWidget(self.comp_dock)
        cp_splitter.addWidget(self.param_dock)
        cp_splitter.setStretchFactor(0, 1)
        cp_splitter.setStretchFactor(1, 5)
        cp_splitter.setHandleWidth(8)
        cp_layout.addWidget(cp_splitter)
        v_splitter.addWidget(self._make_group("元件定义 + 参数配置", cp_container))

        self.stats_dock = StatsDock()
        v_splitter.addWidget(self._make_group("统计信息", self.stats_dock))

        v_splitter.setSizes([250, 500, 200])

        left_layout.addWidget(v_splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_fit = PlotCanvas()
        toolbar = NavigationToolbar2QT(self.canvas_fit, right)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas_fit)
        self._draw_welcome()

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 6)
        splitter.setHandleWidth(8)

        layout.addWidget(splitter)

        self.progress = FloatingProgress()
        self.progress.attach_to(right)

    def _make_group(self, title, widget):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)
        return box

    def _connect_signals(self):
        self.model_dock.model_changed.connect(self._on_model_changed)
        self.comp_dock.components_changed.connect(self._on_components_changed)
        self.param_dock.btn_fit.clicked.connect(self._run_fit)
        self.param_dock.btn_export.clicked.connect(self._export_data)
        self.comp_dock.btn_preset.clicked.disconnect()
        self.comp_dock.btn_preset.clicked.connect(self._apply_preset)

    def _apply_preset(self):
        config = self.model_dock.get_config()
        mode = config.get('mode', 'non_faradaic')
        self.comp_dock.load_preset(mode)
        self.comp_dock.components_changed.emit()

    def _init_state(self):
        self._on_model_changed()

    def _on_model_changed(self):
        config = self.model_dock.get_config()
        mode = config['mode']
        self._model = create_model(config['main_type'], config['boundary'], config['zC_zero'])
        self.comp_dock.update_config(config['main_type'], config['boundary'], config['zC_zero'])
        if config['main_type'] != 'Custom':
            self.comp_dock.load_preset(mode)
        self._apply_components()
        self._update_params()
        self.model_changed.emit(self._model)

    def _on_components_changed(self):
        self._apply_components()
        self._update_params()

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

    def _update_params(self):
        if self._model is None:
            return
        self.param_dock.update_from_model(self._model)

    def _run_fit(self):
        if self.main_window is None:
            return
        filtered = self.main_window.filtered_data
        if filtered is None or len(filtered) < 2:
            QMessageBox.warning(self, "数据不足", "请先在 Data 标签页导入数据")
            return

        if self._model is None:
            self._on_model_changed()

        self._apply_components()
        f_arr = filtered[:, 0]
        z_exp = filtered[:, 1] + 1j * filtered[:, 2]

        param_data = self.param_dock.get_fit_params_data()
        fit_params = FitParams(
            names=param_data['names'],
            initial=param_data['initial'],
            lower=param_data['lower'],
            upper=param_data['upper'],
            fixed=param_data['fixed'],
        )

        self.stats_dock.clear()
        self.stats_dock.append_text("正在拟合，请稍候...\n")
        QApplication.processEvents()

        if fit_params.free_count == 0:
            QMessageBox.warning(self, "参数错误", "请至少保留一个参数不被固定")
            return

        engine = FittingEngine(self._model, f_arr, z_exp)
        self.param_dock.btn_fit.setEnabled(False)
        self.param_dock.btn_export.setEnabled(False)
        self.progress.show_busy("CNLS 拟合中…")
        QApplication.processEvents()
        try:
            result = engine.run(fit_params)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress.finish("失败")
            self.param_dock.btn_fit.setEnabled(True)
            self.param_dock.btn_export.setEnabled(True)
            QMessageBox.critical(self, "拟合失败", str(e))
            return
        self.progress.finish()
        self.param_dock.btn_fit.setEnabled(True)
        self.param_dock.btn_export.setEnabled(True)

        if not result.success:
            self.stats_dock.set_text(f"拟合失败:\n{result.message}")
            return

        self._fit_result = result
        self.param_dock.update_fitted_values(result.params)

        lines = ["=" * 48,
                 f"  收敛状态 : {result.message}",
                 f"  代价函数 : {result.stats['cost']:.4e}",
                 f"  卡方 chi2 = {result.stats['chi_sq']:.6e}",
                 f"  约化 chi2 = {result.stats['chi_red']:.6e}",
                 f"  R2       = {result.stats['r2']:.6f}",
                 f"  自由度   = {result.stats['dof']}",
                 f"  迭代次数 = {result.stats['nfev']}",
                 "-" * 48,
                 "  拟合参数值:",
                 "-" * 48]
        param_data = self.param_dock.get_fit_params_data()
        for name, val in result.params.items():
            idx = param_data['names'].index(name) if name in param_data['names'] else -1
            mark = " (固定)" if idx >= 0 and param_data['fixed'][idx] else ""
            lines.append(f"  {name:<22} = {val:.6e}{mark}")
        lines.append("=" * 48)
        self.stats_dock.set_text("\n".join(lines))

        self._draw_fit(filtered, result)
        self.fitting_started.emit()

    def _draw_fit(self, filtered, result):
        self.canvas_fit.fig.clear()
        f_f, re_f, im_f = filtered.T
        re_fit, im_fit = np.real(result.z_fit), np.imag(result.z_fit)
        gs = self.canvas_fit.fig.add_gridspec(2, 3)
        ax1 = self.canvas_fit.fig.add_subplot(gs[0, 0])
        ax1.plot(re_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax1.plot(re_fit, -im_fit, '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax1.set_xlabel("Z' (Ω)"); ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.legend(framealpha=0.8); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')
        ax2 = self.canvas_fit.fig.add_subplot(gs[0, 1])
        ax2.loglog(f_f, np.sqrt(re_f**2+im_f**2), 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax2.loglog(f_f, np.sqrt(re_fit**2+im_fit**2), '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8); ax2.grid(True, alpha=0.3, which='both')
        ax3 = self.canvas_fit.fig.add_subplot(gs[0, 2])
        res_re_pct = 100.0 * result.res_re / np.maximum(np.abs(re_f), 1e-12)
        ax3.semilogx(f_f, res_re_pct, 'o-', color=COLORS['res_re'], ms=5, lw=1.2, label='Real residual %')
        ax3.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("DZ'/Z' (%)")
        ax3.set_title("Real Residuals %", fontweight='bold')
        ax3.legend(framealpha=0.8); ax3.grid(True, alpha=0.3)
        ax4 = self.canvas_fit.fig.add_subplot(gs[1, 0])
        ax4.semilogx(f_f, re_f, 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax4.semilogx(f_f, re_fit, '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax4.set_xlabel("Frequency (Hz)"); ax4.set_ylabel("Z' (Ω)")
        ax4.set_title("Real Impedance vs f", fontweight='bold')
        ax4.legend(framealpha=0.8); ax4.grid(True, alpha=0.3)
        ax5 = self.canvas_fit.fig.add_subplot(gs[1, 1])
        ax5.semilogx(f_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax5.semilogx(f_f, -im_fit, '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax5.set_xlabel("Frequency (Hz)"); ax5.set_ylabel("-Z'' (Ω)")
        ax5.set_title("-Imag Impedance vs f", fontweight='bold')
        ax5.legend(framealpha=0.8); ax5.grid(True, alpha=0.3)
        ax6 = self.canvas_fit.fig.add_subplot(gs[1, 2])
        res_im_pct = 100.0 * result.res_im / np.maximum(np.abs(im_f), 1e-12)
        ax6.semilogx(f_f, res_im_pct, 's-', color=COLORS['res_im'], ms=5, lw=1.2, label='Imag residual %')
        ax6.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax6.set_xlabel("Frequency (Hz)"); ax6.set_ylabel("DZ''/Z'' (%)")
        ax6.set_title("Imag Residuals %", fontweight='bold')
        ax6.legend(framealpha=0.8); ax6.grid(True, alpha=0.3)
        self.canvas_fit.draw()

    def _draw_welcome(self):
        self.canvas_fit.fig.clear()
        ax = self.canvas_fit.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Run fit to see results", transform=ax.transAxes,
                ha='center', va='center', fontsize=18, color='#9e9e9e')
        ax.axis('off')
        self.canvas_fit.draw()

    def _export_data(self):
        if self._fit_result is None:
            QMessageBox.warning(self, "无结果", "请先运行拟合")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "保存拟合结果", "", "Excel (*.xlsx)")
        if not fp:
            return
        try:
            from gui.widgets.param_table import _get_unit
            with pd.ExcelWriter(fp, engine='openpyxl') as writer:
                rows = []
                config = self.model_dock.get_config()
                rows.append(["模型", config['main_type'], f"边界: {config['boundary']}"])
                rows.append(["", "", ""])
                for name, val in self._fit_result.params.items():
                    rows.append([name, f"{val:.6e}", ""])
                rows.append(["", "", ""])
                rows.append(["代价函数 Cost", f"{self._fit_result.stats['cost']:.4e}", ""])
                rows.append(["卡方 chi2", f"{self._fit_result.stats['chi_sq']:.6e}", ""])
                rows.append(["约化 chi2", f"{self._fit_result.stats['chi_red']:.6e}", ""])
                rows.append(["R2", f"{self._fit_result.stats['r2']:.6f}", ""])
                rows.append(["自由度", f"{self._fit_result.stats['dof']}", ""])
                rows.append(["迭代次数", f"{self._fit_result.stats['nfev']}", ""])
                rows.append(["收敛状态", self._fit_result.stats['status'], ""])
                df_stats = pd.DataFrame(rows, columns=["项目", "值", "备注"])
                df_stats.to_excel(writer, sheet_name="Fit_Statistics", index=False)

                f_data = self.main_window.filtered_data[:, 0]
                df_imp = pd.DataFrame({
                    "f (Hz)": f_data,
                    "Real_exp": self.main_window.filtered_data[:, 1],
                    "Imag_exp": self.main_window.filtered_data[:, 2],
                    "Real_fit": np.real(self._fit_result.z_fit),
                    "Imag_fit": np.imag(self._fit_result.z_fit),
                })
                df_imp.to_excel(writer, sheet_name="Impedance_Data", index=False)

                param_data = self.param_dock.get_fit_params_data()
                slot_map = self._model.get_parameter_slots() if self._model else {}
                param_rows = []
                for i, name in enumerate(param_data['names']):
                    slot = slot_map.get(name, '')
                    fitted = self._fit_result.params.get(name, param_data['initial'][i])
                    param_rows.append([
                        slot, name,
                        param_data['initial'][i], fitted,
                        param_data['lower'][i], param_data['upper'][i],
                        "是" if param_data['fixed'][i] else "否",
                        _get_unit(name, slot),
                    ])
                df_params = pd.DataFrame(param_rows, columns=[
                    "Slot", "参数名", "初始值", "拟合值", "下界", "上界", "固定", "单位"])
                df_params.to_excel(writer, sheet_name="fit_params", index=False)
            QMessageBox.information(self, "导出成功", f"结果已保存至:\n{fp}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"{e}")
