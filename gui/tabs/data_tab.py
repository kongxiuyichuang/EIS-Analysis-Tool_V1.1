"""
Tab 1: Data — 数据导入 + 频率筛选 + K-K 校验 + 可视化 + 版权
"""

import os
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QLabel, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

from gui import resource_path
from gui.docks.data_import_dock import DataImportDock
from gui.docks.filter_dock import FilterDock
from gui.docks.kk_dock import KKDock
from gui.docks.stats_dock import StatsDock

COLORS = {
    'raw_marker': '#bdbdbd', 'sel_marker': '#2962ff',
    'kk_line': '#1565c0', 'res_re': '#2e7d32', 'res_im': '#f57f17',
}


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 4), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='white', layout='constrained')
        super().__init__(self.fig)


class DataTab(QWidget):
    data_loaded = pyqtSignal(object)
    filter_changed = pyqtSignal(object)
    kk_finished = pyqtSignal(object)

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.import_dock = DataImportDock()
        left_layout.addWidget(self._make_group("数据导入", self.import_dock))

        self.filter_dock = FilterDock()
        left_layout.addWidget(self._make_group("频率筛选", self.filter_dock))

        self.kk_dock = KKDock()
        left_layout.addWidget(self._make_group("K-K 校验", self.kk_dock))

        self.kk_stats = StatsDock()
        left_layout.addWidget(self._make_group("K-K 结果", self.kk_stats))

        left_layout.addWidget(self._build_copyright())
        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.plot_tabs = QTabWidget()
        self.plot_tabs.setTabPosition(QTabWidget.South)

        self.plot_raw = QWidget()
        self._setup_plot_tab(self.plot_raw, 'canvas_raw')
        self.plot_tabs.addTab(self.plot_raw, "原始数据")

        self.plot_kk = QWidget()
        self._setup_plot_tab(self.plot_kk, 'canvas_kk')
        self.plot_tabs.addTab(self.plot_kk, "K-K 校验")

        splitter.addWidget(scroll)
        splitter.addWidget(self.plot_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setHandleWidth(8)

        layout.addWidget(splitter)

    def _setup_plot_tab(self, tab, attr):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        canvas = PlotCanvas()
        toolbar = NavigationToolbar2QT(canvas, tab)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        setattr(self, attr, canvas)

    def _make_group(self, title, widget):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)
        return box

    def _build_copyright(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        title = QLabel("Create by : 锂电之路")
        title.setStyleSheet("color: #616161; font-size: 9pt; font-style: italic;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        qr_dir = resource_path('QRcode')
        qr_row = QHBoxLayout()
        for fname, caption in [('qrcode.png', 'buy me a coffee'), ('gzh_qrcode.png', '公众号')]:
            sub = QVBoxLayout()
            lbl = QLabel()
            path = os.path.join(qr_dir, fname)
            if os.path.exists(path):
                pix = QPixmap(path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)
            else:
                lbl.setText(f"[{caption}]")
            lbl.setAlignment(Qt.AlignCenter)
            sub.addWidget(lbl)
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("font-size: 7pt; color: #616161;")
            sub.addWidget(cap)
            qr_row.addLayout(sub)
        layout.addLayout(qr_row)
        return widget

    def _connect_signals(self):
        self.import_dock.data_imported.connect(self._on_data_imported)
        self.filter_dock.filter_changed.connect(self._on_filter_changed)
        self.kk_dock.kk_started.connect(self._on_kk_started)
        self.kk_dock.btn_run.clicked.connect(self._on_kk_run)

    def _on_data_imported(self, raw):
        self.filter_dock.set_range_from_data(raw)
        self._apply_filter()
        self.data_loaded.emit(raw)

    def _apply_filter(self):
        raw = self.import_dock.raw_data
        if raw is None:
            return
        f_min, f_max = self.filter_dock.get_freq_range()
        mask = (raw[:, 0] >= f_min) & (raw[:, 0] <= f_max)
        self.import_dock.filtered_data = raw[mask].copy() if np.sum(mask) >= 2 else raw.copy()
        if self.import_dock.filtered_data is not None:
            self.filter_changed.emit(self.import_dock.filtered_data)

    def _on_filter_changed(self):
        self._apply_filter()

    def _on_kk_run(self):
        pass

    def _on_kk_started(self, max_RC, mu_cutoff, add_cap, add_ind, use_optimized):
        from core.kramers_kronig import perform_kk_test
        from core.kk_optimized import perform_optimized_kk_test
        filtered = self.import_dock.filtered_data
        if filtered is None or len(filtered) < 3:
            self.kk_dock.set_error("数据不足")
            return
        try:
            f_arr = filtered[:, 0]
            z_exp = filtered[:, 1] + 1j * filtered[:, 2]
            if use_optimized:
                result = perform_optimized_kk_test(f_arr, z_exp, max_M=max_RC,
                    add_cap=add_cap, add_ind=add_ind, fit_type='complex')
            else:
                result = perform_kk_test(f_arr, z_exp, fit_type='complex',
                    add_cap=add_cap, add_ind=add_ind,
                    auto_mu=True, mu_cutoff=mu_cutoff, max_M=max_RC)
            self.kk_dock.set_result(result.pseudo_chisqr, result.num_RC, result.mu)
            self.kk_stats.set_kk_text(result)
            self._draw_kk(result)
            self.kk_finished.emit(result)
        except Exception as e:
            self.kk_dock.set_error(str(e))

    def _draw_raw(self, raw, filtered):
        if raw is None:
            return
        self.canvas_raw.fig.clear()
        f_r, re_r, im_r = raw.T
        f_f, re_f, im_f = filtered.T
        gs = self.canvas_raw.fig.add_gridspec(2, 2)
        ax1 = self.canvas_raw.fig.add_subplot(gs[0, 0])
        ax1.plot(re_r, -im_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax1.plot(re_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax1.set_xlabel("Z' (Ω)"); ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.legend(framealpha=0.8); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')
        ax2 = self.canvas_raw.fig.add_subplot(gs[0, 1])
        ax2.loglog(f_r, np.sqrt(re_r**2+im_r**2), 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax2.loglog(f_f, np.sqrt(re_f**2+im_f**2), 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8); ax2.grid(True, alpha=0.3, which='both')
        ax3 = self.canvas_raw.fig.add_subplot(gs[1, 0])
        ax3.semilogx(f_r, re_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax3.semilogx(f_f, re_f, 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Z' (Ω)")
        ax3.set_title("Real Impedance vs f", fontweight='bold')
        ax3.legend(framealpha=0.8); ax3.grid(True, alpha=0.3)
        ax4 = self.canvas_raw.fig.add_subplot(gs[1, 1])
        ax4.semilogx(f_r, -im_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax4.semilogx(f_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax4.set_xlabel("Frequency (Hz)"); ax4.set_ylabel("-Z'' (Ω)")
        ax4.set_title("-Imag Impedance vs f", fontweight='bold')
        ax4.legend(framealpha=0.8); ax4.grid(True, alpha=0.3)
        self.canvas_raw.draw()

    def _draw_kk(self, result):
        self.canvas_kk.fig.clear()
        f = result.frequencies
        Z_exp = result.impedances_exp
        Z_fit = result.impedances_fit
        res_re_pct = 100.0 * result.residuals_real
        res_im_pct = 100.0 * result.residuals_imag
        gs = self.canvas_kk.fig.add_gridspec(2, 2)
        ax1 = self.canvas_kk.fig.add_subplot(gs[0, 0])
        ax1.plot(np.real(Z_exp), -np.imag(Z_exp), 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax1.plot(np.real(Z_fit), -np.imag(Z_fit), '-', color=COLORS['kk_line'], lw=1.8, label='K-K Fit')
        ax1.set_xlabel("Z' (Ω)"); ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot (K-K)", fontweight='bold')
        ax1.legend(framealpha=0.8); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')
        ax2 = self.canvas_kk.fig.add_subplot(gs[0, 1])
        ax2.loglog(f, np.abs(Z_exp), 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax2.loglog(f, np.abs(Z_fit), '-', color=COLORS['kk_line'], lw=1.8, label='K-K Fit')
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8); ax2.grid(True, alpha=0.3, which='both')
        ax3 = self.canvas_kk.fig.add_subplot(gs[1, 0])
        ax3.semilogx(f, res_re_pct, 'o-', color=COLORS['res_re'], ms=5, lw=1.2, label='Real residual %')
        ax3.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("DZ'/|Z| (%)")
        ax3.set_title("Real Residuals %", fontweight='bold')
        ax3.legend(framealpha=0.8); ax3.grid(True, alpha=0.3)
        ax4 = self.canvas_kk.fig.add_subplot(gs[1, 1])
        ax4.semilogx(f, res_im_pct, 's-', color=COLORS['res_im'], ms=5, lw=1.2, label='Imag residual %')
        ax4.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax4.set_xlabel("Frequency (Hz)"); ax4.set_ylabel("DZ''/|Z| (%)")
        ax4.set_title("Imag Residuals %", fontweight='bold')
        ax4.legend(framealpha=0.8); ax4.grid(True, alpha=0.3)
        self.canvas_kk.draw()

    def update_plots(self, raw, filtered):
        self._draw_raw(raw, filtered)

    def get_filtered_data(self):
        return self.import_dock.filtered_data
