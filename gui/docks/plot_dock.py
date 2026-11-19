"""
绘图面板：支持多标签页 (原始数据 / 拟合结果 / K-K 校验)
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget
)
from PyQt5.QtCore import Qt

COLORS = {
    'raw_marker': '#bdbdbd',
    'sel_marker': '#2962ff',
    'fit_line': '#c62828',
    'res_re': '#2e7d32',
    'res_im': '#f57f17',
    'primary': '#1a237e',
    'kk_line': '#1565c0',
    'kk_marker': '#0d47a1',
}


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 4), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='white', layout='constrained')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: white;")


class PlotDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.South)

        self.tab_raw = QWidget()
        self._setup_raw_tab()
        self.tabs.addTab(self.tab_raw, "原始数据")

        self.tab_kk = QWidget()
        self._setup_kk_tab()
        self.tabs.addTab(self.tab_kk, "K-K 校验")

        self.tab_fit = QWidget()
        self._setup_fit_tab()
        self.tabs.addTab(self.tab_fit, "拟合结果")

        layout.addWidget(self.tabs)

    def _setup_raw_tab(self):
        layout = QVBoxLayout(self.tab_raw)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_raw = PlotCanvas()
        toolbar = NavigationToolbar2QT(self.canvas_raw, self.tab_raw)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas_raw)

    def _setup_fit_tab(self):
        layout = QVBoxLayout(self.tab_fit)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_fit = PlotCanvas()
        toolbar = NavigationToolbar2QT(self.canvas_fit, self.tab_fit)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas_fit)

    def _setup_kk_tab(self):
        layout = QVBoxLayout(self.tab_kk)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_kk = PlotCanvas()
        toolbar = NavigationToolbar2QT(self.canvas_kk, self.tab_kk)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas_kk)

    def draw_welcome(self):
        for canvas in [self.canvas_raw, self.canvas_fit, self.canvas_kk]:
            canvas.fig.clear()
            ax = canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Import data to begin", transform=ax.transAxes,
                    ha='center', va='center', fontsize=18, color='#9e9e9e')
            ax.axis('off')
            canvas.draw()

    def draw_experimental(self, raw_data, filtered_data):
        if raw_data is None:
            return

        self.canvas_raw.fig.clear()
        f_r, re_r, im_r = raw_data.T
        f_f, re_f, im_f = filtered_data.T

        gs = self.canvas_raw.fig.add_gridspec(2, 2)

        ax1 = self.canvas_raw.fig.add_subplot(gs[0, 0])
        ax1.plot(re_r, -im_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax1.plot(re_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax1.set_xlabel("Z' (Ω)")
        ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.legend(framealpha=0.8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')

        ax2 = self.canvas_raw.fig.add_subplot(gs[0, 1])
        ax2.loglog(f_r, np.sqrt(re_r**2 + im_r**2), 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax2.loglog(f_f, np.sqrt(re_f**2 + im_f**2), 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8)
        ax2.grid(True, alpha=0.3, which='both')

        ax3 = self.canvas_raw.fig.add_subplot(gs[1, 0])
        ax3.semilogx(f_r, re_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax3.semilogx(f_f, re_f, 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Z' (Ω)")
        ax3.set_title("Real Impedance vs f", fontweight='bold')
        ax3.legend(framealpha=0.8)
        ax3.grid(True, alpha=0.3)

        ax4 = self.canvas_raw.fig.add_subplot(gs[1, 1])
        ax4.semilogx(f_r, -im_r, 'o', color=COLORS['raw_marker'], ms=4, alpha=0.5, label='Raw')
        ax4.semilogx(f_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Selected')
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("-Z'' (Ω)")
        ax4.set_title("-Imag Impedance vs f", fontweight='bold')
        ax4.legend(framealpha=0.8)
        ax4.grid(True, alpha=0.3)

        self.canvas_raw.draw()

    def draw_fit_results(self, raw_data, filtered_data, z_fit, res_re, res_im):
        if filtered_data is None or z_fit is None:
            return

        self.canvas_fit.fig.clear()
        f_r, re_r, im_r = raw_data.T
        f_f, re_f, im_f = filtered_data.T
        re_fit, im_fit = np.real(z_fit), np.imag(z_fit)

        gs = self.canvas_fit.fig.add_gridspec(2, 3)

        ax1 = self.canvas_fit.fig.add_subplot(gs[0, 0])
        ax1.plot(re_r, -im_r, 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45, label='Raw')
        ax1.plot(re_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax1.plot(re_fit, -im_fit, '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax1.set_xlabel("Z' (Ω)"); ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot", fontweight='bold')
        ax1.legend(framealpha=0.8); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')

        ax2 = self.canvas_fit.fig.add_subplot(gs[0, 1])
        ax2.loglog(f_r, np.sqrt(re_r**2+im_r**2), 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45, label='Raw')
        ax2.loglog(f_f, np.sqrt(re_f**2+im_f**2), 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax2.loglog(f_f, np.sqrt(re_fit**2+im_fit**2), '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8); ax2.grid(True, alpha=0.3, which='both')

        ax3 = self.canvas_fit.fig.add_subplot(gs[0, 2])
        res_re_pct = 100.0 * res_re / np.maximum(np.abs(re_f), 1e-12)
        ax3.semilogx(f_f, res_re_pct, 'o-', color=COLORS['res_re'], ms=5, lw=1.2, label='Real residual %')
        ax3.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("ΔZ'/Z' (%)")
        ax3.set_title("Real Residuals %", fontweight='bold')
        ax3.legend(framealpha=0.8); ax3.grid(True, alpha=0.3)

        ax4 = self.canvas_fit.fig.add_subplot(gs[1, 0])
        ax4.semilogx(f_r, re_r, 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45, label='Raw')
        ax4.semilogx(f_f, re_f, 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax4.semilogx(f_f, re_fit, '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax4.set_xlabel("Frequency (Hz)"); ax4.set_ylabel("Z' (Ω)")
        ax4.set_title("Real Impedance vs f", fontweight='bold')
        ax4.legend(framealpha=0.8); ax4.grid(True, alpha=0.3)

        ax5 = self.canvas_fit.fig.add_subplot(gs[1, 1])
        ax5.semilogx(f_r, -im_r, 'o', color=COLORS['raw_marker'], ms=3, alpha=0.45, label='Raw')
        ax5.semilogx(f_f, -im_f, 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax5.semilogx(f_f, -im_fit, '-', color=COLORS['fit_line'], lw=1.8, label='Fit')
        ax5.set_xlabel("Frequency (Hz)"); ax5.set_ylabel("-Z'' (Ω)")
        ax5.set_title("-Imag Impedance vs f", fontweight='bold')
        ax5.legend(framealpha=0.8); ax5.grid(True, alpha=0.3)

        ax6 = self.canvas_fit.fig.add_subplot(gs[1, 2])
        res_im_pct = 100.0 * res_im / np.maximum(np.abs(im_f), 1e-12)
        ax6.semilogx(f_f, res_im_pct, 's-', color=COLORS['res_im'], ms=5, lw=1.2, label='Imag residual %')
        ax6.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax6.set_xlabel("Frequency (Hz)"); ax6.set_ylabel("ΔZ''/Z'' (%)")
        ax6.set_title("Imag Residuals %", fontweight='bold')
        ax6.legend(framealpha=0.8); ax6.grid(True, alpha=0.3)

        self.canvas_fit.draw()

    def draw_kk_results(self, kk_result, filtered_data):
        """绘制 K-K 校验 4 子图"""
        from core.kramers_kronig import KramersKronigResult
        if kk_result is None or filtered_data is None:
            return

        self.canvas_kk.fig.clear()
        f = kk_result.frequencies
        Z_exp = kk_result.impedances_exp
        Z_fit = kk_result.impedances_fit
        res_re_pct = 100.0 * kk_result.residuals_real
        res_im_pct = 100.0 * kk_result.residuals_imag

        gs = self.canvas_kk.fig.add_gridspec(2, 2)

        # 1. Nyquist
        ax1 = self.canvas_kk.fig.add_subplot(gs[0, 0])
        ax1.plot(np.real(Z_exp), -np.imag(Z_exp), 'o', color=COLORS['sel_marker'],
                 ms=5, label='Exp')
        ax1.plot(np.real(Z_fit), -np.imag(Z_fit), '-', color=COLORS['kk_line'],
                 lw=1.8, label='K-K Fit')
        ax1.set_xlabel("Z' (Ω)"); ax1.set_ylabel("-Z'' (Ω)")
        ax1.set_title("Nyquist Plot (K-K)", fontweight='bold')
        ax1.legend(framealpha=0.8); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'datalim')

        # 2. Bode |Z|
        ax2 = self.canvas_kk.fig.add_subplot(gs[0, 1])
        ax2.loglog(f, np.abs(Z_exp), 'o', color=COLORS['sel_marker'], ms=5, label='Exp')
        ax2.loglog(f, np.abs(Z_fit), '-', color=COLORS['kk_line'], lw=1.8, label='K-K Fit')
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|Z| (Ω)")
        ax2.set_title("Bode Plot |Z| vs f", fontweight='bold')
        ax2.legend(framealpha=0.8); ax2.grid(True, alpha=0.3, which='both')

        # 3. Real residuals %
        ax3 = self.canvas_kk.fig.add_subplot(gs[1, 0])
        ax3.semilogx(f, res_re_pct, 'o-', color=COLORS['res_re'], ms=5, lw=1.2,
                     label='Real residual %')
        ax3.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("ΔZ'/|Z| (%)")
        ax3.set_title("Real Residuals %", fontweight='bold')
        ax3.legend(framealpha=0.8); ax3.grid(True, alpha=0.3)

        # 4. Imag residuals %
        ax4 = self.canvas_kk.fig.add_subplot(gs[1, 1])
        ax4.semilogx(f, res_im_pct, 's-', color=COLORS['res_im'], ms=5, lw=1.2,
                     label='Imag residual %')
        ax4.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax4.set_xlabel("Frequency (Hz)"); ax4.set_ylabel("ΔZ''/|Z| (%)")
        ax4.set_title("Imag Residuals %", fontweight='bold')
        ax4.legend(framealpha=0.8); ax4.grid(True, alpha=0.3)

        self.canvas_kk.draw()
