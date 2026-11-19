"""
Tab 3: DRT-DCT — 时间/电容分布解析 + 可视化 + StatsDock
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QGroupBox, QLabel, QPushButton, QRadioButton, QDoubleSpinBox,
    QSpinBox, QTabWidget, QFileDialog, QMessageBox, QApplication,
    QLineEdit, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

from gui.docks.stats_dock import StatsDock
from gui.widgets.floating_progress import FloatingProgress
from core.drt import solve_drt
from core.dct import solve_dct
from core.gdrt import solve_gdrt
from core import solve_drt_dop_ddt, parse_grid_spec

COLORS = {
    'gamma': '#1a237e', 'nyq_exp': '#2962ff', 'nyq_fit': '#c62828',
    'res_re': '#2e7d32', 'res_im': '#f57f17',
    'gamma_rl': '#e91e63', 'gamma_rc': '#1a237e', 'f0_line': '#9e9e9e',
    'dop': '#00897b', 'ddt': '#c62828', 'dop2': '#4db6ac', 'ddt2': '#ab47bc',
}
##6a1b9a
METHOD_KEYS = ['DRT', 'DCT', 'gDRT', 'DRT-DOP', 'DRT-DDT', 'DOP', 'DDT']
METHOD_LABELS = ['DRT (阻抗空间)', 'DCT (导纳空间)', 'gDRT (广义 DRT)',
                 'DRT-DOP (弛豫+相位角)', 'DRT-DDT (弛豫+扩散时间)',
                 'DOP (纯相位角)', 'DDT (纯扩散时间)']

DOP_DDT_METHODS = {'DRT-DOP', 'DRT-DDT', 'DOP', 'DDT'}

DDT_KERNEL_KEYS = [1, 2, 3]
DDT_KERNEL_LABELS = ['1: coth(u)/u (Reflective boundary planar)', '2: I0/(u·I1) (Reflective boundary cylindrical)', '3: tanh(u)/u (Transmissive boundary planar)']

RBF_TYPE_KEYS = [
    'Piecewise Linear', 'Gaussian', 'C0 Matern', 'C2 Matern',
    'C4 Matern', 'C6 Matern', 'Inverse Quadratic',
    'Inverse Quadric', 'Cauchy',
]

RBF_TYPE_LABELS = [
    '分段线性 (PWL)', '高斯 Gaussian', 'C0 Matérn', 'C2 Matérn',
    'C4 Matérn', 'C6 Matérn', '逆二次 Inverse Quadratic',
    '逆二重 Inverse Quadric', '柯西 Cauchy',
]

SHAPE_CONTROL_KEYS = ['FWHM Coefficient', 'Shape Factor']
SHAPE_CONTROL_LABELS = ['FWHM 系数', '形状因子']

TAU_STRATEGY_KEYS = ['logspace', 'data']
TAU_STRATEGY_LABELS = ['logspace 网格', 'τ = 1/f (数据对齐)']

WEIGHT_KEYS = ['boukamp', 'none']
WEIGHT_LABELS = ['Boukamp', '无 (不加权)']

DERIV_ORDER_KEYS = [2, 1]
DERIV_ORDER_LABELS = ['2阶', '1阶']

LAMBDA_SELECT_KEYS = ['manual', 'gcv', 'mgcv', 'lc']
LAMBDA_SELECT_LABELS = ['手动 (固定)', 'GCV', 'mGCV', 'L曲线']


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(5, 4), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='white', layout='constrained')
        super().__init__(self.fig)


class DRTDCTTab(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._result = None
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

        method_grp = QGroupBox("Method")
        method_layout = QVBoxLayout(method_grp)
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHOD_LABELS)
        method_layout.addWidget(self.method_combo)
        left_layout.addWidget(method_grp)

        self.param_grp = QGroupBox("Parameters set")
        param_layout = QVBoxLayout(self.param_grp)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("λ opt method:"))
        self.lam_select_combo = QComboBox()
        self.lam_select_combo.addItems(LAMBDA_SELECT_LABELS)
        row1.addWidget(self.lam_select_combo)
        row1.addStretch()
        param_layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("λ:"))
        self.lam_edit = QLineEdit("1e-3")
        self.lam_edit.setValidator(QDoubleValidator(1e-12, 1.0, 12))
        self.lam_edit.setStyleSheet("font-size: 9pt;")
        self.lam_edit.setFixedWidth(120)
        row2.addWidget(self.lam_edit)
        row2.addStretch()
        param_layout.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Creat τ strategy:"))
        self.tau_strategy_combo = QComboBox()
        self.tau_strategy_combo.addItems(TAU_STRATEGY_LABELS)
        self.tau_strategy_combo.setCurrentIndex(1)  # 默认 τ=1/f
        row3.addWidget(self.tau_strategy_combo)
        row3.addStretch()
        param_layout.addLayout(row3)
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Number of τ:"))
        self.ntau_spin = QSpinBox()
        self.ntau_spin.setRange(50, 1000)
        self.ntau_spin.setValue(400)
        self.ntau_spin.setSingleStep(50)
        self.ntau_spin.setFixedWidth(80)
        row4.addWidget(self.ntau_spin)
        row4.addStretch()
        param_layout.addLayout(row4)
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("RBF type:"))
        self.rbf_combo = QComboBox()
        self.rbf_combo.addItems(RBF_TYPE_LABELS)
        row5.addWidget(self.rbf_combo)
        row5.addStretch()
        param_layout.addLayout(row5)
        row6 = QHBoxLayout()
        row6.addWidget(QLabel("Shape type:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(SHAPE_CONTROL_LABELS)
        row6.addWidget(self.shape_combo)
        row6.addStretch()
        param_layout.addLayout(row6)
        row7 = QHBoxLayout()
        row7.addWidget(QLabel("Shape coefficient:"))
        self.coeff_edit = QLineEdit("0.5")
        self.coeff_edit.setValidator(QDoubleValidator(1e-6, 1e6, 6))
        self.coeff_edit.setStyleSheet("font-size: 9pt;")
        self.coeff_edit.setFixedWidth(120)
        row7.addWidget(self.coeff_edit)
        row7.addStretch()
        param_layout.addLayout(row7)
        row8 = QHBoxLayout()
        row8.addWidget(QLabel("Weight:"))
        self.weight_combo = QComboBox()
        self.weight_combo.addItems(WEIGHT_LABELS)
        self.weight_combo.setCurrentIndex(1)  # 默认 无(不加权)
        row8.addWidget(self.weight_combo)
        row8.addStretch()
        param_layout.addLayout(row8)
        row9 = QHBoxLayout()
        row9.addWidget(QLabel("Derivative order:"))
        self.deriv_combo = QComboBox()
        self.deriv_combo.addItems(DERIV_ORDER_LABELS)
        row9.addWidget(self.deriv_combo)
        row9.addStretch()
        param_layout.addLayout(row9)
        row10 = QHBoxLayout()
        self.cap_cb = QCheckBox("Including the capacitive term(1/(jωC))")
        self.cap_cb.setChecked(True)
        row10.addWidget(self.cap_cb)
        row10.addStretch()
        param_layout.addLayout(row10)
        left_layout.addWidget(self.param_grp)

        self.dop_ddt_grp = QGroupBox("DRT-DOP/DDT parameters")
        dd_layout = QVBoxLayout(self.dop_ddt_grp)

        def _dd_row(layout, label, widget):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            row.addStretch()
            layout.addLayout(row)

        self.fit_ohmic_cb = QCheckBox("拟合 R∞ (fit_ohmic)")
        self.fit_ohmic_cb.setChecked(True)
        self.fit_inductance_cb = QCheckBox("拟合 电感 (fit_inductance)")
        self.fit_inductance_cb.setChecked(False)
        self.fit_capacitance_cb = QCheckBox("拟合 电容 (fit_capacitance)")
        self.fit_capacitance_cb.setChecked(False)
        dd_layout.addWidget(self.fit_ohmic_cb)
        dd_layout.addWidget(self.fit_inductance_cb)
        dd_layout.addWidget(self.fit_capacitance_cb)

        self.ddt_kernel_combo = QComboBox()
        self.ddt_kernel_combo.addItems(DDT_KERNEL_LABELS)
        _dd_row(dd_layout, "DDT kernel:", self.ddt_kernel_combo)

        self.fit_ddt_p_cb = QCheckBox("1D(tau) (fit_ddt_p)")
        self.fit_ddt_p_cb.setChecked(False)
        dd_layout.addWidget(self.fit_ddt_p_cb)

        self.fit_ddt_2d_cb = QCheckBox("2D(tau,p) DDT (fit_ddt_2d)")
        self.fit_ddt_2d_cb.setChecked(False)
        dd_layout.addWidget(self.fit_ddt_2d_cb)

        # self.tau_dd_grid_edit = QLineEdit()
        # self.tau_dd_grid_edit.setPlaceholderText(
        #     "示例: np.logspace(-4, 4, 40) (留空=自动)")
        # self.tau_dd_grid_edit.setStyleSheet("font-size: 9pt;")
        # _dd_row(dd_layout, "basis_tau_dd:", self.tau_dd_grid_edit)

        self.tau_basis_combo = QComboBox()
        self.tau_basis_combo.addItems(['gaussian', 'delta', 'Cole-Cole', 'pwl'])
        _dd_row(dd_layout, "DRT basis function:", self.tau_basis_combo)

        self.tau_dd_basis_combo = QComboBox()
        self.tau_dd_basis_combo.addItems(['gaussian', 'delta'])
        _dd_row(dd_layout, "DDT basis function:", self.tau_dd_basis_combo)

        self.basis_ppd_spin = QSpinBox()
        self.basis_ppd_spin.setRange(2, 50)
        self.basis_ppd_spin.setValue(10)
        _dd_row(dd_layout, "DRT tau grid ppd:", self.basis_ppd_spin)

        self.extend_decades_spin = QSpinBox()
        self.extend_decades_spin.setRange(0, 3)
        self.extend_decades_spin.setValue(1)
        _dd_row(dd_layout, "Extend DRT decade:", self.extend_decades_spin)
        
        ##        
        self.tau_dd_grid_edit = QLineEdit()
        self.tau_dd_grid_edit.setPlaceholderText(
            "示例: np.logspace(-4, 4, 40) (留空=自动)")
        self.tau_dd_grid_edit.setStyleSheet("font-size: 9pt;")
        _dd_row(dd_layout, "Basis_tau_ddt:", self.tau_dd_grid_edit)

        self.l2_lambda_edit = QLineEdit("140")
        self.l2_lambda_edit.setStyleSheet("font-size: 9pt;")
        _dd_row(dd_layout, "DRT_l2_λ0:", self.l2_lambda_edit)  # 默认 140，与 facade 一致

        self.ddt_l2_lambda_edit = QLineEdit("1.0")
        self.ddt_l2_lambda_edit.setStyleSheet("font-size: 9pt;")
        _dd_row(dd_layout, "DDT_l2_λ0:", self.ddt_l2_lambda_edit)

        self.dop_l2_lambda_edit = QLineEdit("10")
        self.dop_l2_lambda_edit.setStyleSheet("font-size: 9pt;")
        _dd_row(dd_layout, "DOP_l2_λ0:", self.dop_l2_lambda_edit)

        self.rp_scale_edit = QLineEdit("50")
        self.rp_scale_edit.setStyleSheet("font-size: 9pt;")
        _dd_row(dd_layout, "rp_scale:", self.rp_scale_edit)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 500)
        self.max_iter_spin.setValue(30)
        _dd_row(dd_layout, "Max_iter:", self.max_iter_spin)

        left_layout.addWidget(self.dop_ddt_grp)

        data_grp = QGroupBox("数据源")
        data_layout = QVBoxLayout(data_grp)
        self.data_info = QLabel("使用 Data 标签页筛选后的数据\n未导入数据")
        self.data_info.setStyleSheet("color: #616161; font-size: 9pt;")
        data_layout.addWidget(self.data_info)
        left_layout.addWidget(data_grp)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ 运行")
        self.btn_run.setObjectName("btnFit")
        self.btn_run.setFixedHeight(36)
        btn_row.addWidget(self.btn_run)
        self.btn_export = QPushButton("💾 导出结果")
        self.btn_export.setObjectName("btnExport")
        self.btn_export.setFixedHeight(36)
        btn_row.addWidget(self.btn_export)
        left_layout.addLayout(btn_row)

        self.stats_dock = StatsDock()
        left_layout.addWidget(self._make_group("结果统计", self.stats_dock))
        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.plot_tabs = QTabWidget()
        self.plot_tabs.setTabPosition(QTabWidget.South)
        
        tab_gamma = QWidget()
        self._setup_plot_tab(tab_gamma, 'canvas_gamma')
        self.plot_tabs.addTab(tab_gamma, "分布谱图")

        tab_res = QWidget()
        self._setup_plot_tab(tab_res, 'canvas_res')
        self.plot_tabs.addTab(tab_res, "残差分析")



        splitter.addWidget(scroll)
        splitter.addWidget(self.plot_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setHandleWidth(8)

        layout.addWidget(splitter)

        self.progress = FloatingProgress()
        self.progress.attach_to(self.plot_tabs)

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

    def _connect_signals(self):
        self.btn_run.clicked.connect(self._run)
        self.btn_export.clicked.connect(self._export_results)
        self.rbf_combo.currentIndexChanged.connect(self._update_rbf_widgets)
        self.tau_strategy_combo.currentIndexChanged.connect(self._update_tau_strategy_widgets)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.fit_ddt_p_cb.toggled.connect(self._on_fit_ddt_p_toggled)
        self.fit_ddt_2d_cb.toggled.connect(self._on_fit_ddt_2d_toggled)
        self._update_rbf_widgets()
        self._update_tau_strategy_widgets()
        self._on_method_changed()

    def _current_method(self):
        return METHOD_KEYS[self.method_combo.currentIndex()]

    def _update_rbf_widgets(self):
        is_rbf = self.rbf_combo.currentIndex() > 0
        self.shape_combo.setEnabled(is_rbf)
        self.coeff_edit.setEnabled(is_rbf)

    def _update_tau_strategy_widgets(self):
        is_data = self.tau_strategy_combo.currentIndex() > 0
        self.ntau_spin.setEnabled(not is_data)

    def _on_fit_ddt_p_toggled(self, checked):
        if checked and self.fit_ddt_2d_cb.isChecked():
            self.fit_ddt_2d_cb.blockSignals(True)
            self.fit_ddt_2d_cb.setChecked(False)
            self.fit_ddt_2d_cb.blockSignals(False)

    def _on_fit_ddt_2d_toggled(self, checked):
        if checked and self.fit_ddt_p_cb.isChecked():
            self.fit_ddt_p_cb.blockSignals(True)
            self.fit_ddt_p_cb.setChecked(False)
            self.fit_ddt_p_cb.blockSignals(False)

    def _on_method_changed(self):
        method = self._current_method()
        is_legacy = method not in DOP_DDT_METHODS
        self.param_grp.setVisible(is_legacy)
        self.dop_ddt_grp.setVisible(not is_legacy)
        self.cap_cb.setEnabled(method == 'DRT')
        has_drt = method in ('DRT-DOP', 'DRT-DDT')
        has_ddt = method in ('DRT-DDT', 'DDT')
        has_dop = method in ('DRT-DOP', 'DOP')
        self.tau_basis_combo.setEnabled(has_drt)
        self.basis_ppd_spin.setEnabled(has_drt)
        self.extend_decades_spin.setEnabled(has_drt)
        self.l2_lambda_edit.setEnabled(has_drt)
        self.ddt_kernel_combo.setEnabled(has_ddt)
        self.fit_ddt_p_cb.setEnabled(has_ddt)
        self.fit_ddt_2d_cb.setEnabled(has_ddt)
        self.tau_dd_basis_combo.setEnabled(has_ddt)
        self.ddt_l2_lambda_edit.setEnabled(has_ddt)
        self.dop_l2_lambda_edit.setEnabled(has_dop)
        self.tau_dd_grid_edit.setEnabled(has_ddt)

    def _settings_items(self):
        """参数设置 (标签, 值) 列表, 供统计与导出共用"""
        method = self._current_method()
        if method in DOP_DDT_METHODS:
            has_drt = method in ('DRT-DOP', 'DRT-DDT')
            has_ddt = method in ('DRT-DDT', 'DDT')
            has_dop = method in ('DRT-DOP', 'DOP')
            items = [
                ("fit_ohmic", "yes" if self.fit_ohmic_cb.isChecked() else "no"),
                ("fit_inductance", "yes" if self.fit_inductance_cb.isChecked() else "no"),
                ("fit_capacitance", "yes" if self.fit_capacitance_cb.isChecked() else "no"),
            ]
            if has_drt:
                items += [
                    ("DRT basis function", self.tau_basis_combo.currentText()),
                    ("DRT tau grid ppd", str(self.basis_ppd_spin.value())),
                    ("Extend decade", str(self.extend_decades_spin.value())),
                    ("DRT_l2_λ0", self.l2_lambda_edit.text()),
                ]
            if has_ddt:
                tau_dd_txt = self.tau_dd_grid_edit.text().strip()
                items += [
                    ("DDT basis function", self.tau_dd_basis_combo.currentText()),
                    ("DDT kernel", DDT_KERNEL_KEYS[self.ddt_kernel_combo.currentIndex()]),
                    ("fit_ddt_p", "yes" if self.fit_ddt_p_cb.isChecked() else "no"),
                    ("fit_ddt_2d", "yes" if self.fit_ddt_2d_cb.isChecked() else "no"),
                    ("basis_tau_ddt", tau_dd_txt if tau_dd_txt else "自动(按频率)"),
                    ("DDT_l2_λ0", self.ddt_l2_lambda_edit.text()),
                ]
            if has_dop:
                items.append(("DOP_l2_λ0", self.dop_l2_lambda_edit.text()))
            items += [
                ("rp_scale", self.rp_scale_edit.text()),
                ("Max_iter", str(self.max_iter_spin.value())),
            ]
            return items
        items = [
            ("RBF type", RBF_TYPE_KEYS[self.rbf_combo.currentIndex()]),
            ("Creat τ strategy", TAU_STRATEGY_LABELS[self.tau_strategy_combo.currentIndex()]),
            ("Weight", WEIGHT_LABELS[self.weight_combo.currentIndex()]),
            ("Derivative order", DERIV_ORDER_LABELS[self.deriv_combo.currentIndex()]),
            ("Shape type", SHAPE_CONTROL_LABELS[self.shape_combo.currentIndex()]),
            ("Shape coefficient", self.coeff_edit.text()),
            ("λ opt method", LAMBDA_SELECT_LABELS[self.lam_select_combo.currentIndex()]),
        ]
        if method == 'DRT':
            items.append(("Including the capacitive term(1/(jωC))", "yes" if self.cap_cb.isChecked() else "no"))
        return items

    def _tau_count_display(self, result):
        """按方法返回实际配点数显示字符串"""
        g = result['gamma']
        if 'tau_rc' in g:
            return (f"RC={result.get('n_tau_rc', '?')}, "
                    f"RL={result.get('n_tau_rl', '?')}")
        if 'basis_tau' in result and result['basis_tau'] is not None:
            return str(len(result['basis_tau']))
        return str(result.get('n_tau_used', self.ntau_spin.value()))

    def update_data_info(self, f_min, f_max, n_pts):
        self.data_info.setText(
            f"使用 Data 标签页筛选后的数据\n"
            f"频率范围: {f_min:.4f} ~ {f_max:.0f} Hz\n"
            f"数据点数: {n_pts}")

    def _set_computing(self, computing):
        self.btn_run.setEnabled(not computing)
        self.btn_export.setEnabled(not computing)

    def _run(self):
        if self.main_window is None or self.main_window.filtered_data is None:
            QMessageBox.warning(self, "无数据", "请先在 Data 标签页导入数据")
            return

        filtered = self.main_window.filtered_data
        self._exp_data = filtered.copy()
        f = filtered[:, 0]
        Z = filtered[:, 1] + 1j * filtered[:, 2]
        method = self._current_method()

        if method in DOP_DDT_METHODS:
            self._run_dop_ddt(f, Z, method)
            return

        try:
            lam = float(self.lam_edit.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "lambda 格式无效，请使用科学计数法如 1e-3")
            return
        n_tau = self.ntau_spin.value()
        rbf_type = RBF_TYPE_KEYS[self.rbf_combo.currentIndex()]
        rbf_shape = SHAPE_CONTROL_KEYS[self.shape_combo.currentIndex()]
        tau_strategy = TAU_STRATEGY_KEYS[self.tau_strategy_combo.currentIndex()]
        weight = WEIGHT_KEYS[self.weight_combo.currentIndex()]
        derivative_order = DERIV_ORDER_KEYS[self.deriv_combo.currentIndex()]
        cross_validation = LAMBDA_SELECT_KEYS[self.lam_select_combo.currentIndex()]
        if cross_validation == 'manual':
            cross_validation = ''
        add_capacitance = self.cap_cb.isChecked()
        try:
            rbf_coeff = float(self.coeff_edit.text())
        except ValueError:
            rbf_coeff = 0.5

        self._set_computing(True)
        try:
            self.progress.show_busy(f"{method} 求解中…")
            QApplication.processEvents()
            if method == 'gDRT':
                result = solve_gdrt(f, Z, lam=lam, n_tau=n_tau,
                                    rbf_type=rbf_type, rbf_shape=rbf_shape,
                                    rbf_coeff=rbf_coeff, tau_strategy=tau_strategy,
                                    weight=weight, derivative_order=derivative_order,
                                    cross_validation=cross_validation)
            elif method == 'DRT':
                result = solve_drt(f, Z, lam=lam, n_tau=n_tau,
                                   rbf_type=rbf_type, rbf_shape=rbf_shape,
                                   rbf_coeff=rbf_coeff, tau_strategy=tau_strategy,
                                   weight=weight, add_capacitance=add_capacitance,
                                   derivative_order=derivative_order,
                                   cross_validation=cross_validation)
            else:
                result = solve_dct(f, Z, lam=lam, n_tau=n_tau,
                                   rbf_type=rbf_type, rbf_shape=rbf_shape,
                                   rbf_coeff=rbf_coeff, tau_strategy=tau_strategy,
                                   weight=weight, derivative_order=derivative_order,
                                   cross_validation=cross_validation)

            self._result = result
            self._draw_drt_page(f, Z, result, method)
            self._draw_residuals(result)
            self._update_stats(result, method)
            self.progress.finish()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "求解失败", str(e))
        finally:
            self._set_computing(False)

    def _run_dop_ddt(self, f, Z, method):
        """DRT-DOP / DRT-DDT / DOP / DDT 联合或单独求解"""
        fit_ddt_p = self.fit_ddt_p_cb.isChecked()
        fit_ddt_2d = self.fit_ddt_2d_cb.isChecked()
        if fit_ddt_p and fit_ddt_2d:
            QMessageBox.warning(self, "参数错误",
                                "fit_ddt_p 与 fit_ddt_2d 互斥，请只勾选其一")
            return

        # basis_tau_dd: 留空自动; 填写 logspace(...) 为用户定义网格
        tau_dd_grid = None
        tau_dd_txt = self.tau_dd_grid_edit.text().strip()
        if tau_dd_txt:
            try:
                tau_dd_grid = parse_grid_spec(tau_dd_txt, p_grid=False)
            except ValueError as e:
                QMessageBox.warning(self, "参数错误", f"basis_tau_dd 无效: {e}")
                return

        def _f(name, default):
            try:
                return float(getattr(self, name).text())
            except ValueError:
                return default

        def _progress_cb(it, total):
            self.progress.show_progress(f"QPHB 迭代 {it}/{total}", total)
            self.progress.set_progress(it)
            QApplication.processEvents()

        self._set_computing(True)
        try:
            result = solve_drt_dop_ddt(
                f, Z, method=method,
                fit_ohmic=self.fit_ohmic_cb.isChecked(),
                fit_inductance=self.fit_inductance_cb.isChecked(),
                fit_capacitance=self.fit_capacitance_cb.isChecked(),
                ddt_kernel_type=DDT_KERNEL_KEYS[self.ddt_kernel_combo.currentIndex()],
                fit_ddt_p=fit_ddt_p,
                fit_ddt_2d=fit_ddt_2d,
                fixed_basis_tau_dd=tau_dd_grid,
                tau_basis_type=self.tau_basis_combo.currentText(),
                tau_dd_basis_type=self.tau_dd_basis_combo.currentText(),
                basis_tau_ppd=self.basis_ppd_spin.value(),
                extend_decades=self.extend_decades_spin.value(),
                l2_lambda_0=_f('l2_lambda_edit', 140.0),
                ddt_l2_lambda_0=_f('ddt_l2_lambda_edit', 1.0),
                dop_l2_lambda_0=_f('dop_l2_lambda_edit', 10.0),
                rp_scale=_f('rp_scale_edit', 50.0),
                max_iter=self.max_iter_spin.value(),
                progress_callback=_progress_cb,
            )
            self._result = result
            self._draw_drt_page(f, Z, result, method)
            self._draw_residuals(result)
            self._update_stats(result, method)
            self.progress.finish()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "求解失败", str(e))
        finally:
            self._set_computing(False)

    def _draw_drt_page(self, f, Z, result, method):
        self.canvas_gamma.fig.clear()
        Z_recon = result['Z_recon']
        f_s = result['f_sorted']
        sort_idx = np.argsort(f)
        if len(f_s) > 1 and f_s[0] > f_s[-1]:
            sort_idx = sort_idx[::-1]
        Z_s = Z[sort_idx]

        gs = self.canvas_gamma.fig.add_gridspec(2, 3)

        if method in DOP_DDT_METHODS:
            self._draw_dop_ddt_page(gs, result, method)
        else:
            gamma = result['gamma']
            ax_spec = self.canvas_gamma.fig.add_subplot(gs[0, :])
            if method == "gDRT":
                ax_spec.semilogx(gamma['tau_rc'], gamma['val_rc'], '-',
                                 color=COLORS['gamma_rc'], lw=2, label='RC (Cap)')
                ax_spec.semilogx(gamma['tau_rl'], gamma['val_rl'], '-',
                                 color=COLORS['gamma_rl'], lw=2, label='RL (Ind)')
                tau0 = 1.0 / (2.0 * np.pi * gamma['f0'])
                ax_spec.axvline(tau0, color=COLORS['f0_line'], ls='--', lw=1.2, label='τ₀')
                ax_spec.set_ylabel("g(τ) (Ω/lnτ)")
                ax_spec.set_title("gDRT Spectrum (RC + RL)", fontweight='bold')
                ax_spec.legend(framealpha=0.8)
            else:
                ax_spec.semilogx(gamma['tau'], gamma['val'], '-', color=COLORS['gamma'], lw=2)
                ax_spec.set_ylabel("g(τ) " + ("(Ω/lnτ)" if method == "DRT" else "(S/lnτ)"))
                ax_spec.set_title(f"{method} Spectrum", fontweight='bold')
            ax_spec.set_xlabel("τ (s)")
            ax_spec.grid(True, alpha=0.3, which='both')

        ax_nyq = self.canvas_gamma.fig.add_subplot(gs[1, 0])
        ax_nyq.plot(np.real(Z_s), -np.imag(Z_s), 'o', color=COLORS['nyq_exp'], ms=5, label='Exp')
        ax_nyq.plot(np.real(Z_recon), -np.imag(Z_recon), '-', color=COLORS['nyq_fit'], lw=2, label=f'{method} Fit')
        ax_nyq.set_xlabel("Z' (Ω)"); ax_nyq.set_ylabel("-Z'' (Ω)")
        ax_nyq.set_title("Nyquist Comparison", fontweight='bold')
        ax_nyq.legend(framealpha=0.8); ax_nyq.grid(True, alpha=0.3)

        ax_real = self.canvas_gamma.fig.add_subplot(gs[1, 1])
        ax_real.semilogx(f_s, np.real(Z_s), 'o', color=COLORS['nyq_exp'], ms=5, label='Exp')
        ax_real.semilogx(f_s, np.real(Z_recon), '-', color=COLORS['nyq_fit'], lw=2, label=f'{method} Fit')
        ax_real.set_xlabel("Frequency (Hz)"); ax_real.set_ylabel("Z' (Ω)")
        ax_real.set_title("Re(Z) vs f", fontweight='bold')
        ax_real.legend(framealpha=0.8); ax_real.grid(True, alpha=0.3, which='both')

        ax_imag = self.canvas_gamma.fig.add_subplot(gs[1, 2])
        ax_imag.semilogx(f_s, -np.imag(Z_s), 'o', color=COLORS['nyq_exp'], ms=5, label='Exp')
        ax_imag.semilogx(f_s, -np.imag(Z_recon), '-', color=COLORS['nyq_fit'], lw=2, label=f'{method} Fit')
        ax_imag.set_xlabel("Frequency (Hz)"); ax_imag.set_ylabel("-Z'' (Ω)")
        ax_imag.set_title("-Im(Z) vs f", fontweight='bold')
        ax_imag.legend(framealpha=0.8); ax_imag.grid(True, alpha=0.3, which='both')

        self.canvas_gamma.draw()

    def _draw_dop_ddt_page(self, gs, result, method):
        """DRT-DOP / DRT-DDT / DOP / DDT 顶行绘图"""
        has_drt = 'gamma' in result
        has_dop = 'dop' in result
        has_ddt = 'ddt' in result

        if has_drt:
            # 联合求解: DRT 占 (0,0:2), DOP/DDT 占 (0,2)
            ax_drt = self.canvas_gamma.fig.add_subplot(gs[0, 0:2])
            g = result['gamma']
            ax_drt.semilogx(g['tau'], g['val'], '-', color=COLORS['gamma'], lw=2)
            ax_drt.set_xlabel("τ (s)"); ax_drt.set_ylabel("g(τ) (Ω/lnτ)")
            ax_drt.set_title("DRT", fontweight='bold')
            ax_drt.grid(True, alpha=0.3, which='both')
            ax_extra = self.canvas_gamma.fig.add_subplot(gs[0, 2])
        else:
            # 纯 DOP/DDT: 顶行整行绘制分布
            ax_extra = self.canvas_gamma.fig.add_subplot(gs[0, :])

        if has_dop:
            d = result['dop']
            ax_extra.plot(d['nu'], d['rho'], '-', color=COLORS['dop'], lw=2)
            #ax_extra.axvline(-0.5, ls=':', c='gray', lw=1, label='ν=-p=-0.5')
            ax_extra.set_xlabel("ν"); ax_extra.set_ylabel("ρ(ν) (Ω)")
            ax_extra.set_title("DOP", fontweight='bold')
            ax_extra.legend(framealpha=0.8)
            ax_extra.grid(True, alpha=0.3)
        elif has_ddt:
            d = result['ddt']
            if 'q2d' in d:
                # 2D 异常 DDT 热图
                q2d = np.asarray(d['q2d']).T  # (n_p, n_tau)
                pc = ax_extra.pcolormesh(d['tau_dd'], d['p'], q2d, shading='auto',
                                         cmap='viridis')
                self.canvas_gamma.fig.colorbar(pc, ax=ax_extra, label='q (Ω)')
                if 'q_tau' in d:
                    ax2 = ax_extra.twinx()
                    ax2.plot(d['tau_dd'], d['q_tau'], '-', color=COLORS['ddt'], lw=1.2,
                             label='p-marginal')
                    ax2.set_ylabel("q(τ_D) (Ω)")
                ax_extra.set_yscale('linear')
                ax_extra.set_title("DDT (2D anomalous)", fontweight='bold')
            else:
                ax_extra.semilogx(d['tau_dd'], d['q'], '-', color=COLORS['ddt'], lw=2)
                ax_extra.set_title("DDT", fontweight='bold')
            ax_extra.set_xscale('log')
            ax_extra.set_xlabel("τ_D (s)")
            if 'q2d' not in d:
                ax_extra.set_ylabel("q(τ_D) (Ω)")
            ax_extra.grid(True, alpha=0.3, which='both')

    def _draw_residuals(self, result):
        self.canvas_res.fig.clear()
        ax1 = self.canvas_res.fig.add_subplot(121)
        ax1.semilogx(result['f_sorted'], 100.0 * result['res_re'], 'o-',
                     color=COLORS['res_re'], ms=5, lw=1.2, label='Real residual')
        ax1.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("DZ'/|Z| (%)")
        ax1.set_title("Real Residuals", fontweight='bold')
        ax1.legend(framealpha=0.8); ax1.grid(True, alpha=0.3)
        ax2 = self.canvas_res.fig.add_subplot(122)
        ax2.semilogx(result['f_sorted'], 100.0 * result['res_im'], 's-',
                     color=COLORS['res_im'], ms=5, lw=1.2, label='Imag residual')
        ax2.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("DZ''/|Z| (%)")
        ax2.set_title("Imag Residuals", fontweight='bold')
        ax2.legend(framealpha=0.8); ax2.grid(True, alpha=0.3)
        self.canvas_res.draw()

    def _update_stats(self, result, method):
        lam_used = result.get('lambda_used', float(self.lam_edit.text()))
        lines = ["=" * 48,
                 f"  {method} 求解结果",
                 "=" * 48,
                 f"  method       : {method}"]
        for label, val in self._settings_items():
            lines.append(f"  {label:<12}: {val}")
        if method not in DOP_DDT_METHODS:
            lines.append(f"  λ opted     : {lam_used:.6e}")
        lines.append(f"  chi2       : {result['chi2']:.6e}")
        gamma = result.get('gamma', {})
        if method in DOP_DDT_METHODS:
            lines.append(f"  R∞ (Ω)     : {result.get('R_inf', 0):.6e}")
            if result.get('inductance'):
                lines.append(f"  L (H)      : {result.get('inductance'):.6e}")
            if result.get('C_inv'):
                lines.append(f"  1/C (Ω·s)  : {result.get('C_inv'):.6e}")
            if 'ddt_p' in result:
                lines.append(f"  ddt_p      : {result.get('ddt_p'):.6f}")
            if 'dop' in result:
                d = result['dop']
                lines.append(f"  DOP ν 范围 : {d['nu'].min():.3f} ~ {d['nu'].max():.3f}")
                lines.append(f"  DOP 峰 ν   : {d['nu'][np.argmax(d['rho'])]:.3f}")
            if 'ddt' in result:
                d = result['ddt']
                if 'q2d' in d:
                    lines.append(f"  DDT τ_D 范围: {d['tau_dd'].min():.3g} ~ {d['tau_dd'].max():.3g} s")
                    lines.append(f"  DDT p 网格  : {len(d['p'])} 点 "
                                 f"[{d['p'].min():.3f}, {d['p'].max():.3f}]")
                    if 'q_tau' in d and np.max(np.abs(d['q_tau'])) > 0:
                        lines.append(f"  DDT 峰 τ_D : {d['tau_dd'][np.argmax(d['q_tau'])]:.4g} s")
                else:
                    lines.append(f"  DDT τ_D 范围: {d['tau_dd'].min():.3g} ~ {d['tau_dd'].max():.3g} s")
                    lines.append(f"  DDT 峰 τ_D : {d['tau_dd'][np.argmax(d['q'])]:.4g} s")
        elif 'G_inf' in gamma:
            lines.append(f"  G_inf (S)  : {gamma['G_inf']:.6e}")
            lines.append(f"  C_inf (F)  : {gamma['C_inf']:.6e}")
        elif 'R0' in gamma and 'L0' in gamma:
            lines.append(f"  R₀ (Ω)     : {gamma['R0']:.6e}")
            lines.append(f"  C  (F)     : {gamma['C']:.6e}")
            lines.append(f"  L₀ (H)     : {gamma['L0']:.6e}")
            lines.append(f"  f₀ (Hz)    : {gamma['f0']:.4f}")
        elif 'R0' in gamma:
            lines.append(f"  R₀ (Ω)     : {gamma['R0']:.6e}")
            if 'C' in gamma:
                lines.append(f"  C  (F)     : {gamma['C']:.6e}")
        lines.append("-" * 48)
        lines.append("  残差统计 (归一化于 |Z|):")
        lines.append(f"  DRe' max = {100*np.max(np.abs(result['res_re'])):.4f}%")
        lines.append(f"  DIm'' max = {100*np.max(np.abs(result['res_im'])):.4f}%")
        lines.append("=" * 48)
        self.stats_dock.set_text("\n".join(lines))

    def _export_results(self):
        if self._result is None:
            QMessageBox.warning(self, "无结果", "请先运行求解")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "保存 DRT/DOP/DDT 结果", "", "xlsx (*.xlsx)")
        if not fp:
            return
        try:
            import pandas as pd
            res = self._result
            method = res.get('method', self._current_method())
            gamma = res.get('gamma', {})

            items = []
            vals = []
            def a(item, val):
                items.append(item); vals.append(val)

            if method in DOP_DDT_METHODS:
                a("Method", method)
                a("R_inf (Ohm)", f"{res.get('R_inf', 0):.6e}")
                if res.get('inductance'):
                    a("L (H)", f"{res.get('inductance'):.6e}")
                if res.get('C_inv'):
                    a("1/C (Ohm·s)", f"{res.get('C_inv'):.6e}")
                if 'ddt_p' in res:
                    a("ddt_p", f"{res.get('ddt_p'):.6f}")
            elif 'G_inf' in gamma:
                a("Method", "DCT")
                a("G_inf (S)", f"{gamma['G_inf']:.6e}")
                a("C_inf (F)", f"{gamma['C_inf']:.6e}")
            elif 'tau_rc' in gamma:
                a("Method", "gDRT")
                a("R0 (Ohm)", f"{gamma['R0']:.6e}")
                a("C (F)", f"{gamma['C']:.6e}")
                a("L0 (H)", f"{gamma['L0']:.6e}")
                a("f0 (Hz)", f"{gamma['f0']:.4f}")
            else:
                a("Method", "DRT")
                a("R0 (Ohm)", f"{gamma['R0']:.6e}")
                if 'C' in gamma:
                    a("C (F)", f"{gamma['C']:.6e}")
            for label, val in self._settings_items():
                a(label, val)
            if 'lambda_used' in res and res['lambda_used'] is not None:
                a("lambda_used", f"{res['lambda_used']:.6e}")
            a("chi2", f"{res['chi2']:.6e}")
            a("DRe' max (%)", f"{100*np.max(np.abs(res['res_re'])):.4f}")
            a("DIm'' max (%)", f"{100*np.max(np.abs(res['res_im'])):.4f}")

            df_stats = pd.DataFrame({"Item": items, "Value": vals})

            df_exp = pd.DataFrame({
                "f(Hz)": self._exp_data[:, 0],
                "Real_exp": self._exp_data[:, 1],
                "Imag_exp": self._exp_data[:, 2],
            })
            df_imp = pd.DataFrame({
                "f(Hz)": res['f_sorted'],
                "Real_recon": np.real(res['Z_recon']),
                "Imag_recon": np.imag(res['Z_recon']),
                "Res_Re": res['res_re'],
                "Res_Im": res['res_im'],
            })

            with pd.ExcelWriter(fp, engine='openpyxl') as writer:
                df_stats.to_excel(writer, sheet_name="Statistics", index=False)
                df_exp.to_excel(writer, sheet_name="experiment_Z", index=False)
                df_imp.to_excel(writer, sheet_name="Reconstructed_Z", index=False)

                if method in DOP_DDT_METHODS:
                    if 'gamma' in res:
                        pd.DataFrame({"tau(s)": res['gamma']['tau'],
                                      "g (Ohm/ln tau)": res['gamma']['val']}).to_excel(
                            writer, sheet_name="Gamma_Spectrum", index=False)
                    if 'dop' in res:
                        pd.DataFrame({"nu": res['dop']['nu'],
                                      "rho (Ohm)": res['dop']['rho']}).to_excel(
                            writer, sheet_name="DOP_Spectrum", index=False)
                    if 'ddt' in res:
                        d = res['ddt']
                        if 'q2d' in d:
                            q2d = np.asarray(d['q2d']).T  # (n_p, n_tau)
                            pd.DataFrame(q2d, index=np.round(d['p'], 6),
                                         columns=np.round(d['tau_dd'], 6)).to_excel(
                                writer, sheet_name="DDT_2D")
                            if 'q_tau' in d:
                                pd.DataFrame({"tau_dd(s)": d['tau_dd'],
                                              "q (Ohm)": d['q_tau']}).to_excel(
                                    writer, sheet_name="DDT_marginal", index=False)
                        else:
                            pd.DataFrame({"tau_dd(s)": d['tau_dd'],
                                          "q (Ohm)": d['q']}).to_excel(
                                writer, sheet_name="DDT_Spectrum", index=False)
                elif 'tau_rc' in gamma:
                    pd.DataFrame({"tau_rc(s)": gamma['tau_rc'],
                                  "g_rc (Ohm/ln tau)": gamma['val_rc']}).to_excel(
                        writer, sheet_name="Gamma_RC", index=False)
                    pd.DataFrame({"tau_rl(s)": gamma['tau_rl'],
                                  "g_rl (Ohm/ln tau)": gamma['val_rl']}).to_excel(
                        writer, sheet_name="Gamma_RL", index=False)
                elif 'G_inf' in gamma:
                    pd.DataFrame({"tau(s)": gamma['tau'],
                                  "g (S/ln tau)": gamma['val']}).to_excel(
                        writer, sheet_name="Gamma_Spectrum", index=False)
                else:
                    pd.DataFrame({"tau(s)": gamma['tau'],
                                  "g (Ohm/ln tau)": gamma['val']}).to_excel(
                        writer, sheet_name="Gamma_Spectrum", index=False)
            QMessageBox.information(self, "导出成功", f"结果已保存至:\n{fp.replace('.csv', '.xlsx')}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
