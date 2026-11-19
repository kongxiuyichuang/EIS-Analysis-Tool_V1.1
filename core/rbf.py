"""
径向基函数 (RBF) 离散方法
移植自 pyDRTtools (Wan et al. 2015), 参考 DearEIS 的 RBFType/RBFShape 枚举

参考:
    T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci,
    Influence of the discretization methods on the distribution of relaxation times
    deconvolution: Implementing radial basis functions with DRTtools,
    Electrochimica Acta 184 (2015) 483-499.

DRT 用 RBF 展开: gamma(ln tau) = sum_m x_m * phi_m(ln tau)
其中 phi_m 是中心在 ln(tau_m) 的径向基函数, 通过 QP 求解展开系数 x, 再映射回 gamma。
"""

import numpy as np
from numpy import exp, log, pi, sqrt
from scipy.optimize import fsolve


RBF_TYPES = [
    'Piecewise Linear',
    'Gaussian',
    'C0 Matern',
    'C2 Matern',
    'C4 Matern',
    'C6 Matern',
    'Inverse Quadratic',
    'Inverse Quadric',
    'Cauchy',
]

SHAPE_CONTROLS = ['FWHM Coefficient', 'Shape Factor']

# 轻尾 RBF (积分域自适应收缩); 重尾 RBF 用宽积分域
_LIGHT_TAILED = {'Gaussian', 'C0 Matern', 'C2 Matern', 'C4 Matern', 'C6 Matern'}


def _rbf_value(x, epsilon, rbf_type):
    """RBF 函数值 phi(x) = rbf(epsilon * x)"""
    if rbf_type == 'Gaussian':
        return exp(-(epsilon * x) ** 2)
    if rbf_type == 'C0 Matern':
        return exp(-np.abs(epsilon * x))
    if rbf_type == 'C2 Matern':
        ax = np.abs(epsilon * x)
        return exp(-ax) * (1.0 + ax)
    if rbf_type == 'C4 Matern':
        ax = np.abs(epsilon * x)
        return exp(-ax) * (3.0 + 3.0 * ax + ax ** 2) / 3.0
    if rbf_type == 'C6 Matern':
        ax = np.abs(epsilon * x)
        return exp(-ax) * (15.0 + 15.0 * ax + 6.0 * ax ** 2 + ax ** 3) / 15.0
    if rbf_type == 'Inverse Quadratic':
        return 1.0 / (1.0 + (epsilon * x) ** 2)
    if rbf_type == 'Inverse Quadric':
        return 1.0 / sqrt(1.0 + (epsilon * x) ** 2)
    if rbf_type == 'Cauchy':
        return 1.0 / (1.0 + np.abs(epsilon * x))
    raise ValueError(f"Unknown RBF type: {rbf_type}")


def compute_epsilon(tau_vec, coeff, rbf_type, shape_control):
    """计算 RBF 形状因子 epsilon

    shape_control:
        'FWHM Coefficient': FWHM = coeff 倍平均对数间距
        'Shape Factor':     epsilon = coeff (直接指定)
    """
    if rbf_type == 'Piecewise Linear':
        return 0.0

    if shape_control == 'FWHM Coefficient':
        def rbf_minus_half(x):
            return _rbf_value(x, 1.0, rbf_type) - 0.5

        fwhm_coeff = 2.0 * fsolve(rbf_minus_half, 1.0)[0]
        delta = np.mean(np.diff(np.log(tau_vec)))
        epsilon = coeff * fwhm_coeff / delta
    else:  # 'Shape Factor'
        epsilon = coeff

    return epsilon


def _integration_grid(epsilon, rbf_type, n_points=2000):
    """自适应积分网格: 轻尾 RBF 收缩到 ~10/epsilon, 重尾 RBF 用宽域"""
    if epsilon <= 0:
        return np.linspace(-60.0, 60.0, n_points)
    if rbf_type in _LIGHT_TAILED:
        w = 10.0 / epsilon
        return np.linspace(-w, w, n_points)
    return np.linspace(-60.0, 60.0, n_points)


def assemble_A(freq_vec, tau_vec, epsilon, rbf_type, kernel='drt'):
    """构造 RBF 离散化矩阵 A_re, A_im

    kernel='drt':  Z = R0 + 1/(j w C) + int g(tau)/(1+j w tau) d ln tau
    kernel='dct':  Y = G  + j w C     + int g(tau) * j w tau/(1+j w tau) d ln tau
    """
    omega = 2.0 * pi * np.asarray(freq_vec, dtype=np.float64)
    tau = np.asarray(tau_vec, dtype=np.float64)
    n_f = len(omega)
    n_t = len(tau)

    u = _integration_grid(epsilon, rbf_type)
    phi_u = _rbf_value(u, epsilon, rbf_type)
    e2u = exp(2.0 * u)
    eu = exp(u)

    A_re = np.zeros((n_f, n_t))
    A_im = np.zeros((n_f, n_t))

    for p in range(n_f):
        alpha = omega[p] * tau  # (n_t,)
        a2e2u = (alpha[:, None] ** 2) * e2u[None, :]
        denom = 1.0 + a2e2u
        if kernel == 'drt':
            k_re = 1.0 / denom
            k_im = -(alpha[:, None] * eu[None, :]) / denom
        elif kernel == 'dct':
            k_re = a2e2u / denom
            k_im = (alpha[:, None] * eu[None, :]) / denom
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        A_re[p, :] = np.trapezoid(k_re * phi_u[None, :], u, axis=1)
        A_im[p, :] = np.trapezoid(k_im * phi_u[None, :], u, axis=1)

    return A_re, A_im


def _inner_prod_rbf_2_inverse_quadric(a, epsilon):
    """Inverse Quadric 二阶导数内积 (数值积分, 解析二阶导数)"""
    sep = abs(a) / epsilon if epsilon > 0 else abs(a)
    u = np.linspace(-80.0, 80.0, 8000)

    def d2phi(x):
        t = (epsilon * x) ** 2
        return epsilon ** 2 * (2.0 * t - 1.0) / (1.0 + t) ** 2.5

    f1 = d2phi(u)
    f2 = d2phi(u - sep)
    return np.trapezoid(f1 * f2, u)


def _inner_prod_rbf_2(a, epsilon, rbf_type):
    """RBF 二阶导数内积 (参考 pyDRTtools inner_prod_rbf_2)"""
    aa = abs(a)
    if rbf_type == 'Gaussian':
        return epsilon ** 3 * (3.0 - 6.0 * a ** 2 + a ** 4) * exp(-(a ** 2 / 2.0)) * sqrt(pi / 2.0)
    if rbf_type == 'C0 Matern':
        return epsilon ** 3 * (1.0 + aa) * exp(-aa)
    if rbf_type == 'C2 Matern':
        return epsilon ** 3 / 6.0 * (3.0 + 3.0 * aa - 6.0 * aa ** 2 + aa ** 3) * exp(-aa)
    if rbf_type == 'C4 Matern':
        return epsilon ** 3 / 30.0 * (45.0 + 45.0 * aa - 15.0 * aa ** 3 - 5.0 * aa ** 4 + aa ** 5) * exp(-aa)
    if rbf_type == 'C6 Matern':
        return epsilon ** 3 / 140.0 * (
            2835.0 + 2835.0 * aa + 630.0 * aa ** 2 - 315.0 * aa ** 3
            - 210.0 * aa ** 4 - 42.0 * aa ** 5 + aa ** 7
        ) * exp(-aa)
    if rbf_type == 'Inverse Quadratic':
        return 48.0 * (16.0 + 5.0 * a ** 2 * (-8.0 + a ** 2)) * pi * epsilon ** 3 / (4.0 + a ** 2) ** 5
    if rbf_type == 'Inverse Quadric':
        return _inner_prod_rbf_2_inverse_quadric(a, epsilon)
    if rbf_type == 'Cauchy':
        if aa == 0:
            return 8.0 / 5.0 * epsilon ** 3
        num = aa * (2.0 + aa) * (
            -96.0 + aa * (2.0 + aa) * (-30.0 + aa * (2.0 + aa)) * (4.0 + aa * (2.0 + aa))
        ) + 12.0 * (1.0 + aa) ** 2 * (16.0 + aa * (2.0 + aa) * (12.0 + aa * (2.0 + aa))) * log(1.0 + aa)
        den = aa ** 5 * (1.0 + aa) * (2.0 + aa) ** 5
        return 8.0 * epsilon ** 3 * num / den
    raise ValueError(f"Unknown RBF type: {rbf_type}")


def assemble_M_2(tau_vec, epsilon, rbf_type):
    """构造二阶 Tikhonov 正则化矩阵 (RBF 二阶导数内积 Gram 矩阵)

    对数均匀 tau 网格下 M 为 Toeplitz 矩阵, 仅需计算第一行;
    非均匀网格 (如 tau=1/f 数据对齐配点) 采用逐对暴力计算。
    """
    n_t = len(tau_vec)
    log_tau = log(tau_vec)

    # 检测是否对数均匀 (允许轻微波动), 若是则用 Toeplitz 快速路径
    if n_t > 1:
        d = np.diff(log_tau)
        if np.all(d > 0) and np.std(d) / np.mean(d) < 0.01:
            delta = np.mean(d)
            offsets = epsilon * np.arange(n_t) * delta
            row = np.array([_inner_prod_rbf_2(off, epsilon, rbf_type) for off in offsets])
            M = np.zeros((n_t, n_t))
            for i in range(n_t):
                for j in range(n_t):
                    M[i, j] = row[abs(i - j)]
            return M

    # 暴力路径 (非均匀网格)
    M = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(n_t):
            a = epsilon * (log_tau[i] - log_tau[j])
            M[i, j] = _inner_prod_rbf_2(a, epsilon, rbf_type)
    return M


def _inner_prod_rbf_1_inverse_quadric(a, epsilon):
    """Inverse Quadric 一阶导数内积 (数值积分, 解析一阶导数)"""
    sep = abs(a) / epsilon if epsilon > 0 else abs(a)
    u = np.linspace(-80.0, 80.0, 8000)

    def dphi(x):
        t = (epsilon * x) ** 2
        return -epsilon ** 2 * x / (1.0 + t) ** 1.5

    f1 = dphi(u)
    f2 = dphi(u - sep)
    return np.trapezoid(f1 * f2, u)


def _inner_prod_rbf_1(a, epsilon, rbf_type):
    """RBF 一阶导数内积 (参考 pyDRTtools inner_prod_rbf_1)"""
    aa = abs(a)
    if rbf_type == 'Gaussian':
        return -epsilon * (-1.0 + a ** 2) * exp(-(a ** 2 / 2.0)) * sqrt(pi / 2.0)
    if rbf_type == 'C0 Matern':
        return epsilon * (1.0 - aa) * exp(-aa)
    if rbf_type == 'C2 Matern':
        return epsilon / 6.0 * (3.0 + 3.0 * aa - aa ** 3) * exp(-aa)
    if rbf_type == 'C4 Matern':
        return epsilon / 30.0 * (
            105.0 + 105.0 * aa + 30.0 * aa ** 2 - 5.0 * aa ** 3 - 5.0 * aa ** 4 - aa ** 5
        ) * exp(-aa)
    if rbf_type == 'C6 Matern':
        return epsilon / 140.0 * (
            10395.0 + 10395.0 * aa + 3780.0 * aa ** 2 + 315.0 * aa ** 3
            - 210.0 * aa ** 4 - 84.0 * aa ** 5 - 14.0 * aa ** 6 - aa ** 7
        ) * exp(-aa)
    if rbf_type == 'Inverse Quadratic':
        return 4.0 * epsilon * (4.0 - 3.0 * a ** 2) * pi / (4.0 + a ** 2) ** 3
    if rbf_type == 'Inverse Quadric':
        return _inner_prod_rbf_1_inverse_quadric(a, epsilon)
    if rbf_type == 'Cauchy':
        if aa == 0:
            return 2.0 / 3.0 * epsilon
        num = aa * (2.0 + aa) * (4.0 + 3.0 * aa * (2.0 + aa)) - 2.0 * (1.0 + aa) ** 2 * (
            4.0 + aa * (2.0 + aa)
        ) * log(1.0 + aa)
        den = aa ** 3 * (1.0 + aa) * (2.0 + aa) ** 3
        return 4.0 * epsilon * num / den
    raise ValueError(f"Unknown RBF type: {rbf_type}")


def _assemble_M(tau_vec, epsilon, derivative_order, rbf_type):
    """按导数阶数构造 RBF 正则化矩阵 (一阶/二阶), 支持均匀/非均匀网格"""
    inner = _inner_prod_rbf_1 if derivative_order == 1 else _inner_prod_rbf_2
    n_t = len(tau_vec)
    log_tau = log(tau_vec)

    if n_t > 1:
        d = np.diff(log_tau)
        if np.all(d > 0) and np.std(d) / np.mean(d) < 0.01:
            delta = np.mean(d)
            offsets = epsilon * np.arange(n_t) * delta
            row = np.array([inner(off, epsilon, rbf_type) for off in offsets])
            M = np.zeros((n_t, n_t))
            for i in range(n_t):
                for j in range(n_t):
                    M[i, j] = row[abs(i - j)]
            return M

    M = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(n_t):
            a = epsilon * (log_tau[i] - log_tau[j])
            M[i, j] = inner(a, epsilon, rbf_type)
    return M


def assemble_M(tau_vec, epsilon, derivative_order, rbf_type):
    """RBF 正则化矩阵分发函数 (derivative_order=1 或 2)"""
    if derivative_order == 1:
        return _assemble_M(tau_vec, epsilon, 1, rbf_type)
    return _assemble_M(tau_vec, epsilon, 2, rbf_type)


def x_to_gamma(x_vec, tau_map_vec, tau_vec, epsilon, rbf_type):
    """将 RBF 展开系数 x 映射回 gamma

    返回 (tau_map_vec, gamma_vec)
    """
    if rbf_type == 'Piecewise Linear':
        return tau_map_vec, np.asarray(x_vec)

    log_tau_map = log(np.asarray(tau_map_vec, dtype=np.float64))
    log_tau = log(np.asarray(tau_vec, dtype=np.float64))
    n_map = len(log_tau_map)
    n_t = len(log_tau)
    B = np.zeros((n_map, n_t))
    for p in range(n_map):
        B[p, :] = _rbf_value(log_tau_map[p] - log_tau, epsilon, rbf_type)
    gamma = B @ np.asarray(x_vec)
    return tau_map_vec, gamma
