"""
基本元件阻抗计算模块
定义各标准元件的复数阻抗 Z(ω) 计算函数
"""

import numpy as np


def z_resistor(R: float) -> np.complex128:
    return np.complex128(R + 0j)


def z_cpe(C: float, p: float, omega: float) -> np.complex128:
    return 1.0 / (C * (1j * omega) ** p)


def z_inductor(L: float, omega: float) -> np.complex128:
    return 1j * omega * L


def _safe_coth(x):
    ax = np.abs(x)
    if ax < 1e-12:
        return 1.0 / x + x / 3.0
    if np.real(x) > 50:
        return 1.0 + 2.0 * np.exp(-2.0 * x)
    ep = np.exp(x)
    en = np.exp(-x)
    return (ep + en) / (ep - en)


def z_warburg_reflective(Rd: float, tau: float, p: float, omega: float) -> np.complex128:
    """Wo: 有限空间扩散 (反射边界)  Rd·coth((jωτ)^p)/((jωτ)^p)"""
    x = (1j * omega * tau) ** p
    if np.abs(x) < 1e-12:
        return np.complex128(Rd / (x * x))
    coth_val = _safe_coth(x)
    return np.complex128(Rd * coth_val / x)


def z_warburg_transmissive(Rd: float, tau: float, p: float, omega: float) -> np.complex128:
    """Ws: 有限长度扩散 (透射边界)  Rd·tanh((jωτ)^p)/((jωτ)^p)"""
    x = (1j * omega * tau) ** p
    if np.abs(x) < 1e-12:
        return np.complex128(Rd)
    if np.real(x) > 50:
        return np.complex128(Rd / x)
    return np.complex128(Rd * np.tanh(x) / x)


def z_semi_infinite_warburg(Aw: float, omega: float) -> np.complex128:
    """W: 半无限扩散  A_W / √(jω)"""
    return Aw / np.sqrt(1j * omega)


def z_havriliak_negami(dC: float, tau: float, a: float, b: float, omega: float) -> np.complex128:
    """H: ((1 + (jωτ)^a)^b) / (jω·dC)"""
    s = (1j * omega * tau) ** a
    return ((1.0 + s) ** b) / (1j * omega * dC)


def z_havriliak_negami_alt(R: float, tau: float, a: float, b: float, omega: float) -> np.complex128:
    """Ha: R / ((1 + (jωτ)^a)^b)"""
    s = (1j * omega * tau) ** a
    return R / ((1.0 + s) ** b)


def z_gerischer(Y: float, k: float, n: float, omega: float) -> np.complex128:
    """G: 1 / (Y * (k + jω)^n)"""
    return 1.0 / (Y * (k + 1j * omega) ** n)


def z_gerischer_alt(R: float, tau: float, n: float, omega: float) -> np.complex128:
    """Ga: R / ((1 + jωτ)^n)"""
    return R / ((1.0 + 1j * omega * tau) ** n)
