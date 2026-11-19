"""
数值稳定性辅助函数
"""

import numpy as np
from numpy import (
    cosh,
    sinh,
    sqrt,
    tanh,
)


def coth(x):
    return 1 / tanh(x)


def safe_sinh(x):
    """
    原始直接调用:
    return sinh(x)
    """
    re_x = np.real(x)
    if re_x > 50:
        return 0.5 * np.exp(x)
    if re_x < -50:
        return -0.5 * np.exp(-x)
    if np.abs(x) < 1e-12:
        return x + x**3 / 6.0
    return sinh(x)


def safe_cosh(x):
    """
    原始直接调用:
    return cosh(x)
    """
    re_x = np.real(x)
    if re_x > 50:
        return 0.5 * np.exp(x)
    if re_x < -50:
        return 0.5 * np.exp(-x)
    if np.abs(x) < 1e-12:
        return 1.0 + x**2 / 2.0
    return cosh(x)


def safe_sinh_cosh(x):
    return safe_sinh(x), safe_cosh(x)


def safe_coth(x):
    """
    原始直接调用:
    return coth(x)
    """
    re_x = np.real(x)
    if re_x > 50:
        return 1.0 + 2.0 * np.exp(-2.0 * x)
    if re_x < -50:
        return -1.0 - 2.0 * np.exp(2.0 * x)
    if np.abs(x) < 1e-12:
        return 1.0 / x + x / 3.0
    return cosh(x) / sinh(x)
