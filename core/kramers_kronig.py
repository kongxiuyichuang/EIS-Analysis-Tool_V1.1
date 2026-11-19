"""
Kramers-Kronig 校验算法
基于 Schönleber et al. (2014) 的 Lin-KK 方法，
拟合串联 R-(RC)_n 等效电路，可选串联电容和电感。

改进:
- 全范围 + 多准则评分模式 (替代正向递增后停止)
- 数值稳定性: np.linalg.lstsq 替代 np.linalg.solve
- log_F_ext 时间常数范围扩展
- 虚部拟合 R0 后验估计: 减去 RC/C/L 贡献
- 实部拟合两步 C/L 校正
- 修复 except LinAlgError 导致 mu 值陈旧的 bug
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class KramersKronigResult:
    frequencies: np.ndarray
    impedances_exp: np.ndarray
    impedances_fit: np.ndarray
    residuals_real: np.ndarray
    residuals_imag: np.ndarray
    pseudo_chisqr: float
    num_RC: int
    mu: float
    time_constants: np.ndarray
    series_R: float
    RC_R: np.ndarray
    series_L: float = 0.0
    series_C: float = 0.0
    log_F_ext: float = 0.0
    test: str = "complex"
    add_cap: bool = False
    add_ind: bool = True


def _generate_time_constants(f, num_RC: int, log_F_ext: float = 0.0) -> np.ndarray:
    F_ext = 10.0 ** log_F_ext
    omega_min = 2.0 * np.pi * np.min(f)
    omega_max = 2.0 * np.pi * np.max(f)
    t_max = F_ext / omega_min
    t_min = 1.0 / (omega_max * F_ext)
    if num_RC == 1:
        return np.array([np.sqrt(t_min * t_max)])
    indices = np.arange(1, num_RC + 1)
    return 10.0 ** (
        np.log10(t_min)
        + (indices - 1) / (num_RC - 1) * np.log10(t_max / t_min)
    )


def _calc_mu(Rs: np.ndarray) -> float:
    neg_sum = sum(abs(x) for x in Rs if x < 0)
    pos_sum = sum(abs(x) for x in Rs if x >= 0)
    if pos_sum == 0:
        return 1.0
    return 1.0 - neg_sum / pos_sum


def _build_matrices(f, Z, num_RC: int, time_constants: np.ndarray,
                    add_cap: bool = False, add_ind: bool = True,
                    exclude_cl: bool = False):
    omega = 2.0 * np.pi * f
    N = len(f)
    n_cols = num_RC + 1

    if not exclude_cl:
        if add_ind:
            n_cols += 1
        if add_cap:
            n_cols += 1

    A_re = np.zeros((N, n_cols), dtype=np.float64)
    A_im = np.zeros((N, n_cols), dtype=np.float64)
    absZ = np.maximum(np.abs(Z), 1e-15)

    A_re[:, 0] = 1.0 / absZ
    A_im[:, 0] = 0.0

    col = 1
    for k in range(num_RC):
        tau = time_constants[k]
        K_val = 1.0 / (1.0 + 1j * omega * tau)
        A_re[:, col] = np.real(K_val) / absZ
        A_im[:, col] = np.imag(K_val) / absZ
        col += 1

    if not exclude_cl:
        if add_cap:
            A_re[:, col] = 0.0
            A_im[:, col] = -1.0 / (omega * absZ)
            col += 1
        if add_ind:
            A_re[:, col] = 0.0
            A_im[:, col] = omega / absZ
            col += 1

    return A_re, A_im


def _fit_complex(A_re, A_im, Z):
    absZ = np.maximum(np.abs(Z), 1e-15)
    b_re = np.real(Z) / absZ
    b_im = np.imag(Z) / absZ
    A = np.vstack([A_re, A_im])
    b = np.concatenate([b_re, b_im])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _fit_real(A_re, Z, f, time_constants, add_cap, add_ind):
    absZ = np.maximum(np.abs(Z), 1e-15)
    M = len(time_constants)
    elements_partial = np.linalg.lstsq(A_re, np.real(Z) / absZ, rcond=None)[0]

    n_cols = 1 + M
    if add_cap:
        n_cols += 1
    if add_ind:
        n_cols += 1
    elements = np.zeros(n_cols)
    elements[0] = elements_partial[0]
    elements[1:1 + M] = elements_partial[1:]

    Z_fit_partial = _eval_circuit(elements, f, time_constants, add_cap, add_ind)
    res_imag = np.imag(Z) - np.imag(Z_fit_partial)

    omega = 2.0 * np.pi * f
    n_cl = 0
    if add_cap:
        n_cl += 1
    if add_ind:
        n_cl += 1

    if n_cl > 0:
        A_cl = np.zeros((len(f), n_cl), dtype=np.float64)
        j = 0
        if add_cap:
            A_cl[:, j] = -1.0 / (omega * absZ)
            j += 1
        if add_ind:
            A_cl[:, j] = omega / absZ

        cl_vals = np.linalg.lstsq(A_cl, res_imag / absZ, rcond=None)[0]
        j = 0
        col = 1 + M
        if add_cap:
            elements[col] = cl_vals[j]
            col += 1
            j += 1
        if add_ind:
            elements[col] = cl_vals[j]

    return elements


def _fit_imaginary(A_im, Z, f, time_constants, add_cap, add_ind):
    absZ = np.maximum(np.abs(Z), 1e-15)
    elements = np.linalg.lstsq(A_im, np.imag(Z) / absZ, rcond=None)[0]

    elements_noR0 = elements.copy()
    elements_noR0[0] = 0.0
    Z_fit_noR0 = _eval_circuit(elements_noR0, f, time_constants, add_cap, add_ind)

    ws = 1.0 / (np.real(Z)**2 + np.imag(Z)**2 + 1e-30)
    elements[0] = np.sum(ws * (np.real(Z) - np.real(Z_fit_noR0))) / np.sum(ws)
    return elements


def _eval_circuit(elements, f, time_constants, add_cap=False, add_ind=True):
    omega = 2.0 * np.pi * f
    Z = np.zeros(len(f), dtype=np.complex128)
    Z += elements[0]

    col = 1
    for k, tau in enumerate(time_constants):
        Z += elements[col] / (1.0 + 1j * omega * tau)
        col += 1

    if add_cap:
        if abs(elements[col]) > 1e-30:
            Z += elements[col] / (1j * omega)
        col += 1
    if add_ind:
        Z += 1j * omega * elements[col]
        col += 1

    return Z


def _calculate_pseudo_chisqr(Z_exp, Z_fit) -> float:
    res = Z_exp - Z_fit
    w = 1.0 / np.maximum(np.abs(Z_exp), 1e-15)
    return float(np.sum((w * np.real(res))**2 + (w * np.imag(res))**2))


def _perform_single_fit(f, Z, num_RC, taus, fit_type, add_cap, add_ind):
    if fit_type == 'real':
        A_re, _ = _build_matrices(f, Z, num_RC, taus,
                                   add_cap=add_cap, add_ind=add_ind,
                                   exclude_cl=True)
        return _fit_real(A_re, Z, f, taus, add_cap, add_ind)
    elif fit_type == 'imag':
        A_re, A_im = _build_matrices(f, Z, num_RC, taus,
                                      add_cap=add_cap, add_ind=add_ind)
        return _fit_imaginary(A_im, Z, f, taus, add_cap, add_ind)
    else:
        A_re, A_im = _build_matrices(f, Z, num_RC, taus,
                                      add_cap=add_cap, add_ind=add_ind)
        return _fit_complex(A_re, A_im, Z)


def perform_kk_test(f, Z, num_RC: int = 10, log_F_ext: float = 0.0,
                    fit_type: str = 'complex', add_cap: bool = False,
                    add_ind: bool = True, auto_mu: bool = True,
                    mu_cutoff: float = 0.85, max_M: int = 50):
    f = np.asarray(f, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.complex128)
    n_data = len(f)

    if auto_mu:
        max_M = min(max_M, n_data - 2)
        if max_M < 1:
            raise RuntimeError("K-K 校验失败: 数据点数不足")

        results_pool = []

        for M in range(1, max_M + 1):
            try:
                taus = _generate_time_constants(f, M, log_F_ext=log_F_ext)
                elements = _perform_single_fit(f, Z, M, taus, fit_type, add_cap, add_ind)

                RC_Rs = elements[1:1 + M]
                mu = _calc_mu(RC_Rs)
                Z_fit = _eval_circuit(elements, f, taus, add_cap, add_ind)
                res = (Z - Z_fit) / np.maximum(np.abs(Z), 1e-15)
                pc = _calculate_pseudo_chisqr(Z, Z_fit)

                results_pool.append(KramersKronigResult(
                    frequencies=f.copy(), impedances_exp=Z.copy(),
                    impedances_fit=Z_fit,
                    residuals_real=np.real(res), residuals_imag=np.imag(res),
                    pseudo_chisqr=pc, num_RC=M, mu=mu,
                    time_constants=taus, series_R=float(elements[0]),
                    RC_R=RC_Rs.copy(),
                    series_L=float(elements[-1]) if add_ind else 0.0,
                    series_C=float(elements[-2]) if add_cap else 0.0,
                    log_F_ext=log_F_ext,
                    test=fit_type, add_cap=add_cap, add_ind=add_ind,
                ))
            except np.linalg.LinAlgError:
                continue

        if not results_pool:
            raise RuntimeError("K-K 校验失败: 无法完成任何拟合")

        qualifying = [r for r in results_pool if r.mu > mu_cutoff]
        if qualifying:
            best = min(qualifying, key=lambda r: (r.pseudo_chisqr, r.num_RC))
        else:
            best = max(results_pool, key=lambda r: (r.mu, -r.pseudo_chisqr))
        return best

    else:
        taus = _generate_time_constants(f, num_RC, log_F_ext=log_F_ext)
        elements = _perform_single_fit(f, Z, num_RC, taus, fit_type, add_cap, add_ind)

        Z_fit = _eval_circuit(elements, f, taus, add_cap, add_ind)
        res = (Z - Z_fit) / np.maximum(np.abs(Z), 1e-15)
        pc = _calculate_pseudo_chisqr(Z, Z_fit)
        RC_Rs = elements[1:1 + num_RC]
        mu = _calc_mu(RC_Rs)

        return KramersKronigResult(
            frequencies=f.copy(), impedances_exp=Z.copy(),
            impedances_fit=Z_fit,
            residuals_real=np.real(res), residuals_imag=np.imag(res),
            pseudo_chisqr=pc, num_RC=num_RC, mu=mu,
            time_constants=taus, series_R=float(elements[0]),
            RC_R=RC_Rs.copy(),
            series_L=float(elements[-1]) if add_ind else 0.0,
            series_C=float(elements[-2]) if add_cap else 0.0,
            log_F_ext=log_F_ext,
            test=fit_type, add_cap=add_cap, add_ind=add_ind,
        )
