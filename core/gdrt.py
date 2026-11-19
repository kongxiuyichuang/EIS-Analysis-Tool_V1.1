"""
gDRT (Generalized Distribution of Relaxation Times) 反演
支持同时处理 RC（容性）和 RL（感性）阻抗数据
基于 PWL 离散化 + 符号掩码核矩阵 + cvxopt QP 求解

移植自 resources/solve_DRT_with_L_revised.py
修改: τ 范围由 -Im(Z) 符号分布自适应确定，非固定依赖 f₀
"""

import numpy as np
import cvxopt
from cvxopt import matrix

from .rbf import assemble_A, assemble_M, compute_epsilon, x_to_gamma
from .lambda_selection import pick_lambda


def _calc_d_weights(tau_vec):
    N = len(tau_vec)
    d = np.zeros(N)
    d[0] = 0.5 * np.log(tau_vec[1] / tau_vec[0])
    d[-1] = 0.5 * np.log(tau_vec[-1] / tau_vec[-2])
    d[1:-1] = 0.5 * np.log(tau_vec[2:] / tau_vec[:-2])
    return d


def _build_pwl_regularization(tau_vec, derivative_order=2):
    """PWL 一阶/二阶差分正则化矩阵 (对数间距加权, 与 DRT/DCT/pyimpspec 一致)"""
    N_t = len(tau_vec)
    if derivative_order == 1:
        L = np.zeros((N_t - 1, N_t))
        for p in range(N_t - 1):
            d = np.log(tau_vec[p+1] / tau_vec[p])
            L[p, p] = -1.0 / d
            L[p, p+1] = 1.0 / d
    else:  # 2
        L = np.zeros((N_t - 2, N_t))
        for p in range(N_t - 2):
            d = np.log(tau_vec[p+1] / tau_vec[p])
            factor = 2.0 if (p == 0 or p == N_t - 3) else 1.0
            L[p, p]   = factor / (d*d)
            L[p, p+1] = factor * -2.0 / (d*d)
            L[p, p+2] = factor / (d*d)
    return L.T @ L


def _find_f0(f_sorted, Z_obs):
    for i in range(len(f_sorted) - 1):
        if np.imag(Z_obs[i]) > 0 and np.imag(Z_obs[i+1]) <= 0:
            im1, im2 = np.imag(Z_obs[i]), np.imag(Z_obs[i+1])
            f1, f2 = f_sorted[i], f_sorted[i+1]
            return f1 + (0 - im1) * (f2 - f1) / (im2 - im1)
    return f_sorted[0]


def solve_gdrt(f, Z, lam=1e-4, n_tau=500,
               rbf_type='Piecewise Linear',
               rbf_shape='FWHM Coefficient', rbf_coeff=0.5,
               tau_strategy='logspace',
               weight='boukamp',
               derivative_order=2,
               cross_validation=''):
    f = np.asarray(f, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.complex128)

    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(Z)):
        raise ValueError("Input data contains NaN or Inf values")

    sort_idx = np.argsort(f)[::-1]
    f_sorted = f[sort_idx]
    Z_obs = Z[sort_idx]
    omega = 2.0 * np.pi * f_sorted
    N_f = len(omega)

    f0 = _find_f0(f_sorted, Z_obs)

    imag_sign = -np.imag(Z_obs)
    f_rl = f_sorted[imag_sign < 0]
    f_rc = f_sorted[imag_sign > 0]

    if len(f_rl) < 2 or len(f_rc) < 2:
        raise ValueError("数据中电感区或电容区数据点不足，无法执行 gDRT")

    if tau_strategy == 'data':
        tau_rl_vec = np.sort(1.0 / f_rl)
        tau_rc_vec = np.sort(1.0 / f_rc)
        N_rl = len(tau_rl_vec)
        N_rc = len(tau_rc_vec)
        if N_rl < 3 or N_rc < 3:
            raise ValueError("电感区或电容区数据点过少，无法执行 gDRT")
    else:  # 'logspace'
        if np.min(f_rl) >= f0:
            tau_rl_max = 1.0 / (2.0 * np.pi * np.min(f0))
            tau_rl_min = 1.0 / (2.0 * np.pi * np.max(f_rl) * 10.0)
        else:
            tau_rl_max = 1.0 / (2.0 * np.pi * np.min(f_rl) / 10.0)
            tau_rl_min = 1.0 / (2.0 * np.pi * np.max(f0))
        tau_rl_vec = np.logspace(np.log10(tau_rl_min), np.log10(tau_rl_max), n_tau)
        N_rl = n_tau

        if np.max(f_rc) <= f0:
            tau_rc_max = 1.0 / (2.0 * np.pi * np.min(f_rc) / 10.0)
            tau_rc_min = 1.0 / (2.0 * np.pi * np.max(f0))
        else:
            tau_rc_max = 1.0 / (2.0 * np.pi * np.min(f0))
            tau_rc_min = 1.0 / (2.0 * np.pi * np.max(f_rc) * 10.0)
        tau_rc_vec = np.logspace(np.log10(tau_rc_min), np.log10(tau_rc_max), n_tau)
        N_rc = n_tau

    if rbf_type == 'Piecewise Linear':
        d_rl = _calc_d_weights(tau_rl_vec)
        d_rc = _calc_d_weights(tau_rc_vec)

        wt_rl = np.outer(omega, tau_rl_vec)
        Are_rl = (wt_rl**2 / (1.0 + wt_rl**2)) * d_rl
        Aim_rl = (wt_rl / (1.0 + wt_rl**2)) * d_rl

        wt_rc = np.outer(omega, tau_rc_vec)
        Are_rc = (1.0 / (1.0 + wt_rc**2)) * d_rc
        Aim_rc = (-wt_rc / (1.0 + wt_rc**2)) * d_rc

        M_sub_rl = _build_pwl_regularization(tau_rl_vec, derivative_order)
        M_sub_rc = _build_pwl_regularization(tau_rc_vec, derivative_order)
    else:
        eps_rl = compute_epsilon(tau_rl_vec, rbf_coeff, rbf_type, rbf_shape)
        eps_rc = compute_epsilon(tau_rc_vec, rbf_coeff, rbf_type, rbf_shape)
        Are_rl, Aim_rl = assemble_A(f_sorted, tau_rl_vec, eps_rl, rbf_type, kernel='dct')
        Are_rc, Aim_rc = assemble_A(f_sorted, tau_rc_vec, eps_rc, rbf_type, kernel='drt')
        M_sub_rl = assemble_M(tau_rl_vec, eps_rl, derivative_order, rbf_type)
        M_sub_rc = assemble_M(tau_rc_vec, eps_rc, derivative_order, rbf_type)

    zero_col = np.zeros((N_f, 1))
    one_col = np.ones((N_f, 1))
    omega_col = omega.reshape(-1, 1)
    inv_omega_col = (-1.0 / omega).reshape(-1, 1)

    A_re = np.hstack([one_col, zero_col, zero_col, Are_rc, Are_rl])
    A_im = np.hstack([zero_col, inv_omega_col, omega_col, Aim_rc, Aim_rl])
    H_mat = np.vstack([A_re, A_im])

    n_total = 3 + N_rc + N_rl
    M_full = np.zeros((n_total, n_total))
    M_full[3:3+N_rc, 3:3+N_rc] = M_sub_rc
    M_full[3+N_rc:, 3+N_rc:] = M_sub_rl

    absZ = np.maximum(np.abs(Z_obs), 1e-15)
    if weight == 'none':
        w_sq = np.ones(2 * N_f)
    else:  # 'boukamp'
        w_sq = np.tile(1.0 / (absZ * absZ), 2)
    Z_vec = np.concatenate([np.real(Z_obs), np.imag(Z_obs)])

    if cross_validation:
        lam = pick_lambda(A_re, A_im, np.real(Z_obs), np.imag(Z_obs),
                          M_full, lam, cross_validation)

    P = 2.0 * ((H_mat.T * w_sq) @ H_mat + lam * M_full + 1e-12 * np.eye(n_total))
    P = (P + P.T) / 2.0
    q_vec = -2.0 * ((Z_vec * w_sq) @ H_mat)

    for factor in [1, 10, 100, 1000]:
        lam_eff = lam * factor
        P_reg = P.copy()
        P_reg += 2.0 * (lam_eff - lam) * M_full
        P_reg = (P_reg + P_reg.T) / 2.0
        try:
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['abstol'] = 1e-12
            cvxopt.solvers.options['reltol'] = 1e-12
            cvxopt.solvers.options['feastol'] = 1e-12
            cvxopt.solvers.options['maxiters'] = 500

            sol = cvxopt.solvers.qp(
                matrix(P_reg), matrix(q_vec.astype(np.float64)),
                matrix(-np.eye(n_total)), matrix(np.zeros(n_total))
            )
            if sol['status'] == 'optimal':
                x_sol = np.asarray(sol['x']).flatten()
                break
        except Exception:
            continue
    else:
        raise RuntimeError(f"gDRT QP solver failed after multiple attempts (lam_eff={lam_eff:.1e})")

    if rbf_type == 'Piecewise Linear':
        out_tau_rc = tau_rc_vec
        out_tau_rl = tau_rl_vec
        val_rc = x_sol[3:3+N_rc]
        val_rl = x_sol[3+N_rc:]
    else:
        eps_rl = compute_epsilon(tau_rl_vec, rbf_coeff, rbf_type, rbf_shape)
        eps_rc = compute_epsilon(tau_rc_vec, rbf_coeff, rbf_type, rbf_shape)
        if tau_strategy == 'data':
            tau_fine_rc = np.logspace(
                np.log10(tau_rc_vec.min()) - 0.5,
                np.log10(tau_rc_vec.max()) + 0.5,
                10 * N_rc)
            tau_fine_rl = np.logspace(
                np.log10(tau_rl_vec.min()) - 0.5,
                np.log10(tau_rl_vec.max()) + 0.5,
                10 * N_rl)
            out_tau_rc, val_rc = x_to_gamma(x_sol[3:3+N_rc], tau_fine_rc, tau_rc_vec, eps_rc, rbf_type)
            out_tau_rl, val_rl = x_to_gamma(x_sol[3+N_rc:], tau_fine_rl, tau_rl_vec, eps_rl, rbf_type)
        else:
            out_tau_rc, val_rc = x_to_gamma(x_sol[3:3+N_rc], tau_rc_vec, tau_rc_vec, eps_rc, rbf_type)
            out_tau_rl, val_rl = x_to_gamma(x_sol[3+N_rc:], tau_rl_vec, tau_rl_vec, eps_rl, rbf_type)

    gamma = {
        'tau_rc': out_tau_rc,
        'val_rc': val_rc,
        'tau_rl': out_tau_rl,
        'val_rl': val_rl,
        'R0': x_sol[0],
        'C': 1.0 / x_sol[1] if x_sol[1] > 1e-12 else np.inf,
        'L0': x_sol[2],
        'f0': f0,
    }

    Z_est_raw = H_mat @ x_sol
    Z_est = Z_est_raw[:N_f] + 1j * Z_est_raw[N_f:]

    res_re = (np.real(Z_obs) - np.real(Z_est)) / absZ
    res_im = (np.imag(Z_obs) - np.imag(Z_est)) / absZ
    chi2 = np.mean(res_re**2 + res_im**2)

    return {
        'gamma': gamma,
        'Z_recon': Z_est,
        'f_sorted': f_sorted,
        'res_re': res_re,
        'res_im': res_im,
        'chi2': chi2,
        'lambda_used': lam_eff,
        'n_tau_rc': N_rc,
        'n_tau_rl': N_rl,
        'n_tau_used': N_rc + N_rl,
    }
