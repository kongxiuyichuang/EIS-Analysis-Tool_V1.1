"""
DRT (Distribution of Relaxation Times) 反演
基于 PWL 离散化 + cvxopt QP 求解

移植自 resources/solve_DRT.m
"""

import numpy as np
import cvxopt
from cvxopt import matrix

from .rbf import assemble_A, assemble_M, compute_epsilon, x_to_gamma
from .lambda_selection import pick_lambda


def _build_pwl_kernel(omega, tau_vec):
    N_f = len(omega)
    N_t = len(tau_vec)
    Are = np.zeros((N_f, N_t))
    Aim = np.zeros((N_f, N_t))
    for p in range(N_f):
        for q in range(N_t):
            if q == 0:
                delta = 0.5 * np.log(tau_vec[1] / tau_vec[0])
            elif q == N_t - 1:
                delta = 0.5 * np.log(tau_vec[-1] / tau_vec[-2])
            else:
                delta = 0.5 * np.log(tau_vec[q+1] / tau_vec[q-1])
            wt = omega[p] * tau_vec[q]
            Are[p, q] = (1.0 / (1.0 + wt*wt)) * delta
            Aim[p, q] = (-wt / (1.0 + wt*wt)) * delta
    return Are, Aim


def _build_pwl_regularization(tau_vec, derivative_order=2):
    """PWL 一阶/二阶差分正则化矩阵 (对数间距加权, 返回未填充 M_temp)"""
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


def _build_P_and_solve_qp(H_mat, w_sq, M_full, q_vec, n_vars, lam):
    for factor in [1, 10, 100, 1000]:
        lam_eff = lam * factor
        P = 2.0 * ((H_mat.T * w_sq) @ H_mat + lam_eff * M_full + 1e-12 * np.eye(n_vars))
        P = (P + P.T) / 2.0
        try:
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['abstol'] = 1e-10
            cvxopt.solvers.options['reltol'] = 1e-10
            cvxopt.solvers.options['feastol'] = 1e-10

            sol = cvxopt.solvers.qp(
                matrix(P), matrix(q_vec.astype(np.float64)),
                matrix(-np.eye(n_vars)), matrix(np.zeros(n_vars))
            )
            if sol['status'] == 'optimal':
                return np.asarray(sol['x']).flatten(), lam_eff
        except Exception:
            continue
    raise RuntimeError(f"QP solver failed after multiple attempts (final lam_eff={lam * 1000:.1e})")


def solve_drt(f, Z, lam=1e-3, n_tau=400,
              rbf_type='Piecewise Linear',
              rbf_shape='FWHM Coefficient', rbf_coeff=0.5,
              tau_strategy='logspace',
              weight='boukamp',
              add_capacitance=True,
              derivative_order=2,
              cross_validation=''):
    f = np.asarray(f, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.complex128)

    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(Z)):
        raise ValueError("Input data contains NaN or Inf values")

    idx = np.argsort(f)
    f_sorted = f[idx]
    omega = 2.0 * np.pi * f_sorted
    Z_obs = Z[idx]
    N_f = len(omega)

    if tau_strategy == 'data':
        tau_vec = np.sort(1.0 / f_sorted)
        N_t = len(tau_vec)
        if N_t < 3:
            raise ValueError("数据点过少，无法进行 DRT 反演")
    else:  # 'logspace'
        tau_min = 1.0 / (np.max(omega) * 10.0)
        tau_max = 1.0 / (np.min(omega) / 10.0)
        tau_vec = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        N_t = n_tau

    if rbf_type == 'Piecewise Linear':
        Are, Aim = _build_pwl_kernel(omega, tau_vec)
        M_temp = _build_pwl_regularization(tau_vec, derivative_order)
    else:
        epsilon = compute_epsilon(tau_vec, rbf_coeff, rbf_type, rbf_shape)
        Are, Aim = assemble_A(f_sorted, tau_vec, epsilon, rbf_type, kernel='drt')
        M_temp = assemble_M(tau_vec, epsilon, derivative_order, rbf_type)

    num_lead = 2 if add_capacitance else 1
    n_vars = N_t + num_lead
    if add_capacitance:
        A_full_re = np.column_stack([np.ones(N_f), np.zeros(N_f), Are])
        A_full_im = np.column_stack([np.zeros(N_f), -1.0 / omega, Aim])
    else:
        A_full_re = np.column_stack([np.ones(N_f), Are])
        A_full_im = np.column_stack([np.zeros(N_f), Aim])
    H_mat = np.vstack([A_full_re, A_full_im])

    M_full = np.zeros((n_vars, n_vars))
    M_full[num_lead:, num_lead:] = M_temp

    if weight == 'none':
        w_sq = np.ones(2 * N_f)
    else:  # 'boukamp'
        absZ = np.maximum(np.abs(Z_obs), 1e-15)
        w_sq = np.tile(1.0 / (absZ * absZ), 2)
    Z_vec = np.concatenate([np.real(Z_obs), np.imag(Z_obs)])

    if cross_validation:
        lam = pick_lambda(A_full_re, A_full_im, np.real(Z_obs), np.imag(Z_obs),
                          M_full, lam, cross_validation)

    q_vec = -2.0 * ((Z_vec * w_sq) @ H_mat)

    x_sol, lam_used = _build_P_and_solve_qp(H_mat, w_sq, M_full, q_vec, n_vars, lam)

    if rbf_type == 'Piecewise Linear':
        out_tau = tau_vec
        gamma_val = x_sol[num_lead:]
    else:
        epsilon = compute_epsilon(tau_vec, rbf_coeff, rbf_type, rbf_shape)
        if tau_strategy == 'data':
            tau_fine = np.logspace(
                np.log10(tau_vec.min()) - 0.5,
                np.log10(tau_vec.max()) + 0.5,
                10 * N_t)
            out_tau, gamma_val = x_to_gamma(x_sol[num_lead:], tau_fine, tau_vec, epsilon, rbf_type)
        else:
            out_tau, gamma_val = x_to_gamma(x_sol[num_lead:], tau_vec, tau_vec, epsilon, rbf_type)

    gamma = {'tau': out_tau, 'val': gamma_val,
             'R0': x_sol[0]}
    if add_capacitance:
        gamma['C'] = 1.0 / x_sol[1]

    absZ = np.maximum(np.abs(Z_obs), 1e-15)
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
        'lambda_used': lam_used,
        'n_tau_used': N_t,
    }
