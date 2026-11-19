"""
正则化参数 λ 的自动选择 (交叉验证 / L 曲线)

移植自 pyimpspec (analysis/drt/tr_rbf.py 与 analysis/drt/utility.py)

参考:
    - GCV: G. Wahba, Ann. Statist. 13 (1985) 1378-1402
    - mGCV: Y.J. Kim, C. Gu, J. Royal Statist. Soc. 66 (2004) 337-356
    - L-curve: P.C. Hansen, D.P. O'Leary, SIAM J. Sci. Comput. 14 (1993) 1487-1503
    - Maradesa et al., J. Electrochem. Soc. 170 (2023) 030502
"""

import numpy as np
from numpy import exp, eye, log, trace
from numpy.linalg import cholesky, eigvals, inv, norm, LinAlgError, svd
from scipy.optimize import minimize

LAMBDA_MIN = 1e-7
LAMBDA_MAX = 1e0


def _is_positive_definite(matrix):
    try:
        cholesky(matrix)
        return True
    except LinAlgError:
        return False


def _nearest_positive_definite(A):
    """Higham 最近正定矩阵 (基于 John D'Errico 的 nearestSPD)"""
    B = (A + A.T) / 2.0
    _, Sigma_mat, V = svd(B)
    H = V.T @ np.diag(Sigma_mat) @ V
    A_nPD = (B + H) / 2.0
    A_symm = (A_nPD + A_nPD.T) / 2.0

    k = 1
    I = np.eye(A_symm.shape[0])
    while not _is_positive_definite(A_symm):
        eps = np.spacing(norm(A_symm))
        min_eig = min(0.0, np.min(np.real(eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1
    return A_symm


def _gcv_score(ln_lambda, A_re, A_im, Z_re, Z_im, M):
    """广义交叉验证 (GCV)"""
    lam = exp(ln_lambda)
    A = np.concatenate((A_re, A_im), axis=0)
    Z = np.concatenate((Z_re, Z_im), axis=0)
    n_cv = Z.shape[0]

    A_agm = A.T @ A + lam * M
    if not _is_positive_definite(A_agm):
        A_agm = _nearest_positive_definite(A_agm)
    L_agm = cholesky(A_agm)
    inv_A_agm = inv(L_agm).T @ inv(L_agm)
    A_GCV = A @ inv_A_agm @ A.T

    num = (norm((eye(n_cv) - A_GCV) @ Z) ** 2) / n_cv
    den = (trace(eye(n_cv) - A_GCV) / n_cv) ** 2
    return num / den


def _mgcv_score(ln_lambda, A_re, A_im, Z_re, Z_im, M):
    """修正 GCV (mGCV)"""
    lam = exp(ln_lambda)
    A = np.concatenate((A_re, A_im), axis=0)
    Z = np.concatenate((Z_re, Z_im), axis=0)
    n_cv = Z.shape[0]

    A_agm = A.T @ A + lam * M
    if not _is_positive_definite(A_agm):
        A_agm = _nearest_positive_definite(A_agm)
    L_agm = cholesky(A_agm)
    inv_A_agm = inv(L_agm).T @ inv(L_agm)
    A_GCV = A @ inv_A_agm @ A.T

    rho = 2.0 if Z_re.shape[0] >= 50 else 1.3
    num = (norm((eye(n_cv) - A_GCV) @ Z) ** 2) / n_cv
    den = (trace(eye(n_cv) - rho * A_GCV) / n_cv) ** 2
    return num / den


def _lc_score(ln_lambda, A_re, A_im, Z_re, Z_im, M):
    """L 曲线 (LC)"""
    lam = exp(ln_lambda)
    A = np.concatenate((A_re, A_im), axis=0)
    Z = np.concatenate((Z_re, Z_im), axis=0)

    A_agm = A.T @ A + lam * M
    if not _is_positive_definite(A_agm):
        A_agm = _nearest_positive_definite(A_agm)
    L_agm = cholesky(A_agm)
    inv_A_agm = inv(L_agm).T @ inv(L_agm)
    A_LC = A @ ((inv_A_agm.T @ inv_A_agm) @ inv_A_agm) @ A.T
    eta_num = Z.T @ A_LC @ Z

    A_agm_d = A @ A.T + lam * eye(A.shape[0])
    if not _is_positive_definite(A_agm_d):
        A_agm_d = _nearest_positive_definite(A_agm_d)
    L_agm_d = cholesky(A_agm_d)
    inv_A_agm_d = inv(L_agm_d).T @ inv(L_agm_d)
    eta_denom = lam * Z.T @ (inv_A_agm_d.T @ inv_A_agm_d) @ Z

    eta_prime = eta_num / eta_denom
    theta_num = eta_num
    A_LC_d = A @ (inv_A_agm.T @ inv_A_agm) @ A.T
    theta_denom = Z.T @ A_LC_d @ Z
    theta_prime = -theta_num / theta_denom

    a_sq = (eta_num / (eta_denom * theta_denom)) ** 2
    p = (Z.T @ (inv_A_agm_d.T @ inv_A_agm_d) @ Z) * theta_denom
    m = (2 * lam * Z.T @ ((inv_A_agm_d.T @ inv_A_agm_d) @ inv_A_agm_d) @ Z) * theta_denom
    q = (2 * lam * Z.T @ (inv_A_agm_d.T @ inv_A_agm_d) @ Z) * eta_num
    LC_num = a_sq * (p + m - q)
    LC_denom = (eta_prime**2 + theta_prime**2) ** 1.5
    return -LC_num / LC_denom


_SCORES = {
    'gcv': _gcv_score,
    'mgcv': _mgcv_score,
    'lc': _lc_score,
}


def pick_lambda(A_re, A_im, Z_re, Z_im, M, lambda_0, method):
    """通过交叉验证/L曲线在 [1e-7, 1e0] 内自动选择最优 λ

    lambda_0 作为优化初值 (会被钳制到可行区间内)
    """
    if method not in _SCORES:
        raise ValueError(f"Unknown cross-validation method: {method}")

    lam0 = float(np.clip(lambda_0, LAMBDA_MIN, LAMBDA_MAX))
    result = minimize(
        _SCORES[method],
        log(lam0),
        args=(A_re, A_im, Z_re, Z_im, M),
        method='SLSQP',
        bounds=[(log(LAMBDA_MIN), log(LAMBDA_MAX))],
        options={'disp': False, 'maxiter': 2000},
    )
    return float(exp(result.x)[0])
