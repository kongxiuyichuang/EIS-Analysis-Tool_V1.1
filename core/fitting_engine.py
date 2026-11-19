"""
拟合引擎：基于 scipy.least_squares 的参数辨识
"""

import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class FitParams:
    names: list
    initial: list
    lower: list
    upper: list
    fixed: list

    @property
    def free_count(self):
        return sum(1 for f in self.fixed if not f)

    def get_free(self):
        vals, lbs, ubs, idx = [], [], [], []
        for i, (v, lo, hi, fx) in enumerate(zip(self.initial, self.lower, self.upper, self.fixed)):
            if not fx:
                vals.append(v)
                lbs.append(lo)
                ubs.append(hi)
                idx.append(i)
        return vals, lbs, ubs, idx


@dataclass
class FitResult:
    success: bool
    message: str
    params: dict
    z_fit: np.ndarray
    res_re: np.ndarray
    res_im: np.ndarray
    stats: dict = field(default_factory=dict)


class FittingEngine:
    def __init__(self, model, f_arr, z_exp):
        self.model = model
        self.f_arr = f_arr
        self.z_exp = z_exp

    def run(self, fit_params: FitParams) -> FitResult:
        free_vals, lb_list, ub_list, free_idx = fit_params.get_free()

        if not free_vals:
            return FitResult(
                success=False, message="No free parameters to fit",
                params={}, z_fit=np.array([]),
                res_re=np.array([]), res_im=np.array([])
            )

        all_init = list(fit_params.initial)

        def full_params(x_free):
            p = list(all_init)
            for idx, val in zip(free_idx, x_free):
                p[idx] = val
            return dict(zip(fit_params.names, p))

        def objective(x_free):
            p_full = full_params(x_free)
            z_model = self.model.compute_Z(self.f_arr, p_full)
            res = self.z_exp - z_model
            wgt = 1.0 / np.maximum(np.abs(self.z_exp), 1e-15)
            return np.concatenate([np.real(res) * wgt, np.imag(res) * wgt])

        try:
            result = least_squares(
                objective, free_vals,
                bounds=(lb_list, ub_list),
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=8000, method='trf'
            )

            final_p = full_params(result.x)
            z_fit = self.model.compute_Z(self.f_arr, final_p)

            res_re = np.real(self.z_exp - z_fit)
            res_im = np.imag(self.z_exp - z_fit)

            wgt = 1.0 / np.maximum(np.abs(self.z_exp), 1e-15)
            chi_sq = np.sum((res_re * wgt)**2 + (res_im * wgt)**2)
            ss_res = np.sum(res_re**2 + res_im**2)
            ss_tot = np.sum(
                (np.real(self.z_exp) - np.mean(np.real(self.z_exp)))**2 +
                (np.imag(self.z_exp) - np.mean(np.imag(self.z_exp)))**2
            )
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            n_data = 2 * len(self.f_arr)
            n_par = len(free_vals)
            dof = max(n_data - n_par, 1)
            chi_red = chi_sq / dof

            stats = {
                'r2': r2, 'chi_sq': chi_sq, 'chi_red': chi_red,
                'cost': result.cost, 'nfev': result.nfev,
                'status': result.message, 'n_param': n_par, 'dof': dof,
            }

            return FitResult(
                success=True,
                message=result.message,
                params=final_p,
                z_fit=z_fit,
                res_re=res_re,
                res_im=res_im,
                stats=stats,
            )

        except Exception as e:
            return FitResult(
                success=False, message=str(e),
                params={}, z_fit=np.array([]),
                res_re=np.array([]), res_im=np.array([])
            )
