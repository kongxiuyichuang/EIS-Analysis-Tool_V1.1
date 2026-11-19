"""
DRT-DOP / DRT-DDT / 纯 DOP / 纯 DDT 求解门面 (EIS 频域)

包装 core.hybdrt.models.DRT 的 fit_eis, 返回纯 dict 供 GUI/上层使用。
"""

import ast
import numpy as np
import warnings

from .models import DRT


def parse_grid_spec(text, p_grid=False):
    """解析并校验用户输入的网格序列

    接受格式:
      "np.linspace(0.1,0.5,50)"  /  "linspace(0.1, 0.5, 50)"
      "np.logspace(-4,4,200)"    /  "logspace(-4, 4, 200)"
      "0.1,0.2,0.3,..."          (逗号分隔数值列表)
      空字符串 -> 返回 None

    校验: 有限、严格递增、长度 >= 2; p 网格范围须在 [0, 0.5]。
    实现使用 ast 白名单, 仅允许 np.linspace/logspace/数值/负号/逗号。
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    allowed_names = {'np', 'linspace', 'logspace', 'arange'}
    try:
        tree = ast.parse(text, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise ValueError(f"不允许的标识符: {node.id}")
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id not in ('linspace', 'logspace', 'arange'):
                    raise ValueError(f"不允许的函数: {fn.id}")
                if isinstance(fn, ast.Attribute) and fn.attr not in ('linspace', 'logspace', 'arange'):
                    raise ValueError(f"不允许的函数: {fn.attr}")
        val = eval(compile(tree, '<grid>', 'eval'),
                   {'np': np, 'linspace': np.linspace, 'logspace': np.logspace,
                    'arange': np.arange}, {})
    except Exception as e:
        raise ValueError(f"网格格式无效: {e}\n"
                         f"示例: np.linspace(0.1, 0.5, 50) 或 0.1,0.2,0.3,...")

    arr = np.asarray(val, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("网格须为一维且至少 2 个点")
    if not np.all(np.isfinite(arr)):
        raise ValueError("网格包含非有限值")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("网格须严格递增")
    if p_grid:
        if arr.min() < 0.0 or arr.max() > 0.5:
            raise ValueError("p 网格范围须在 [0, 0.5] 内 (异常扩散指数)")
    return arr


def _make_tau_grid(f, ppd=10, extend_decades=1):
    """自动生成 DRT 的 log 均匀 tau 网格 (数据频率 ± extend 个数量级)"""
    from .preprocessing import get_basis_tau
    return get_basis_tau(np.asarray(f, dtype=float), None, None,
                         ppd=ppd, extend_decades=extend_decades)


def solve_drt_dop_ddt(f, Z,
                      method='DRT-DDT',
                      fit_ohmic=True,
                      fit_inductance=False,
                      fit_capacitance=False,
                      ddt_kernel_type=1,
                      fit_ddt_2d=False,
                      ddt_p_grid=None,
                      tau_basis_type='gaussian',
                      nu_basis_type='gaussian',
                      tau_dd_basis_type='gaussian',
                      basis_tau_ppd=10,
                      extend_decades=1,
                      tau_dd_ppd=10,
                      fixed_basis_tau=None,
                      fixed_basis_tau_dd=None,
                      l2_lambda_0=140.0,
                      ddt_l2_lambda_0=1.0,
                      dop_l2_lambda_0=10.0,
                      rp_scale=50.0,
                      max_iter=30,
                      fit_ddt_p=False,
                      ddt_p_bounds=(0.05, 0.5),
                      nonneg=True,
                      warn=False,
                      progress_callback=None):
    """DRT-DOP / DRT-DDT / DOP / DDT 联合或单独求解 (EIS 频域)

    返回 dict:
      method, gamma, dop, ddt, R_inf, inductance, C_inv, ddt_p,
      Z_recon, f_sorted, res_re, res_im, chi2, fit_parameters, basis grids
    """
    method = method.upper()
    valid = {'DRT-DOP', 'DRT-DDT', 'DOP', 'DDT'}
    if method not in valid:
        raise ValueError(f"Unknown method: {method}")

    fit_drt = 'DRT' in method
    fit_dop = 'DOP' in method
    fit_ddt = 'DDT' in method

    # fit_ddt_p 与 fit_ddt_2d 互斥
    if fit_ddt and fit_ddt_p and fit_ddt_2d:
        raise ValueError("fit_ddt_p 与 fit_ddt_2d 互斥，不能同时为 True")

    # 1D / 标准核模式下忽略 p 网格, 避免误用
    if not fit_ddt_2d:
        ddt_p_grid = None
    elif ddt_p_grid is None:
        # 2D 异常 DDT 默认 p 网格
        ddt_p_grid = np.linspace(0.1, 0.5, 50)

    # p 网格解析校验
    if ddt_p_grid is not None:
        if isinstance(ddt_p_grid, str):
            ddt_p_grid = parse_grid_spec(ddt_p_grid, p_grid=True)
        else:
            ddt_p_grid = np.asarray(ddt_p_grid, dtype=float)
            if fit_ddt_2d and not (np.all(np.isfinite(ddt_p_grid))
                                   and np.all(np.diff(ddt_p_grid) > 0)
                                   and 0.0 <= ddt_p_grid.min() <= ddt_p_grid.max() <= 0.5):
                raise ValueError("ddt_p_grid 须为递增的有限数组, 范围 [0,0.5]")

    # 自动生成固定 DRT 网格 (用户选"自动生成"; 若提供 fixed_basis_tau 则使用)
    if fixed_basis_tau is None:
        fixed_basis_tau = _make_tau_grid(f, basis_tau_ppd, extend_decades)

    try:
        model = DRT(
            fit_inductance=fit_inductance, warn=warn, fit_ohmic=fit_ohmic,
            fit_capacitance=fit_capacitance,
            fit_drt=fit_drt, fit_dop=fit_dop, fit_ddt=fit_ddt,
            tau_basis_type=tau_basis_type, fixed_basis_tau=fixed_basis_tau,
            nu_basis_type=nu_basis_type,
            tau_dd_basis_type=tau_dd_basis_type,
            fixed_basis_tau_dd=fixed_basis_tau_dd,
            ddt_kernel_type=ddt_kernel_type,
            fit_ddt_p=fit_ddt_p, ddt_p_bounds=ddt_p_bounds,
            ddt_p_grid=ddt_p_grid, fit_ddt_2d=fit_ddt_2d,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit_kwargs = dict(rp_scale=rp_scale, l2_lambda_0=l2_lambda_0,
                              max_iter=max_iter)
            if progress_callback is not None:
                fit_kwargs['progress_callback'] = progress_callback
            if fit_ddt:
                fit_kwargs['ddt_l2_lambda_0'] = ddt_l2_lambda_0
            if fit_dop:
                fit_kwargs['dop_l2_lambda_0'] = dop_l2_lambda_0
            model.fit_eis(
                np.asarray(f, dtype=float), np.asarray(Z, dtype=complex),
                nonneg=nonneg, **fit_kwargs,
            )
    except Exception as e:
        raise RuntimeError(f"DRT-DOP/DDT 求解失败: {e}")

    fp = model.fit_parameters

    # 重建阻抗
    Z_recon = model.predict_z(model.get_fit_frequencies() if model.get_fit_frequencies() is not None else f)
    f_sorted = model.f_fit if len(model.f_fit) > 0 else np.asarray(f, dtype=float)

    absZ = np.maximum(np.abs(np.asarray(Z, dtype=complex)[:len(f_sorted)]), 1e-15)
    res_re = (np.real(Z)[:len(f_sorted)] - np.real(Z_recon)) / absZ
    res_im = (np.imag(Z)[:len(f_sorted)] - np.imag(Z_recon)) / absZ
    chi2 = np.mean(res_re**2 + res_im**2)

    result = {
        'method': method,
        'Z_recon': Z_recon,
        'f_sorted': f_sorted,
        'res_re': res_re,
        'res_im': res_im,
        'chi2': chi2,
        'R_inf': fp.get('R_inf', 0.0),
        'inductance': fp.get('inductance', 0.0),
        'C_inv': fp.get('C_inv', 0.0),
        'ddt_p': fp.get('ddt_p', 0.5),
        'lambda_used': None,
        'basis_tau': model.basis_tau,
        'fit_parameters': fp,
        '_model': model,
    }

    # DRT 分布
    if fit_drt and model.basis_tau is not None:
        tau = model.get_tau_eval(20)
        gamma = model.predict_drt(tau)
        result['gamma'] = {'tau': tau, 'val': gamma}

    # DOP 分布
    if fit_dop and model.basis_nu is not None:
        nu, dop = model.predict_dop(return_nu=True)
        result['dop'] = {'nu': nu, 'rho': dop}

    # DDT 分布 (1D 或 2D)
    if fit_ddt and model.basis_tau_dd is not None:
        if model.basis_p is None:
            td, q = model.predict_ddt(return_tau_dd=True, marginalize='tau')
            result['ddt'] = {'tau_dd': td, 'q': q}
        else:
            td, p_grid, q2d = model.predict_ddt(return_tau_dd=True, return_p=True)
            q_tau = model.predict_ddt(tau_dd=td, marginalize='tau')
            result['ddt'] = {
                'tau_dd': td, 'p': np.asarray(p_grid), 'q2d': np.asarray(q2d),
                'q_tau': q_tau,
            }

    return result
