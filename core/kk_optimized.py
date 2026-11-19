"""
优化 K-K 校验算法
基于 Plank et al. (2022) 和 Yrjana & Bobacka (2024) 的方法，
使用 Method 4 → Method 3 → Method 5 → chi2 流水线自动选择最优 RC 数量。

核心步骤:
  1. 对 M = 1..max_M 逐次执行 K-K 拟合
  2. 频率细分 + 曲率预计算
  3. Method 4 (符号变化次数) → 初筛候选
  4. Method 3 (曲率范数) → 缩小范围
  5. Method 5 (符号变化平均距离) → 缩小范围
  6. Pseudo chi2 → 决胜
"""

import numpy as np
from numpy.linalg import norm
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .kramers_kronig import (
    perform_kk_test,
    _generate_time_constants,
    _eval_circuit,
    KramersKronigResult,
)


# ─── 辅助函数 ──────────────────────────────────────────

def subdivide_frequencies(f, subdivision: int = 4):
    """在每对频率之间插入 subdivision 个对数等间距频率"""
    f = np.asarray(f, dtype=np.float64)
    new_f = [f[0]]
    for f1, f2 in zip(f[:-1], f[1:]):
        new_f.extend(
            np.logspace(np.log10(f1), np.log10(f2), subdivision + 2).tolist()[1:]
        )
    return np.array(new_f)


def calculate_curvatures(Z):
    """
    利用密切圆 (osculating circle) 估计阻抗谱的曲率
    参考: Plank et al. (2022)
    """
    Z = np.asarray(Z, dtype=np.complex128)
    N = Z.size
    nk = N - 2  # 曲率点数
    if nk < 1:
        return np.array([])

    # 向量 a = Zj - Zi, b = Zk - Zj, c = Zk - Zi
    a = np.column_stack([(Z[1:-1] - Z[:-2]).real, (Z[1:-1] - Z[:-2]).imag])
    b = np.column_stack([(Z[2:] - Z[1:-1]).real, (Z[2:] - Z[1:-1]).imag])
    c = np.column_stack([(Z[2:] - Z[:-2]).real, (Z[2:] - Z[:-2]).imag])

    a_dot_b = np.sum(a * b, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    c_norm = np.linalg.norm(c, axis=1)

    cos_alpha = np.clip(a_dot_b / np.maximum(a_norm * b_norm, 1e-30), -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    abs_kappa = 2.0 * np.sin(alpha) / np.maximum(c_norm, 1e-30)

    # 方向判定: 行列式符号 (Nyquist 图 -Z'' vs Z')
    matrix = np.ones((N, 3))
    matrix[:, 0] = Z.real
    matrix[:, 1] = -Z.imag
    det = np.array([np.linalg.det(matrix[i:i+3]) for i in range(nk)])

    return abs_kappa * np.sign(det)


def _count_sign_changes(kappas):
    """统计曲率符号变化的次数 (Method 4)"""
    if len(kappas) < 2:
        return 0
    signs = np.sign(kappas)
    prev = signs[0]
    n = 0
    for s in signs[1:]:
        if s == 0:
            continue
        if s != prev:
            n += 1
            prev = s
    return n


def _calc_sign_change_distances(kappas):
    """计算曲率符号变化之间的距离 (Method 5)"""
    if len(kappas) < 2:
        return np.array([len(kappas)], dtype=np.float64)
    signs = np.sign(kappas)
    distances = []
    prev_idx = 0
    prev_s = signs[0]
    for i in range(1, len(kappas)):
        if signs[i] == 0:
            continue
        if signs[i] != prev_s:
            distances.append(i - prev_idx)
            prev_s = signs[i]
            prev_idx = i
    if not distances:
        distances.append(len(kappas))
    return np.array(distances, dtype=np.float64)


# ─── Method 评分函数 ─────────────────────────────────────

def _method_4_scores(results: List[KramersKronigResult],
                     f_sub, curvatures) -> Dict[int, float]:
    """Method 4: 曲率符号变化次数 (越少越好) + pseudo chi2 偏移"""
    scores = {}
    chi2_vals = {r.num_RC: r.pseudo_chisqr for r in results}
    chi2_min = min(chi2_vals.values())
    chi2_max = max(chi2_vals.values()) - chi2_min

    for r in results:
        kappa = curvatures[r.num_RC]
        n_sign = _count_sign_changes(kappa)
        # pseudo chi2 偏移 (tiebreaker)
        offset = 0.0
        if chi2_max > 0:
            offset = 0.1 * (chi2_vals[r.num_RC] - chi2_min) / chi2_max
        scores[r.num_RC] = float(n_sign) + offset
    return scores


def _method_3_scores_direct(kappa_dict: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Method 3: 曲率范数 (越小越好) — 直接接受 {num_RC: kappa} 字典"""
    return {m: float(norm(k)) for m, k in kappa_dict.items()}


def _method_5_scores(results: List[KramersKronigResult],
                     f_sub, curvatures) -> Dict[int, float]:
    """Method 5: 符号变化平均距离 (越大越好)"""
    scores = {}
    m_orig = len(results[0].frequencies)
    n_sub = len(f_sub)
    for r in results:
        kappa = curvatures[r.num_RC]
        dist = _calc_sign_change_distances(kappa)
        mean_d = float(np.mean(dist)) if len(dist) > 0 else float(len(kappa))
        # 按原论文缩放
        scores[r.num_RC] = mean_d * m_orig / n_sub
    return scores


# ─── 下限估计 ──────────────────────────────────────────

def _estimate_lower_limit(results: List[KramersKronigResult]) -> int:
    """
    根据 log(χ²) vs num_RC 曲线的拐点估算 RC 数量下限。
    
    步骤 (参考 pyimpspec _approximate_transition_and_end_point):
      1. 找最大可用 RC: chi2 在 max_distance 个连续 M 内不再改善
      2. 剔除尾端数值不稳定段 (chi2 jitter)
      3. 拟合双折线，交点即为过渡点
    """
    if len(results) < 4:
        return max(1, results[0].num_RC)

    sorted_r = sorted(results, key=lambda r: r.num_RC)
    ns = np.array([r.num_RC for r in sorted_r], dtype=np.float64)
    log_chi2 = np.array([np.log10(max(r.pseudo_chisqr, 1e-30)) for r in sorted_r])

    # 步骤 1: 找 max_num_RC — chi2 改善停止点
    max_distance = 6
    lowest_idx = 0
    lowest_val = log_chi2[0]
    i = 0
    for i, y in enumerate(log_chi2):
        if y <= lowest_val:
            lowest_idx = i
            lowest_val = y
        elif (i - lowest_idx) >= max_distance:
            break

    # 步骤 2: 剔除尾端不稳定段
    while (log_chi2[i] - lowest_val) > 0.5:
        i -= 1
        if i <= 1:
            return int(ns[0])

    while i > 1 and (log_chi2[i - 1] - log_chi2[i]) > 1.0:
        i -= 1

    max_m = int(max(ns[:i + 1]))
    if max_m <= 2:
        return max_m

    # 步骤 3: 拟合双折线找交点
    x = ns[:i + 1]
    y = log_chi2[:i + 1]
    best_intercept = 0
    best_total = np.inf

    for k in range(2, len(x)):
        x1, y1 = x[:k], y[:k]
        x2, y2 = x[k:], y[k:]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s1, c1 = np.polyfit(x1, y1, 1)
            s2, c2 = np.polyfit(x2, y2, 1)

        r1 = np.sum((y1 - (s1 * x1 + c1))**2)
        r2 = np.sum((y2 - (s2 * x2 + c2))**2)
        # 惩罚: 第一段应该是负斜率 (骤降)
        penalty = 0.0
        if s1 > -0.02:
            penalty = 100.0 * s1 * s1
        total = r1 + r2 + penalty

        if total < best_total:
            best_total = total
            denom = s1 - s2
            if abs(denom) > 1e-12:
                best_intercept = int(np.ceil((c2 - c1) / denom))

    lower = max(1, min(best_intercept, max_m))
    if lower >= max_m:
        lower = max(1, max_m // 3)
    return lower


def _estimate_upper_limit(results: List[KramersKronigResult]) -> int:
    """根据 Method 5 平均距离阈值估计上限 (防止过拟合)"""
    if len(results) < 3:
        return max(r.num_RC for r in results)

    sorted_r = sorted(results, key=lambda r: r.num_RC)
    f_orig = sorted_r[0].frequencies
    f_sub = subdivide_frequencies(f_orig, subdivision=4)
    m_orig = len(f_orig)

    # 对所有结果计算 Method 5 得分
    scores = {}
    for r in sorted_r:
        taus = _generate_time_constants(f_orig, r.num_RC)
        elements, add_cap, add_ind = _extract_elements_info(r)
        Z_sub = _eval_circuit(elements, f_sub, taus, add_cap, add_ind)
        kappa = calculate_curvatures(Z_sub)
        dist = _calc_sign_change_distances(kappa)
        mean_d = float(np.mean(dist)) if len(dist) > 0 else float(len(kappa))
        scores[r.num_RC] = mean_d * m_orig / len(f_sub)

    # 找平均距离峰值后的下降: 距离 < 阈值 → 视为过拟合开始
    threshold = 4.0 * m_orig / len(f_sub)
    for r in reversed(sorted_r):
        if scores[r.num_RC] >= threshold:
            return r.num_RC
    return sorted_r[-1].num_RC


# ─── 流水线决策 (累积评分模式) ───────────────────────────

def suggest_optimal(results: List[KramersKronigResult],
                    lower_limit: int = 0,
                    upper_limit: int = 0) -> int:
    """
    使用 Method 4 → Method 3 → Method 5 → chi2 累积评分流水线

    累积评分模式:
      - 以 Method 4 符号变化次数为基础分
      - 逐步叠加归一化的 Method 3 / Method 5 / chi2 偏移量 (×0.1)
      - 最终选总分最低的 RC
      - 额外检查: 是否存在更低 RC 且 chi2 更好且符号变化 ≤ 当前
    """
    if lower_limit <= 0:
        lower_limit = _estimate_lower_limit(results)
    if upper_limit <= 0:
        upper_limit = min(_estimate_upper_limit(results), max(r.num_RC for r in results))
    upper_limit = max(upper_limit, lower_limit + 1)

    candidates = [r for r in results if lower_limit <= r.num_RC <= upper_limit]
    if len(candidates) == 0:
        return lower_limit
    if len(candidates) == 1:
        return candidates[0].num_RC

    # 1. 频率细分
    f_orig = candidates[0].frequencies
    f_sub = subdivide_frequencies(f_orig, subdivision=4)

    # 2. 预计算曲率
    curvatures = {}
    for r in candidates:
        taus = _generate_time_constants(f_orig, r.num_RC)
        elements, add_cap, add_ind = _extract_elements_info(r)
        Z_fit_sub = _eval_circuit(elements, f_sub, taus, add_cap, add_ind)
        curvatures[r.num_RC] = calculate_curvatures(Z_fit_sub)

    # 3. 计算各方法原始得分
    scores_4 = _method_4_scores(candidates, f_sub, curvatures)
    scores_3 = _method_3_scores_direct({r.num_RC: curvatures[r.num_RC] for r in candidates})
    scores_5 = _method_5_scores(candidates, f_sub, curvatures)
    log_chi2 = {r.num_RC: np.log10(max(r.pseudo_chisqr, 1e-30)) for r in candidates}

    # 4. 累积评分 (参考 pyimpspec _suggest_using_default)
    #    基础分 = 符号变化次数 (Method 4)
    modified = {r.num_RC: float(scores_4[r.num_RC]) for r in candidates}

    #    叠加归一化 Method 3 偏移 (曲率范数, 越小越好)
    min_v, max_v = min(scores_3.values()), max(scores_3.values())
    if max_v > min_v:
        for r in candidates:
            if r.num_RC in modified:
                modified[r.num_RC] += 0.1 * (scores_3[r.num_RC] - min_v) / (max_v - min_v)

    #    叠加归一化 Method 5 偏移 (距离, 越大越好 → 反转)
    s5_vals = {r.num_RC: scores_5[r.num_RC] for r in candidates}
    min_v5, max_v5 = min(s5_vals.values()), max(s5_vals.values())
    if max_v5 > min_v5:
        for r in candidates:
            if r.num_RC in modified:
                inv = 1.0 - (s5_vals[r.num_RC] - min_v5) / (max_v5 - min_v5)
                modified[r.num_RC] += 0.1 * inv

    #    叠加归一化 log(chi2) 偏移 (越小越好)
    if len(log_chi2) > 1:
        min_ch, max_ch = min(log_chi2.values()), max(log_chi2.values())
        if max_ch > min_ch:
            for r in candidates:
                modified[r.num_RC] += 0.1 * (log_chi2[r.num_RC] - min_ch) / (max_ch - min_ch)

    # 5. 用 np.isclose 找平局候选，选最低 RC
    min_score = min(modified.values())
    top_candidates = [m for m, s in modified.items() if np.isclose(float(s), float(min_score))]
    best_m = min(top_candidates)

    # 6. 最终检查: 是否存在更低 RC 且 chi2 更好 且 符号变化 ≤ 当前
    best_r = next(r for r in candidates if r.num_RC == best_m)
    best_chi2_log = log_chi2[best_m]
    best_nsign = scores_4[best_m]
    for r in sorted(candidates, key=lambda x: x.num_RC):
        if r.num_RC < best_m and log_chi2[r.num_RC] < best_chi2_log and scores_4[r.num_RC] <= best_nsign:
            best_m = r.num_RC
            break

    return best_m


def _extract_elements_info(result: KramersKronigResult):
    """从 K-K 结果中提取 elements 数组和配置信息"""
    n_rc = result.num_RC
    n_cols = 1 + n_rc
    if result.add_ind:
        n_cols += 1
    if result.add_cap:
        n_cols += 1
    elements = np.zeros(n_cols)
    elements[0] = result.series_R
    elements[1:1+n_rc] = result.RC_R
    col = 1 + n_rc
    if result.add_cap:
        elements[col] = result.series_C
        col += 1
    if result.add_ind:
        elements[col] = result.series_L
    return elements, result.add_cap, result.add_ind


# ─── 主入口 ─────────────────────────────────────────────

def perform_optimized_kk_test(f, Z, max_M: int = 100,
                              add_cap: bool = False,
                              add_ind: bool = True,
                              fit_type: str = 'complex',
                              lower_limit: int = 0) -> KramersKronigResult:
    """
    使用优化算法执行 K-K 校验:
    逐阶拟合 M = 1..max_M，然后用 Method 4→3→5→χ² 流水线选择最优结果

    参数:
        f: 频率数组 (Hz)
        Z: 复数阻抗数组 (Ω)
        max_M: 最大 RC 元素数量
        add_cap: 是否添加串联电容
        add_ind: 是否添加串联电感
        fit_type: 拟合类型
        lower_limit: 候选下限 (最小 M)

    返回:
        KramersKronigResult (最优 M 的结果)
    """
    f = np.asarray(f, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.complex128)
    n_data = len(f)

    if n_data < 3:
        raise ValueError("数据点数不足，至少需要 3 个点")

    results = []
    for M in range(1, min(max_M, n_data - 1) + 1):
        try:
            r = perform_kk_test(f, Z, num_RC=M, fit_type=fit_type,
                                add_cap=add_cap, add_ind=add_ind,
                                auto_mu=False)
            if np.all(np.isfinite(r.impedances_fit)):
                results.append(r)
        except Exception:
            continue

    if len(results) == 0:
        raise RuntimeError("K-K 优化校验失败: 无法完成任何拟合")

    best_m = suggest_optimal(results, lower_limit=lower_limit)
    best_result = [r for r in results if r.num_RC == best_m][0]
    return best_result
