"""ΔQ(V) 特征工程。

本模块提供两种定义，**含义不同、不可混用**：

1. ``dq_curve``  —— 滞回口径（hysteresis）
   ``ΔQ_hys(V) = Q_discharge(V) - Q_charge(V)``
   同一循环内、充放电两个方向相减，反映库仑效率损失与极化滞后。
   量级 0~0.35 Ah，**对充电协议高度敏感**，跨批次易漂移。

2. ``dq_cross_cycle`` —— 论文口径（Severson et al. 2019）
   ``ΔQ(V) = Q_100(V) - Q_10(V)``
   跨循环、同一放电方向相减，反映老化导致的容量损失在电压轴上的分布。
   恒为负，量级 -0.4~0 Ah，与充电协议关系弱，跨批次更稳。
   **这是论文 Table 1 的三个 ΔQ 特征的来源。**

实现要点：
1. 充电段取 I>+thr，放电段取 I<-thr，各自把 Q 对 V 做单调化插值；
2. 电压栅格固定为 [2.0, 3.5] V（1000 点在论文中，本项目用 300 点，兼顾存储与精度）；
3. 覆盖不足的循环返回 NaN，由上层剔除并计数（数据质量报告要用）。
"""
from __future__ import annotations

import numpy as np

V_MIN, V_MAX, N_GRID = 2.0, 3.5, 300
V_GRID = np.linspace(V_MIN, V_MAX, N_GRID)


def _q_of_v(V: np.ndarray, Q: np.ndarray, increasing: bool) -> np.ndarray:
    """把 (V, Q) 映射到公共电压栅格；V 需单调，Q 单调递增。"""
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if not increasing:
        V = V[::-1]
        Q = Q[::-1]
        # 放电容量 Qd 从高压端(≈3.6V)起累积到低压端；ΔQ(V) 要求两条曲线
        # 都以「低压端 = 0」为共同起点，故换算为「自低压端起算的已放电容量」。
        Q = Q[0] - Q
    # 单调化：充电时 V 用累进最大值，放电反向处理后同理
    V = np.maximum.accumulate(V)
    # 去掉重复电压点（CV 平台），保留首次到达该电压对应的容量
    keep = np.concatenate([[True], np.diff(V) > 1e-6])
    V, Q = V[keep], Q[keep]
    if V.size < 5:
        return np.full(N_GRID, np.nan)
    # 覆盖度检查：batch2 的充电从 ~2.47 V 起（上一圈放电到 2.0 V 后静置回升），
    # 低压端缺口用端点值常值外推；若充电起点高于 2.6 V（部分充电/异常循环）则判无效。
    if V[0] > V_MIN + 0.6 or V[-1] < V_MAX - 0.05:
        return np.full(N_GRID, np.nan)
    return np.interp(V_GRID, V, Q)


def dq_curve(V, I, Qc, Qd, v_grid: np.ndarray = V_GRID) -> np.ndarray:
    """单循环 ΔQ(V)。返回长度 len(v_grid) 的数组，覆盖不足时全 NaN。"""
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    Qc = np.asarray(Qc, dtype=float)
    Qd = np.asarray(Qd, dtype=float)
    n = min(len(V), len(I), len(Qc), len(Qd))
    V, I, Qc, Qd = V[:n], I[:n], Qc[:n], Qd[:n]

    chg = I > 0.05
    dis = I < -0.05
    if chg.sum() < 30 or dis.sum() < 30:
        return np.full(len(v_grid), np.nan)

    q_c = _q_of_v(V[chg], Qc[chg], increasing=True)
    q_d = _q_of_v(V[dis], Qd[dis], increasing=False)
    if np.isnan(q_c).all() or np.isnan(q_d).all():
        return np.full(len(v_grid), np.nan)
    # 栅格若被调用方替换，重采样一次
    if len(v_grid) != N_GRID or not np.allclose(v_grid, V_GRID):
        q_c = np.interp(v_grid, V_GRID, q_c)
        q_d = np.interp(v_grid, V_GRID, q_d)
    return q_d - q_c


def dq_stats(dq: np.ndarray) -> dict:
    """单个 ΔQ(V) 曲线的标量特征。"""
    if dq is None or np.isnan(dq).all():
        return {"dq_min": np.nan, "dq_var": np.nan, "dq_logmin": np.nan, "dq_mean": np.nan}
    mn = float(np.nanmin(dq))
    var = float(np.nanvar(dq))
    return {
        "dq_min": mn,
        "dq_var": var,
        "dq_logmin": float(np.log10(abs(mn) + 1e-12)),
        "dq_mean": float(np.nanmean(dq)),
    }


VDLIN = np.linspace(3.5, 2.0, 1000)  # 数据集 Qdlin 的固定电压栅格（降序，3.5 → 2.0 V）


def qd_from_qdlin(qdlin, v_grid: np.ndarray = V_GRID) -> np.ndarray:
    """把数据集自带的 ``Qdlin`` 重采样到公共栅格。

    Qdlin 与 Vdlin 一一对应：Vdlin 从 3.5 V 降到 2.0 V，Qdlin 从 0 增到总放电容量，
    即「自高压端起算的已放电容量」——这正是论文 ΔQ(V)=Q_100−Q_10 所用的口径。
    优先用本函数而非 ``qd_curve`` 自算：官方曲线已做过插值与平滑，
    在 3.4–3.5 V 陡降段比从原始 V/I 积分稳定得多。
    """
    q = np.asarray(qdlin, dtype=float)
    if q.size < 10 or np.isnan(q).all():
        return np.full(len(v_grid), np.nan)
    n = min(q.size, VDLIN.size)
    return np.interp(v_grid, VDLIN[:n][::-1], q[:n][::-1])


def qd_curve(V, I, Qd, v_grid: np.ndarray = V_GRID) -> np.ndarray:
    """单循环放电容量曲线 Q(V)，**自高压端(3.5 V)起算**，单位 Ah。

    备用实现：当数据集不提供 Qdlin 时，从原始 V/I 序列自算。
    与 ``Qdlin`` 同口径（V=3.5 V 处≈0，V=2.0 V 处≈总放电容量）。
    注意这与 ``_q_of_v`` 里滞回口径的「自低压端起算」方向相反——论文的
    ΔQ(V)=Q_100−Q_10 必须用高压端起算的曲线，否则整条曲线会翻过来。
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    Qd = np.asarray(Qd, dtype=float)
    n = min(len(V), len(I), len(Qd))
    V, I, Qd = V[:n], I[:n], Qd[:n]

    dis = I < -0.05
    if dis.sum() < 30:
        return np.full(len(v_grid), np.nan)

    vd = V[dis][::-1]   # 放电段电压递减 → 反转为递增
    qd = Qd[dis][::-1]  # 容量随之递减
    vd = np.maximum.accumulate(vd)
    keep = np.concatenate([[True], np.diff(vd) > 1e-6])
    vd, qd = vd[keep], qd[keep]
    if vd.size < 5:
        return np.full(len(v_grid), np.nan)
    if vd[0] > V_MIN + 0.6 or vd[-1] < V_MAX - 0.05:
        return np.full(len(v_grid), np.nan)
    return np.interp(v_grid, vd, qd)


def dq_cross_cycle(qd_early: np.ndarray, qd_late: np.ndarray) -> np.ndarray:
    """论文口径 ΔQ(V) = Q_late(V) - Q_early(V)，恒为负值。

    物理含义：老化导致的容量损失在电压轴上的分布。任一输入全 NaN 时返回全 NaN。
    """
    if qd_early is None or qd_late is None:
        return np.full(N_GRID, np.nan)
    a = np.asarray(qd_early, dtype=float)
    b = np.asarray(qd_late, dtype=float)
    if np.isnan(a).all() or np.isnan(b).all():
        return np.full(len(a), np.nan)
    return b - a


def dq_paper_stats(dq: np.ndarray) -> dict:
    """论文 Table 1 的三个 ΔQ 特征，取 log10 压缩跨电池数量级差异。

    - xq_log_var : log10 Var(ΔQ)
    - xq_log_min : log10 |min ΔQ|（曲线最深谷；全正曲线退回取绝对值最大点）
    - xq_log_mean: log10 |mean ΔQ|（整体下移幅度）
    """
    if dq is None or np.isnan(dq).all():
        return {"xq_log_var": np.nan, "xq_log_min": np.nan, "xq_log_mean": np.nan}
    d = np.asarray(dq, dtype=float)
    mn = float(np.nanmin(d))
    depth = -mn if mn < 0 else float(np.nanmax(np.abs(d)))
    return {
        "xq_log_var": float(np.log10(np.nanvar(d) + 1e-12)),
        "xq_log_min": float(np.log10(depth + 1e-12)),
        "xq_log_mean": float(np.log10(abs(np.nanmean(d)) + 1e-12)),
    }
