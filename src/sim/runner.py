"""仿真运行器：跑实验、抽曲线、判寿命。

与项目其它阶段的接口约定：

- 输出统一用 :class:`src.sim.models.SimResult`；
- 放电容量曲线 ``Q(V)`` 与数据集 ``Qdlin`` **同口径**（自高压端起算），
  因此可以直接喂给 :func:`src.features.dq.dq_cross_cycle` 与
  :func:`src.features.dq.dq_paper_stats`，与真实数据的 ΔQ 特征逐项对齐；
- 寿命判定统一为「容量首次跌破初值 80% 的圈数」，与阶段 B/C 的
  ``cycle_life`` 口径一致。
"""
from __future__ import annotations

import time

import numpy as np

from .models import SimResult, _pb, build_model, capacity_scale, load_params
from .protocols import Protocol, experiment_steps

#: ΔQ(V) 的公共电压栅格（与 src.features.dq 保持一致）
V_GRID = {
    "lfp": np.linspace(2.0, 3.5, 300),
    "nmc": np.linspace(3.0, 4.2, 300),
    "nmc_plate": np.linspace(2.5, 4.2, 300),
}

#: 80% 容量保持率判寿
EOL_RATIO = 0.8


def _step_flow(step) -> int:
    """判断一步是放电(+1)、充电(-1) 还是静置(0)。

    **不能依赖 ``Current [A]`` 的符号**：PyBaMM 在不同版本里对放电取正还是取负
    并不一致（24.x 取负、26.x 取正），按符号过滤会在升级后静默把充放电对调。
    这里改用电压走向判断——放电必然降压、充电必然升压、恒压段电压不变但只可能
    出现在充电侧，物理上无歧义。
    """
    try:
        v = np.asarray(step["Voltage [V]"].entries, dtype=float)
        i = np.asarray(step["Current [A]"].entries, dtype=float)
    except (KeyError, TypeError, IndexError):
        return 0
    if i.size < 2 or not np.isfinite(i).any() or float(np.nanmax(np.abs(i))) < 1e-6:
        return 0
    dv = float(np.nan_to_num(v[-1] - v[0]))
    if dv < -1e-4:
        return 1
    if dv > 1e-4:
        return -1
    return -1


def _abs_cap(sol) -> np.ndarray:
    """逐圈容量。PyBaMM 的放电容量符号随版本变化，统一取绝对值。"""
    return np.abs(np.asarray(sol.summary_variables["Capacity [A.h]"], dtype=float))


def _charge_metrics(cyc, frac: float = 0.8) -> tuple[float, float]:
    """单圈的充电时长与「充到 ``frac`` 荷电量」所需时长（秒）。

    只统计恒流段的总时长会掩盖快充的真实收益：C 率越高，恒压段越长，
    总时长反而接近。Severson 用的是「充到 80% 的时长」，这里同口径，
    另外也返回含恒压段的总时长供对照。

    ``frac`` 的分母取该圈的放电安时数（即实际可充回去的量），
    因此老化后仍然可比。

    实现上必须**逐步累加**：若把整圈按 ``I > 0`` 过滤后再积分，
    会把「上一圈充电末 → 下一圈充电初」之间的静置间隔也算进去，
    电流在两端都是正的，梯形积分会凭空多出几安时。
    """
    steps = []
    for step in cyc.steps:
        try:
            t = np.asarray(step["Time [s]"].entries, dtype=float)
            i = np.asarray(step["Current [A]"].entries, dtype=float)
        except (KeyError, TypeError, IndexError):
            continue
        if t.size >= 2:
            steps.append((t, i, _step_flow(step)))
    if not steps:
        return float("nan"), float("nan")

    q_dis = sum(float(_cum_ah(t, i)[-1]) for t, i, fl in steps if fl > 0)
    if not np.isfinite(q_dis) or q_dis <= 0:
        return float("nan"), float("nan")

    target = frac * q_dis
    total, cum, t_frac = 0.0, 0.0, float("nan")
    for t, i, fl in steps:
        if fl >= 0:
            continue
        dur = float(t[-1] - t[0])
        local = _cum_ah(t, i)
        if not np.isfinite(t_frac) and cum + local[-1] >= target:
            k = int(np.searchsorted(cum + local, target))
            k = min(max(k, 0), len(t) - 1)
            t_frac = total + float(t[k] - t[0])
        total += dur
        cum += float(local[-1])
    return total, t_frac


def _cum_ah(t: np.ndarray, i: np.ndarray) -> np.ndarray:
    """由电流对时间积分得到累计安时数（Ah，取绝对值）。"""
    if t.size < 2:
        return np.zeros_like(t)
    return np.abs(np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))])) / 3600.0


def _discharge_curve(cyc, v_grid: np.ndarray) -> np.ndarray:
    """取该圈的放电段，返回自高压端起算的 Q(V)（Ah），插值到 v_grid。"""
    for step in cyc.steps:
        try:
            i = np.asarray(step["Current [A]"].entries, dtype=float)
        except (KeyError, TypeError, IndexError):
            continue
        if i.size and _step_flow(step) > 0:
            V = np.asarray(step["Voltage [V]"].entries, dtype=float)
            t = np.asarray(step["Time [s]"].entries, dtype=float)
            # 优先用 PyBaMM 自带的累计放电容量；Thevenin 等降阶模型没有该变量，
            # 退回电流积分（两者在恒流段完全等价）。
            try:
                Q = np.abs(np.asarray(step["Discharge capacity [A.h]"].entries, dtype=float))
            except KeyError:
                Q = _cum_ah(t, i)
            return qd_from_sim(V, Q, v_grid)
    return np.full(len(v_grid), np.nan)


def qd_from_sim(V, Q, v_grid: np.ndarray) -> np.ndarray:
    """把仿真的 (V, Q) 序列映射到公共电压栅格。

    仿真天然给出「自高压端起算」的累计放电容量（V 从高到低、Q 从 0 增到总容量），
    与数据集 ``Qdlin`` 同口径。电压单调化后线性插值，超出覆盖范围处为 NaN。

    注意 V 是**递减**的，必须先翻转为递增再做 ``maximum.accumulate``，
    否则累积最大值会退化成一个点、整条曲线变 NaN。
    """
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if V.size < 10:
        return np.full(len(v_grid), np.nan)
    V = V[::-1]
    Q = Q[::-1]
    V = np.maximum.accumulate(V)
    keep = np.concatenate([[True], np.diff(V) > 1e-6])
    V, Q = V[keep], Q[keep]
    if V.size < 10:
        return np.full(len(v_grid), np.nan)
    out = np.interp(v_grid, V, Q)
    out[(v_grid < V[0]) | (v_grid > V[-1])] = np.nan
    return out


def life_at(cap: np.ndarray, ratio: float = EOL_RATIO) -> float:
    """容量首次跌破 ``ratio`` 的圈数；未跌破返回 NaN（右删失）。"""
    if cap.size == 0 or not np.isfinite(cap[0]) or cap[0] <= 0:
        return float("nan")
    hit = np.where(cap / cap[0] < ratio)[0]
    return float(hit[0] + 1) if hit.size else float("nan")


def run_cycles(protocol: Protocol, n_cycles: int = 150, chemistry: str = "lfp",
               kind: str = "spm", sei_scale: float | None = 1e3,
               thermal: bool = False, plating: bool = False,
               period: str = "15 seconds",
               curve_cycles: tuple[int, int] = (9, 99)) -> SimResult:
    """跑 ``n_cycles`` 个循环，抽取容量衰减、温升、充电时长与 ΔQ 曲线。"""
    pb = _pb()
    params = load_params(chemistry, sei_scale=sei_scale)
    model = build_model(kind, sei=sei_scale is not None, thermal=thermal,
                        plating=plating)
    steps = experiment_steps(protocol, chemistry, scale=capacity_scale(chemistry))
    exp = pb.Experiment([tuple(steps)] * n_cycles, period=period)

    t0 = time.time()
    sim = pb.Simulation(model, experiment=exp, parameter_values=params)
    sol = sim.solve()
    elapsed = time.time() - t0

    cap = _abs_cap(sol)
    lli = np.asarray(sol.summary_variables["Loss of lithium inventory [%]"], dtype=float)
    n = min(len(cap), n_cycles)

    t_ch, t80, tmax = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        cyc = sol.cycles[i]
        t_ch[i], t80[i] = _charge_metrics(cyc)
        if thermal:
            try:
                T = np.asarray(cyc["Volume-averaged cell temperature [K]"].entries,
                               dtype=float)
                tmax[i] = float(np.nanmax(T))
            except KeyError:
                pass

    vg = V_GRID[chemistry]
    i_early = min(curve_cycles[0], n - 1)
    i_late = min(curve_cycles[1], n - 1)
    qd_e = _discharge_curve(sol.cycles[i_early], vg)
    qd_l = _discharge_curve(sol.cycles[i_late], vg)

    return SimResult(
        name=protocol.key,
        cap=cap[:n],
        lli=lli[:n],
        t_charge=t_ch,
        t80=t80,
        tmax=tmax,
        qd_early=qd_e,
        qd_late=qd_l,
        life=life_at(cap[:n]),
        n_cycles=n,
        seconds=elapsed,
    )


def run_single_discharge(kind: str = "spm", chemistry: str = "lfp",
                         c_rate: float = 1.0, thermal: bool = False,
                         thevenin: bool = False, params_override=None,
                         period: str = "5 seconds"):
    """单次恒流放电，返回 (t_s, V, Q_Ah, 耗时)。

    用于模型谱系对比与倍率能力曲线；不涉及老化，一次求解即可。
    ``params_override`` 可直接传入一组已装配好的参数（例如用反演值
    驱动的 Thevenin），便于隔离「结构误差」与「参数误差」。
    """
    pb = _pb()
    if params_override is not None:
        params = params_override
    elif thevenin:
        from .models import thevenin_params

        params = thevenin_params(chemistry)
    else:
        params = load_params(chemistry)
    model = build_model(kind, thermal=thermal)

    v_min = {"lfp": 2.0, "nmc": 3.0}.get(chemistry)
    if v_min is None:
        from .models import V_WINDOW

        v_min = V_WINDOW[chemistry][0]
    exp = pb.Experiment([f"Discharge at {c_rate:g}C until {v_min:g} V"], period=period)
    sim = pb.Simulation(model, experiment=exp, parameter_values=params)
    t0 = time.time()
    sol = sim.solve()
    elapsed = time.time() - t0

    t = np.asarray(sol["Time [s]"].entries, dtype=float)
    V = np.asarray(sol["Voltage [V]"].entries, dtype=float)
    I = np.asarray(sol["Current [A]"].entries, dtype=float)
    # 统一用电流积分算累计安时：Thevenin 等降阶模型不提供容量变量，
    # 而 SPM/DFN 的 "Discharge capacity [A.h]" 在恒流段与积分结果一致。
    Q = _cum_ah(t, I)
    return t, V, Q, elapsed
