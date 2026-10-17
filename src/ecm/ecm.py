"""一阶/二阶 RC 等效电路（Thevenin）模型 + OCV-SOC 估计 + 最小二乘参数辨识。

方法说明（诚实口径）：
- NASA 数据为恒流(CC)放电/充电，非 HPPC 脉冲。放电起始处电流 0→2A 阶跃
  提供一次瞬态激励，可用以分离 R0（欧姆内阻，瞬时）与 R1/C1、R2/C2（极化）。
- OCV(SOC) 采用充放电平均电压法（去除大部分欧姆与对称极化）。
- 全曲线 2-RC 最小二乘拟合给出 R0/R1/C1/R2/C2 单组参数；SOC 分辨的总内阻
  R_dc(SOC)=(OCV-V)/I 随 SOC 变化可直接求得。CC 数据无法分离各 RC 的 SOC 依赖，
  属数据固有可辨识性限制，报告中如实标注。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# OCV-SOC
# ---------------------------------------------------------------------------
def _cum_q(i_mag, t):
    """Cumulative charge (Ah) via trapezoidal integration of |I| over t (s)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (i_mag[1:] + i_mag[:-1]) * np.diff(t))]) / 3600.0


def build_ocv(battery, n_cycles: int = 5, n_bins: int = 100):
    """Build OCV(SOC) via mean charge/discharge voltage binned on SOC.

    Returns (soc_grid, ocv_grid). Uses the first `n_cycles` discharge and charge
    cycles to keep the OCV representative of the fresh cell.
    """
    from ..parsers import nasa

    c_nom = None
    charge = []  # (soc, V) lists
    discharge = []

    n_d = 0
    n_c = 0
    for _i, cc in nasa.iter_cycles(battery):
        ctype = nasa.cycle_type(cc)
        if ctype not in ("charge", "discharge"):
            continue
        d = nasa.cycle_data(cc)
        i_mag = np.abs(d["Current_measured"])
        t = d["Time"]
        V = d["Voltage_measured"]
        q = _cum_q(i_mag, t)
        if ctype == "discharge":
            if c_nom is None:
                c_nom = nasa.discharge_capacity(cc)
            if n_d >= n_cycles:
                continue
            soc = 1.0 - q / c_nom
            discharge.append((soc, V))
            n_d += 1
        else:
            # 充电圈仅在「已发生至少一次放电」后才从空电(SOC≈0)开始；
            # 首个充电圈是初始充电（从部分 SOC 开始），SOC 映射无效，须跳过。
            if n_d == 0 or n_c >= n_cycles:
                continue
            soc = q / c_nom if c_nom else q / np.max(q)
            charge.append((soc, V))
            n_c += 1
        if n_d >= n_cycles and n_c >= n_cycles:
            break

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ocv_c = np.zeros(n_bins)
    ocv_d = np.zeros(n_bins)
    cnt_c = np.zeros(n_bins)
    cnt_d = np.zeros(n_bins)
    for soc, V in charge:
        idx = np.clip(np.digitize(soc, bins) - 1, 0, n_bins - 1)
        # accumulate sums then average
        np.add.at(ocv_c, idx, np.nan_to_num(V))
        np.add.at(cnt_c, idx, np.nan_to_num(np.ones_like(V)))
    for soc, V in discharge:
        idx = np.clip(np.digitize(soc, bins) - 1, 0, n_bins - 1)
        np.add.at(ocv_d, idx, np.nan_to_num(V))
        np.add.at(cnt_d, idx, np.nan_to_num(np.ones_like(V)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ocv_c = ocv_c / cnt_c
        ocv_d = ocv_d / cnt_d
        ocv = np.where(
            np.isfinite(ocv_c) & np.isfinite(ocv_d),
            0.5 * (ocv_c + ocv_d),
            np.where(np.isfinite(ocv_d), ocv_d, ocv_c),
        )
    # fill leading/trailing NaN by nearest
    ocv = _fill_nan_edges(centers, ocv)
    return centers, ocv


def _fill_nan_edges(x, y):
    y = y.astype(float).copy()
    nan = np.isnan(y)
    if nan.all():
        return y
    good = ~nan
    # fill internal gaps via linear interp
    y[nan] = np.interp(x[nan], x[good], y[good])
    return y


# ---------------------------------------------------------------------------
# Model voltage
# ---------------------------------------------------------------------------
def model_voltage(t, i_mag, ocv_of_soc, soc, p, order: int = 2):
    """Terminal voltage of a 1- or 2-RC Thevenin model under discharge.

    p = [R0, R1, tau1]                       for order 1
    p = [R0, R1, tau1, R2, tau2]             for order 2
    V = OCV(SOC) - I*[R0 + R1(1-e^-t/tau1) + R2(1-e^-t/tau2)]
    """
    ocv = ocv_of_soc(soc)
    eta = i_mag * p[0]
    eta = eta + i_mag * p[1] * (1.0 - np.exp(-t / p[2]))
    if order >= 2:
        eta = eta + i_mag * p[3] * (1.0 - np.exp(-t / p[4]))
    return ocv - eta


def _params(order: int):
    if order == 1:
        return [0.05, 0.05, 60.0]
    return [0.05, 0.05, 30.0, 0.05, 300.0]


def _bounds(order: int):
    if order == 1:
        return ([0.0, 0.0, 1.0], [1.0, 1.0, 1e4])
    return ([0.0, 0.0, 1.0, 0.0, 5.0], [1.0, 1.0, 1e4, 1.0, 1e5])


def fit_rc(t, i_mag, V_meas, ocv_of_soc, soc, order: int = 2, r0_fixed=None):
    """Least-squares fit of RC parameters on a discharge curve (steady region).

    If r0_fixed is given, R0 is held fixed (physical ohmic value from the current
    step) and only the polarization R1/C1, R2/C2 are estimated — better conditioned
    because the polarization voltage is small relative to the OCV uncertainty.
    Returns res with res.x as the FULL parameter vector [R0, R1, tau1, (R2, tau2)].
    """
    if r0_fixed is None:
        p0 = _params(order)
        lb, ub = _bounds(order)

        def residual(p):
            return model_voltage(t, i_mag, ocv_of_soc, soc, p, order) - V_meas
    else:
        if order == 1:
            p0 = [0.05, 60.0]
            lb, ub = [0.0, 1.0], [1.0, 1e4]

            def residual(p):
                return model_voltage(t, i_mag, ocv_of_soc, soc, [r0_fixed, p[0], p[1]], 1) - V_meas
        else:
            p0 = [0.05, 30.0, 0.05, 300.0]
            lb, ub = [0.0, 1.0, 0.0, 5.0], [1.0, 1e4, 1.0, 1e5]

            def residual(p):
                return model_voltage(t, i_mag, ocv_of_soc, soc, [r0_fixed, p[0], p[1], p[2], p[3]], 2) - V_meas

    res = least_squares(
        residual, p0, bounds=(lb, ub), method="trf",
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000,
    )
    if r0_fixed is not None:
        res.x = np.concatenate([[r0_fixed], res.x])
    return res


def params_to_dict(p, order: int = 2) -> dict:
    d = {"R0": p[0], "R1": p[1], "C1": p[2] / p[1]}
    if order >= 2:
        d["R2"] = p[3]
        d["C2"] = p[4] / p[3]
    return d


def steady_region(cc):
    """Return (t_ss, i_ss, V_ss, soc_ss, i_dc) for the constant-current region.

    Starts at the first sample where |I| >= 0.9*max|I|; current is then ~constant.
    soc_ss is computed from full-curve cumulative charge (1 -> 0).
    """
    from ..parsers import nasa

    d = nasa.cycle_data(cc)
    t = d["Time"]
    V = d["Voltage_measured"]
    i_mag = np.abs(d["Current_measured"])
    q = _cum_q(i_mag, t)
    c_nom = nasa.discharge_capacity(cc)
    soc = 1.0 - q / c_nom
    i_start = int(np.argmax(i_mag >= 0.9 * i_mag.max()))
    t_ss = t[i_start:] - t[i_start]
    i_ss = i_mag[i_start:]
    V_ss = V[i_start:]
    soc_ss = soc[i_start:]
    i_dc = float(np.median(i_ss))
    return t_ss, i_ss, V_ss, soc_ss, i_dc


def dc_resistance_soc(cc, ocv_of_soc):
    """SOC-resolved DC total resistance R_dc(SOC) = (OCV(SOC)-V)/I."""
    from ..parsers import nasa

    t_ss, i_ss, V_ss, soc_ss, i_dc = steady_region(cc)
    ocv = ocv_of_soc(soc_ss)
    return soc_ss, (ocv - V_ss) / i_ss


def step_resistance(cc) -> float:
    """R0 (ohmic) from the discharge current step (0 -> I). R0 = dV/dI at the step."""
    from ..parsers import nasa

    d = nasa.cycle_data(cc)
    V = d["Voltage_measured"]
    i_mag = np.abs(d["Current_measured"])
    i_max = i_mag.max()
    i_start = int(np.argmax(i_mag >= 0.9 * i_max))
    if i_start <= 0:
        return float("nan")
    i_before = int(np.argmin(i_mag[:i_start]))
    dV = float(V[i_before] - V[i_start])
    dI = float(i_mag[i_start] - i_mag[i_before])
    return dV / dI if dI > 1e-9 else float("nan")


def cc_region(cc):
    """Constant-current region of a discharge cycle: trims current ramp-up at start
    AND current taper/cutoff at the end (keeps only |I| >= 0.8 * I_dc)."""
    t_ss, i_ss, V_ss, soc_ss, i_dc = steady_region(cc)
    valid = i_ss >= 0.8 * i_dc
    return t_ss[valid], i_ss[valid], V_ss[valid], soc_ss[valid], i_dc


def soc_resistance(cc, ocv_of_soc, n_bins: int = 40):
    """SOC-binned DC total resistance R_dc(SOC) from a discharge cycle (CC region only).

    Returns (soc_centers, R_dc_binned). Used as the SOC-dependent resistance model.
    """
    t_ss, i_ss, V_ss, soc_ss, i_dc = cc_region(cc)
    r = (ocv_of_soc(soc_ss) - V_ss) / i_ss
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    r_sum = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    idx = np.clip(np.digitize(soc_ss, bins) - 1, 0, n_bins - 1)
    np.add.at(r_sum, idx, r)
    np.add.at(cnt, idx, np.ones_like(r))
    with np.errstate(invalid="ignore", divide="ignore"):
        r_bin = r_sum / cnt
    r_bin = _fill_nan_edges(centers, r_bin)
    return centers, r_bin
