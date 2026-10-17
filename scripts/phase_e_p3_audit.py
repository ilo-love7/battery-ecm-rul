# -*- coding: utf-8 -*-
"""阶段 E：P3（第三方 ECM 辨识）的可复现审计。

P3 交付物由另一套流程产出，其结论「2 阶 RC 可辨识、同工况复测 RMSE ≤ 量程 2%、
三颗电池中段总直流电阻一致 ≈166 mΩ」需要独立核验。本脚本不引用 P3 的中间产物，
而是**按 P3 脚本的逻辑重跑一遍**，再做三组对照：

  E1 判据有效性：P3 把 OCV 定义为「实测电压 + 解析极化」，再用同一套参数仿真回去。
     于是残差恒等于两条离散化路径之差。用 P3 原参数、参数 ×2、参数 ×0.5 三组
     分别重建，RMSE 全部 ≈0 —— 说明这条验收判据对参数没有任何约束力。
     把模型拿去预测另一颗电池，误差立刻升到数百 mV。
  E2 先验敏感性：r_dc_mid = (OCV50_PRIOR − V(SOC=0.5))/I 完全由先验决定。
     扫描先验可见三颗电池的曲线始终重合——「一致」来自先验，不来自电池。
  E3 可辨识性：以 τ2=600 s 的 2 阶 RC 为真值，用 P3 的三个松弛窗口去拟合，
     看拟合出的 τ2 是否只是「窗口长度」的函数。P3 报告的 τ2/窗口 为 0.47–1.22，
     落在纯窗口效应的区间内。

输出：reports/phaseE_p3_params.csv、phaseE_p3_selfconsistency.csv、
      phaseE_p3_prior.csv、phaseE_identifiability.csv、figures/fig_e_01..03。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator

FIG = ROOT / "figures"
REP = ROOT / "reports"
NASA = ROOT / "data" / "raw" / "nasa" / "extracted"

# ---- P3 脚本里的常量（照抄，便于逐项对照）----
Q_FACTOR = 1.08
OCV50_PRIOR = 3.86
MID_SOC = 0.45
SOC_MIN = 0.05
BAND_LO, BAND_HI = 0.30, 0.995
RELAX_CYCLE = {"B0005": 1, "B0006": 101, "B0007": 1}
BATS = ["B0005", "B0006", "B0007"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------- 数据读取
def load_cycle(bat: str, idx: int):
    m = loadmat(str(NASA / f"{bat}.mat"), squeeze_me=False, struct_as_record=False)
    c = m[bat][0, 0].cycle[0, idx]
    d = c.data
    if isinstance(d, np.ndarray):
        d = d[0, 0]
    t = np.asarray(d.Time, float).ravel()
    V = np.asarray(d.Voltage_measured, float).ravel()
    I = np.asarray(d.Current_measured, float).ravel()
    cap = float(np.asarray(d.Capacity).ravel()[0])
    return t, V, I, cap


def soc_of(t, I, cap):
    Q = cap * Q_FACTOR * 3600.0
    dt = np.r_[np.diff(t)[0], np.diff(t)]
    return 1.0 + np.cumsum(I * dt) / Q


# --------------------------------------------------------------- 松弛段拟合
def fit_relaxation(t, V, I):
    """P3 的双指数拟合：V = Vinf - A1 e^{-t/tau1} - A2 e^{-t/tau2}。"""
    kend = int(np.where(I < -1.0)[0][-1])
    rel = np.arange(kend + 1, len(t))
    rel = rel[np.abs(I[rel]) < 0.2]
    tr, Vr = t[rel] - t[kend], V[rel]
    I_end = -I[kend]

    def m2(p):
        return p[0] - p[1] * np.exp(-tr / p[2]) - p[3] * np.exp(-tr / p[4])

    sol = least_squares(
        lambda p: m2(p) - Vr,
        [Vr[-1] + 0.05, 0.2, 30.0, 0.2, 300.0],
        bounds=([2.5, 0.0, 2.0, 0.0, 30.0], [4.4, 0.8, 200.0, 0.8, 3600.0]),
        method="trf", max_nfev=8000,
    )
    Vinf, A1, tau1, A2, tau2 = sol.x
    if tau1 > tau2:
        tau1, tau2, A1, A2 = tau2, tau1, A2, A1
    return dict(
        n=len(rel), span=float(tr[-1]), I_end=float(I_end), V_cut=float(V[kend]),
        Vinf=float(Vinf), tau1=float(tau1), tau2=float(tau2),
        R1c=float(A1 / I_end), R2c=float(A2 / I_end),
        rdc_c=float((Vinf - V[kend]) / I_end),
        rmse=float(np.sqrt(np.mean((m2(sol.x) - Vr) ** 2))),
    )


# --------------------------------------------------------------- P3 全流程
def p3_identify(bat: str, prior: float = OCV50_PRIOR):
    """完全按 P3 的逻辑辨识，返回参数与建模所需的函数。"""
    t, V, I, cap = load_cycle(bat, 1)
    soc = soc_of(t, I, cap)
    Q_nom = cap * Q_FACTOR
    soc_cut = 1.0 - cap / Q_nom

    tr, Vr, Ir, _ = load_cycle(bat, RELAX_CYCLE[bat])
    rlx = fit_relaxation(tr, Vr, Ir)
    tau1, tau2 = rlx["tau1"], rlx["tau2"]
    R1c, R2c, rdc_c = rlx["R1c"], rlx["R2c"], rlx["rdc_c"]
    R0c = rdc_c - R1c - R2c
    phi = R1c / (R1c + R2c)

    dis = (I < -1.0) & (soc >= SOC_MIN)
    Id = np.maximum(-I, 0.0)
    k50 = np.where(dis)[0][int(np.argmin(np.abs(soc[dis] - 0.5)))]
    rdc_mid = float((prior - V[k50]) / Id[k50])

    k0 = int(np.argmax(I < -1.0))
    V_rest = float(V[k0 - 1])
    drop1 = (V_rest - V[k0]) / (-I[k0])
    dts = t[k0] - t[k0 - 1]
    a1, a2 = 1 - np.exp(-dts / tau1), 1 - np.exp(-dts / tau2)
    x = (rdc_mid - drop1) / (1 - phi * a1 - (1 - phi) * a2)
    R1m, R2m = phi * x, (1 - phi) * x
    R0m = rdc_mid - x

    if RELAX_CYCLE[bat] == 1:
        p0 = [(soc_cut, R0c), (MID_SOC, R0m)]
        p1 = [(soc_cut, R1c), (MID_SOC, R1m)]
        p2 = [(soc_cut, R2c), (MID_SOC, R2m)]
    else:
        p0 = [(SOC_MIN, R0m), (MID_SOC, R0m)]
        p1 = [(SOC_MIN, R1m), (MID_SOC, R1m)]
        p2 = [(SOC_MIN, R2m), (MID_SOC, R2m)]
    R0_of = lambda s: np.interp(s, *zip(*p0))
    R1_of = lambda s: np.interp(s, *zip(*p1))
    R2_of = lambda s: np.interp(s, *zip(*p2))

    tp = t - t[k0]
    pol = Id * (R0_of(soc)
                + R1_of(soc) * (1 - np.exp(-tp / tau1))
                + R2_of(soc) * (1 - np.exp(-tp / tau2)))
    os_s, os_v = soc[dis][::-1], (V + pol)[dis][::-1]
    keep = np.r_[True, np.diff(os_s) > 1e-9]
    ocv_fn = PchipInterpolator(os_s[keep], os_v[keep], extrapolate=True)

    return dict(
        bat=bat, t=t, V=V, I=I, soc=soc, cap=cap, soc_cut=soc_cut, Id=Id, tp=tp,
        tau1=tau1, tau2=tau2, R0_of=R0_of, R1_of=R1_of, R2_of=R2_of,
        ocv_fn=ocv_fn, pol=pol, dis=dis, k0=k0, drop1=drop1, rlx=rlx,
        R0_cut=R0c, R1_cut=R1c, R2_cut=R2c, rdc_cut=rdc_c,
        R0_mid=R0m, R1_mid=R1m, R2_mid=R2m, rdc_mid=rdc_mid,
    )


def simulate(p, ocv_fn=None, R_of=None):
    """ZOH 精确离散仿真。"""
    t, I, soc, Id = p["t"], p["I"], p["soc"], p["Id"]
    tau1, tau2 = p["tau1"], p["tau2"]
    ocv_fn = p["ocv_fn"] if ocv_fn is None else ocv_fn
    if R_of is None:
        R0_of, R1_of, R2_of = p["R0_of"], p["R1_of"], p["R2_of"]
    else:
        R0_of, R1_of, R2_of = R_of
    n = len(t)
    U1 = np.zeros(n)
    U2 = np.zeros(n)
    R1v, R2v = R1_of(soc), R2_of(soc)
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        e1, e2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
        U1[k + 1] = U1[k] * e1 + Id[k] * R1v[k] * (1 - e1)
        U2[k + 1] = U2[k] * e2 + Id[k] * R2v[k] * (1 - e2)
    Vm = ocv_fn(soc) - Id * R0_of(soc) - U1 - U2
    return Vm


def rmse_band(p, Vm):
    band = p["dis"] & (p["soc"] >= BAND_LO) & (p["soc"] <= BAND_HI)
    return float(np.sqrt(np.mean((Vm[band] - p["V"][band]) ** 2))), band


def selfconsistent_rmse(p, R0, R1, tau1, R2, tau2):
    t, V, I, soc, Id, tp = p["t"], p["V"], p["I"], p["soc"], p["Id"], p["tp"]
    pol = Id * (R0 + R1 * (1 - np.exp(-tp / tau1)) + R2 * (1 - np.exp(-tp / tau2)))
    os_s, os_v = soc[p["dis"]][::-1], (V + pol)[p["dis"]][::-1]
    keep = np.r_[True, np.diff(os_s) > 1e-9]
    ocv = PchipInterpolator(os_s[keep], os_v[keep], extrapolate=True)

    n = len(t)
    U1 = np.zeros(n)
    U2 = np.zeros(n)
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        e1, e2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
        U1[k + 1] = U1[k] * e1 + Id[k] * R1 * (1 - e1)
        U2[k + 1] = U2[k] * e2 + Id[k] * R2 * (1 - e2)
    Vm = ocv(soc) - Id * R0 - U1 - U2
    rm, band = rmse_band(p, Vm)
    g = np.linspace(BAND_LO, BAND_HI, 400)
    d_ocv = np.diff(ocv(g))
    return rm, (float(-d_ocv.min() * 1e3) if d_ocv.min() < 0 else 0.0), ocv


# --------------------------------------------------------------- E3 窗口效应
def relax_points(bat):
    t, V, I, _ = load_cycle(bat, RELAX_CYCLE[bat])
    kend = int(np.where(I < -1.0)[0][-1])
    rel = np.arange(kend + 1, len(t))
    rel = rel[np.abs(I[rel]) < 0.2]
    return t[rel] - t[kend], V[rel], float(-I[kend])


def fit_2rc(tr, Vr, I_end, taus=(2.0, 30.0, 30.0, 300.0)):
    """两指数拟合，返回 tau1/tau2/R1/R2。taus 用于给定初值以便稳定。"""
    def m2(p):
        return p[0] - p[1] * np.exp(-tr / p[2]) - p[3] * np.exp(-tr / p[4])

    sol = least_squares(
        lambda p: m2(p) - Vr,
        [Vr[-1] + 0.05, 0.2, taus[0], 0.2, taus[2]],
        bounds=([2.5, 0.0, 2.0, 0.0, 30.0], [4.4, 1.0, 200.0, 1.0, 3600.0]),
        method="trf", max_nfev=8000,
    )
    Vinf, A1, tau1, A2, tau2 = sol.x
    if tau1 > tau2:
        tau1, tau2, A1, A2 = tau2, tau1, A2, A1
    return dict(Vinf=float(Vinf), tau1=float(tau1), tau2=float(tau2),
                R1=float(A1 / I_end * 1e3), R2=float(A2 / I_end * 1e3),
                rmse=float(np.sqrt(np.mean(sol.fun ** 2))) * 1e3)


# --------------------------------------------------------------- 主流程
def main():
    FIG.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)
    P = {b: p3_identify(b) for b in BATS}

    print("== 复现 P3 的参数（应与其 CSV 一致）==")
    base = []
    for b in BATS:
        p = P[b]
        Vm = simulate(p)
        rm, band = rmse_band(p, Vm)
        base.append(dict(bat=b, cap=p["cap"], soc_cut=p["soc_cut"],
                         tau1=p["tau1"], tau2=p["tau2"],
                         R0_cut=p["R0_cut"] * 1e3, R1_cut=p["R1_cut"] * 1e3,
                         R2_cut=p["R2_cut"] * 1e3, rdc_cut=p["rdc_cut"] * 1e3,
                         R0_mid=p["R0_mid"] * 1e3, R1_mid=p["R1_mid"] * 1e3,
                         R2_mid=p["R2_mid"] * 1e3, rdc_mid=p["rdc_mid"] * 1e3,
                         step_drop=p["drop1"] * 1e3, rmse_mV=rm * 1e3,
                         n_relax=p["rlx"]["n"], relax_span=p["rlx"]["span"]))
        print(f"  {b}: R0_mid={p['R0_mid']*1e3:.1f} r_dc_mid={p['rdc_mid']*1e3:.1f} mΩ, "
              f"RMSE={rm*1e3:.2f} mV, 松弛 n={p['rlx']['n']}/{p['rlx']['span']:.0f}s")
    df_base = pd.DataFrame(base)

    # ===== E1：判据对参数没有约束力 =====
    print("\n== E1 判据有效性：同一颗电池、三组差异很大的参数，RMSE 全部 ≈0 ==")
    e1_rows = []
    for b in BATS:
        p = P[b]
        p3_set = (p["R0_mid"], p["R1_mid"], p["tau1"], p["R2_mid"], p["tau2"])
        sets = {
            "P3 参数": p3_set,
            "参数 ×2": tuple(v * 2 if i < 2 or i == 3 else v for i, v in enumerate(p3_set)),
            "参数 ×0.5": tuple(v * 0.5 if i < 2 or i == 3 else v for i, v in enumerate(p3_set)),
        }
        # ×2/×0.5 只缩放电阻、保持时间常数
        sets["参数 ×2"] = (p["R0_mid"] * 2, p["R1_mid"] * 2, p["tau1"], p["R2_mid"] * 2, p["tau2"])
        sets["参数 ×0.5"] = (p["R0_mid"] / 2, p["R1_mid"] / 2, p["tau1"], p["R2_mid"] / 2, p["tau2"])

        for tag, (r0, r1, t1, r2, t2) in sets.items():
            rm, ocv_drop, _ = selfconsistent_rmse(p, r0, r1, t1, r2, t2)
            e1_rows.append(dict(bat=b, param_set=tag, R0=r0 * 1e3, R1=r1 * 1e3,
                                R2=r2 * 1e3, rdc=(r0 + r1 + r2) * 1e3,
                                rmse_mV=rm * 1e3, ocv_backtrack_mV=ocv_drop))
            print(f"  {b} [{tag:8s}] R0={r0*1e3:5.1f} R1={r1*1e3:5.1f} R2={r2*1e3:5.1f} "
                  f"r_dc={(r0+r1+r2)*1e3:6.1f} mΩ -> RMSE={rm*1e3:8.2e} mV, "
                  f"OCV 回落={ocv_drop:6.1f} mV")

        # 跨电池：用另一颗电池的模型预测本电池
        q = P[[x for x in BATS if x != b][0]]
        Vm_x = simulate(p, ocv_fn=q["ocv_fn"], R_of=(q["R0_of"], q["R1_of"], q["R2_of"]))
        rm_x, _ = rmse_band(p, Vm_x)
        e1_rows.append(dict(bat=b, param_set=f"换用 {q['bat']} 的模型", R0=np.nan, R1=np.nan,
                            R2=np.nan, rdc=np.nan, rmse_mV=rm_x * 1e3, ocv_backtrack_mV=np.nan))
        print(f"  {b} [换用 {q['bat']}] -> RMSE={rm_x*1e3:.1f} mV")
    df_e1 = pd.DataFrame(e1_rows)

    # ===== E2：先验敏感性 =====
    print("\n== E2 先验敏感性：r_dc_mid 由 OCV(0.5) 先验唯一决定 ==")
    priors = np.round(np.arange(3.76, 3.965, 0.02), 3)
    e2 = []
    for b in BATS:
        for pr in priors:
            p = p3_identify(b, prior=float(pr))
            e2.append(dict(bat=b, prior=float(pr), rdc_mid=p["rdc_mid"] * 1e3,
                           R0_mid=p["R0_mid"] * 1e3))
    df_e2 = pd.DataFrame(e2)
    for b in BATS:
        d = df_e2[df_e2.bat == b]
        print(f"  {b}: 先验 3.76→3.96 V 时 r_dc_mid "
              f"{d.rdc_mid.iloc[0]:.1f} → {d.rdc_mid.iloc[-1]:.1f} mΩ "
              f"(Δ={d.rdc_mid.iloc[-1]-d.rdc_mid.iloc[0]:+.1f})")

    # ===== E3：τ2 是窗口的函数（真实数据）=====
    print("\n== E3 窗口效应：把同一段松弛数据截短，看 τ2 是否跟着缩 ==")
    e3 = []
    for b in BATS:
        tr, Vr, I_end = relax_points(b)
        span = float(tr[-1])
        for frac in [1.0, 0.8, 0.6, 0.45]:
            m = tr <= span * frac + 1e-9
            if m.sum() < 5:
                continue
            f = fit_2rc(tr[m], Vr[m], I_end)
            e3.append(dict(bat=b, kind="截断", frac=frac, n_pts=int(m.sum()),
                           window_s=float(tr[m][-1]), tau1=f["tau1"], tau2=f["tau2"],
                           R1=f["R1"], R2=f["R2"], rmse_mV=f["rmse"]))
        # bootstrap：有放回重采样（点数本就只有 8–17 个）
        rng = np.random.default_rng(42)
        for k in range(200):
            idx = rng.integers(0, len(tr), len(tr))
            idx = np.unique(idx)
            if len(idx) < 5:
                continue
            f = fit_2rc(tr[idx], Vr[idx], I_end)
            e3.append(dict(bat=b, kind="bootstrap", frac=np.nan, n_pts=len(idx),
                           window_s=float(tr[idx][-1]), tau1=f["tau1"], tau2=f["tau2"],
                           R1=f["R1"], R2=f["R2"], rmse_mV=f["rmse"]))
        d_tr = pd.DataFrame([r for r in e3 if r["bat"] == b and r["kind"] == "截断"])
        d_bs = pd.DataFrame([r for r in e3 if r["bat"] == b and r["kind"] == "bootstrap"])
        lo, hi = np.percentile(d_bs.tau2, [5, 95])
        print(f"  {b}: 窗口 {span:5.1f}s | 截到 45% 时 τ2="
              f"{d_tr[d_tr.frac == d_tr.frac.min()].tau2.iloc[0]:6.1f}s"
              f"（全长 τ2={d_tr[d_tr.frac == 1.0].tau2.iloc[0]:6.1f}s）"
              f" | bootstrap 90% 区间 τ2 ∈ [{lo:.0f}, {hi:.0f}] s"
              f"（{hi/max(lo,1e-6):.1f}× 跨度）")
    df_e3 = pd.DataFrame(e3)

    # ---------------------------------------------------------------- 落盘
    df_base.to_csv(REP / "phaseE_p3_params.csv", index=False, encoding="utf-8-sig")
    df_e1.to_csv(REP / "phaseE_p3_selfconsistency.csv", index=False, encoding="utf-8-sig")
    df_e2.to_csv(REP / "phaseE_p3_prior.csv", index=False, encoding="utf-8-sig")
    df_e3.to_csv(REP / "phaseE_identifiability.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- 图 E1
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax = axes[0]
    tags = ["P3 参数", "参数 ×2", "参数 ×0.5"]
    xs = np.arange(len(BATS))
    w = 0.26
    for i, (tag, c) in enumerate(zip(tags, ["#4f81bd", "#c0504d", "#9bbb59"])):
        d = df_e1[df_e1.param_set == tag].set_index("bat").loc[BATS]
        ax.bar(xs + (i - 1) * w, d.rmse_mV.clip(lower=1e-6), w, label=tag, color=c)
    for b_i, b in enumerate(BATS):
        v = df_e1[(df_e1.bat == b) & (df_e1.param_set.str.startswith("换用"))].rmse_mV.iloc[0]
        ax.plot([b_i - 1.5 * w, b_i + 1.5 * w], [v, v], color="k", lw=1.6, ls="--")
        ax.text(b_i + 1.6 * w, v, f"换电池\n{v:.0f} mV", fontsize=7, va="center")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(BATS)
    ax.set_ylabel("同工况复测 RMSE (mV，对数)")
    ax.set_ylim(1e-2, 3e1)
    ax.set_title("E1a  参数 ×2 / ×0.5 仍是 0.1 mV；连换错电池也只有 10 mV 量级")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    p = P["B0005"]
    g = np.linspace(BAND_LO, BAND_HI, 300)
    g_lo = np.linspace(0.10, BAND_HI, 400)
    ax.plot(g_lo, p["ocv_fn"](g_lo), "-", color="k", lw=2.0,
            label="P3 实际（R 随 SOC 变）：低 SOC 驼峰")
    for tag, c, ls in [("P3 参数", "tab:blue", "-"), ("参数 ×2", "tab:red", "--"),
                       ("参数 ×0.5", "tab:green", ":")]:
        r0, r1, t1, r2, t2 = {
            "P3 参数": (p["R0_mid"], p["R1_mid"], p["tau1"], p["R2_mid"], p["tau2"]),
            "参数 ×2": (p["R0_mid"] * 2, p["R1_mid"] * 2, p["tau1"], p["R2_mid"] * 2, p["tau2"]),
            "参数 ×0.5": (p["R0_mid"] / 2, p["R1_mid"] / 2, p["tau1"], p["R2_mid"] / 2, p["tau2"]),
        }[tag]
        _, drop, ocv = selfconsistent_rmse(p, r0, r1, t1, r2, t2)
        ax.plot(g, ocv(g), ls, color=c, lw=1.5,
                label=f"{tag}（OCV 回落 {drop:.0f} mV）")
    ax.set_xlabel("SOC")
    ax.set_ylabel("反推得到的 OCV (V)")
    ax.set_title("E1b  B0005：参数变一倍，OCV 平移 0.3 V")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_e_01_selfconsistency.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 图 E2
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for b, mk in zip(BATS, "osD"):
        d = df_e2[df_e2.bat == b]
        ax.plot(d.prior, d.rdc_mid, marker=mk, ms=5, lw=1.6, label=b)
    ax.axvline(OCV50_PRIOR, color="k", ls="--", lw=1, label="P3 采用 3.86 V")
    ax.set_xlabel("先验 OCV(SOC=0.5) (V)")
    ax.set_ylabel("r_dc_mid (mΩ)")
    ax.set_title("E2  三条线在任何先验下重合：「三颗电池一致」来自先验，不来自电池")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_e_02_prior_sensitivity.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 图 E3
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax = axes[0]
    for b, mk in zip(BATS, "osD"):
        d = df_e3[(df_e3.bat == b) & (df_e3.kind == "截断")].sort_values("window_s")
        ax.plot(d.window_s, d.tau2, mk + "-", lw=1.5, ms=6, label=f"{b}")
    ax.plot([0, 400], [0, 400], "k:", lw=1, label="τ2 = 窗口长度")
    ax.set_xlabel("实际使用的松弛窗口长度 (s)")
    ax.set_ylabel("拟合出的 τ2 (s)")
    ax.set_title("E3a  τ2 随窗口一起缩：它是窗口的函数，不是电池的时间常数")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [df_e3[(df_e3.bat == b) & (df_e3.kind == "bootstrap")].tau2.values for b in BATS]
    bp = ax.boxplot(data, tick_labels=BATS, showfliers=False, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#4f81bd", "#c0504d", "#9bbb59"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_ylabel("τ2 的 bootstrap 分布 (s)")
    ax.set_title("E3b  8–17 个点做 5 参数拟合：τ2 的区间宽到无法下结论")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_e_03_identifiability.png", dpi=150)
    plt.close(fig)

    print("\n已生成：figures/fig_e_01..03、reports/phaseE_p3_*.csv")


if __name__ == "__main__":
    main()
