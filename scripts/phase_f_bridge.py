# -*- coding: utf-8 -*-
"""阶段 F：三条主线的桥（P3 → P2 → P1）。

阶段 E 证明 P3 的验收判据没有约束力、参数值由先验与窗口决定。那么 P3 的
参数还能不能用？本脚本用 P2 的机理模型当「已知真值的试验台」回答两个问题：

  F1 采样率审计：同一条 DFN 曲线，先按 0.05 s 采样读出初始阶跃内阻，
     再降到 NASA 的 18.7 s 采样跑 P3 的辨识流程。两者的差距就是
     「低采样率 + 先验锚定」的代价。
  F2 特征族仲裁：把 LLI / LAM_pos / LAM_neg 三档退化注入 DFN，分别读
     「初始阶跃内阻 R0_eff」与「ΔQ(V) 深度」，看哪一族特征对这些退化模式
     敏感。这解释了为什么 P1 用 ΔQ(V) 而不用内阻——也说明 P3 的内阻参数
     在寿命预测里能扮演什么角色。

**符号约定**：PyBaMM 24.x 把放电电流记为负、26.x 记为正（见 runner._step_flow）。
本脚本只用 ``Time`` / ``Voltage`` / ``|Current|``，不依赖任何符号约定，
因此在两个版本下都能跑。绝对数值仍随版本变化，故只做同版本内的相对比较。

产物：reports/phaseF_bridge.csv、reports/phaseF_sensitivity.csv、
      figures/fig_f_01_sampling_audit.png、figures/fig_f_02_feature_families.png

运行：
    python scripts/phase_f_bridge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from src.sim.models import load_params  # noqa: E402

FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

CHEM = "lfp"
I_DIS = 2.0                      # NASA 的放电电流
V_CUT = 2.5                      # NASA B0006 的截止电压，取中间值
Q_FACTOR = 1.08                  # P3 的 SOC 口径
LEVELS = (0.03, 0.06, 0.10)
MODES = {
    "LLI": ("锂库存损失 LLI", "Initial concentration in negative electrode [mol.m-3]"),
    "LAM_pos": ("正极活性物质损失 LAM_pos", "Positive electrode active material volume fraction"),
    "LAM_neg": ("负极活性物质损失 LAM_neg", "Negative electrode active material volume fraction"),
}
COLORS = {"LLI": "#2b6cb0", "LAM_pos": "#dd6b20", "LAM_neg": "#38a169"}


def _ah(t, i):
    """累计安时（取绝对值，与电流符号约定无关）。"""
    if t.size < 2:
        return np.zeros_like(t)
    return np.abs(np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))])) / 3600.0


def _params(mode=None, level=0.0):
    params = load_params(CHEM)
    if mode is not None:
        key = MODES[mode][1]
        params[key] = params[key] * (1.0 - level)
    return params


def run_dfn(period: str, rest_s: float = 400.0):
    """DFN：静置 10 s（建立 OCV 参考）→ 2 A 放电到 2.5 V → 静置 rest_s 秒。

    开头必须留一段静置，否则 t=0 时电流已加载，拿不到"加载前"的电压，
    初始阶跃内阻无从计算。
    """
    import gc

    import pybamm

    model = pybamm.lithium_ion.DFN()
    exp = pybamm.Experiment([
        "Rest for 10 seconds",
        f"Discharge at {I_DIS:g} A until {V_CUT:g} V",
        f"Rest for {rest_s:g} seconds",
    ], period=period)
    sol = pybamm.Simulation(model, experiment=exp, parameter_values=_params()).solve()
    t = np.asarray(sol["Time [s]"].entries, float)
    V = np.asarray(sol["Voltage [V]"].entries, float)
    I = np.abs(np.asarray(sol["Current [A]"].entries, float))
    out = (t, V, I, _ah(t, I))
    del sol
    gc.collect()
    return out


def run_pulse(mode=None, level=0.0):
    """2 A / 5 s 脉冲：只用来读初始阶跃内阻，解很小，跑得快。"""
    import gc

    import pybamm

    model = pybamm.lithium_ion.DFN()
    exp = pybamm.Experiment(["Rest for 10 seconds",
                             f"Discharge at {I_DIS:g} A for 5 seconds"],
                            period="0.1 seconds")
    sol = pybamm.Simulation(model, experiment=exp,
                            parameter_values=_params(mode, level)).solve()
    t = np.asarray(sol["Time [s]"].entries, float)
    V = np.asarray(sol["Voltage [V]"].entries, float)
    I = np.abs(np.asarray(sol["Current [A]"].entries, float))
    out = (t, V, I)
    del sol
    gc.collect()
    return out


def run_ocv(period: str = "120 seconds"):
    """C/50 准平衡放电，用于给出 OCV(SOC)，不依赖任何先验。"""
    import pybamm

    model = pybamm.lithium_ion.DFN()
    sol = pybamm.Simulation(model, experiment=pybamm.Experiment(
        [f"Discharge at {I_DIS/20:g} A until {V_CUT:g} V"], period=period),
        parameter_values=load_params(CHEM)).solve()
    t = np.asarray(sol["Time [s]"].entries, float)
    V = np.asarray(sol["Voltage [V]"].entries, float)
    I = np.abs(np.asarray(sol["Current [A]"].entries, float))
    q = _ah(t, I)
    return 1.0 - q / q[-1], V


def r0_step(t, V, I):
    """初始阶跃内阻：首个加载点到加载前一点的压降 / 电流。"""
    k = int(np.argmax(I > 0.5))
    return float((V[k - 1] - V[k]) / I[k]) * 1e3


def p3_on_synthetic(t, V, I, ocv50_prior: float):
    """把 P3 的辨识流程套到任意一条放电+静置曲线上（只用 |I|）。"""
    cap = float(np.max(_ah(t, I)))
    Q = cap * Q_FACTOR * 3600.0
    soc = 1.0 - np.cumsum(I * np.r_[np.diff(t)[0], np.diff(t)]) / Q

    kend = int(np.where(I > 0.5)[0][-1])
    rel = np.arange(kend + 1, len(t))
    rel = rel[I[rel] < 0.2]
    tr, Vr = t[rel] - t[kend], V[rel]
    I_end = float(I[kend])

    def m2(p):
        return p[0] - p[1] * np.exp(-tr / p[2]) - p[3] * np.exp(-tr / p[4])

    sol = least_squares(lambda p: m2(p) - Vr,
                        [Vr[-1] + 0.05, 0.2, 30.0, 0.2, 300.0],
                        bounds=([2.0, 0.0, 2.0, 0.0, 30.0], [4.0, 1.0, 200.0, 1.0, 3600.0]),
                        method="trf", max_nfev=8000)
    Vinf, A1, tau1, A2, tau2 = sol.x
    if tau1 > tau2:
        tau1, tau2, A1, A2 = tau2, tau1, A2, A1
    R1c, R2c = A1 / I_end, A2 / I_end
    rdc_c = (Vinf - V[kend]) / I_end
    R0c = rdc_c - R1c - R2c
    phi = R1c / (R1c + R2c)

    dis = (I > 0.5) & (soc >= 0.05)
    k50 = np.where(dis)[0][int(np.argmin(np.abs(soc[dis] - 0.5)))]
    rdc_mid = float((ocv50_prior - V[k50]) / I[k50])

    k0 = int(np.argmax(I > 0.5))
    drop1 = (V[k0 - 1] - V[k0]) / I[k0]
    dts = t[k0] - t[k0 - 1]
    a1, a2 = 1 - np.exp(-dts / tau1), 1 - np.exp(-dts / tau2)
    x = (rdc_mid - drop1) / (1 - phi * a1 - (1 - phi) * a2)
    return dict(cap_Ah=cap, tau1=tau1, tau2=tau2, R0_cut=R0c * 1e3,
                R1_cut=R1c * 1e3, R2_cut=R2c * 1e3, rdc_cut=rdc_c * 1e3,
                R0_mid=(rdc_mid - x) * 1e3, R1_mid=phi * x * 1e3,
                R2_mid=(1 - phi) * x * 1e3, rdc_mid=rdc_mid * 1e3,
                step_drop=drop1 * 1e3, n_relax=int(len(rel)),
                relax_span=float(tr[-1]) if len(tr) else 0.0)


def main():
    import pybamm
    print(f"PyBaMM {pybamm.__version__}")

    # ===== F1 =====
    print("\n== F1 采样率审计：同一条 DFN 曲线，按不同采样率跑 P3 的流程 ==")
    soc_eq, V_eq = run_ocv()
    ocv50 = float(np.interp(0.5, soc_eq[::-1], V_eq[::-1]))
    print(f"  DFN 准平衡 OCV(SOC=0.5) = {ocv50:.3f} V（作为 P3 的先验，给它最好条件）")

    t_hi, V_hi, I_hi, _ = run_dfn("0.05 seconds")
    R0_hi = r0_step(t_hi, V_hi, I_hi)
    print(f"  0.05 s 采样下的初始阶跃内阻 = {R0_hi:.1f} mΩ")

    rows_f1 = []
    for dt in [0.05, 0.5, 2.0, 5.0, 18.7]:
        k = max(1, int(round(dt / 0.05)))
        t, V, I = t_hi[::k], V_hi[::k], I_hi[::k]
        r = p3_on_synthetic(t, V, I, ocv50)
        rows_f1.append(dict(dt_s=float(np.median(np.diff(t))), R0=r["R0_mid"],
                            R1=r["R1_mid"], R2=r["R2_mid"], rdc=r["rdc_mid"],
                            step=r["step_drop"], tau1=r["tau1"], tau2=r["tau2"],
                            n_relax=r["n_relax"]))
        print(f"  dt={rows_f1[-1]['dt_s']:5.2f}s: R0={r['R0_mid']:5.1f} "
              f"R1+R2={r['R1_mid']+r['R2_mid']:5.1f} r_dc={r['rdc_mid']:6.1f} mΩ "
              f"| 首点压降/I={r['step_drop']:5.1f}")
    df1 = pd.DataFrame(rows_f1)
    df1.to_csv(REP / "phaseF_bridge.csv", index=False, encoding="utf-8-sig")

    # ===== F2 =====
    print("\n== F2 特征族仲裁：内阻 vs ΔQ(V) 对三种退化模式的敏感度 ==")
    dq_ref = pd.read_csv(REP / "phaseD2_modes.csv")
    dq_map = {}
    for _, r in dq_ref.dropna(subset=["dq_depth_mAh"]).iterrows():
        if pd.notna(r.get("level")):
            dq_map[(r["mode"], round(float(r["level"]), 3))] = float(r["dq_depth_mAh"])

    rows = []
    t_p, V_p, I_p = run_pulse()
    r0_fresh = r0_step(t_p, V_p, I_p)
    rows.append(dict(mode="fresh", level=0.0, R0_eff_mOhm=r0_fresh, dq_mAh=0.0))
    print(f"  新鲜: R0_eff={r0_fresh:.1f} mΩ")
    for mode in MODES:
        for lv in LEVELS:
            tp, Vp, Ip = run_pulse(mode=mode, level=lv)
            r0 = r0_step(tp, Vp, Ip)
            rows.append(dict(mode=mode, level=lv, R0_eff_mOhm=r0,
                             dq_mAh=dq_map.get((mode, round(lv, 3)), np.nan)))
            print(f"  {mode:8s} {lv*100:4.0f}%: R0_eff={r0:6.1f} mΩ "
                  f"({100*(r0/r0_fresh-1):+5.1f}%) | ΔQ 深度={rows[-1]['dq_mAh']:6.1f} mAh")
    df2 = pd.DataFrame(rows)
    df2.to_csv(REP / "phaseF_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- 图 F1
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax = axes[0]
    ax.plot(t_hi / 60, V_hi, lw=1.0, color="tab:blue", label="0.05 s 采样（真值）")
    k = max(1, int(round(18.7 / 0.05)))
    ax.plot(t_hi[::k] / 60, V_hi[::k], "o", ms=3.5, color="tab:red",
            label="18.7 s 采样（NASA 级）")
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("端电压 (V)")
    ax.set_title("F1a  同一颗 DFN 电池的两种采样率")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(df1.dt_s, df1.R0, "o-", lw=1.8, label="R0（公式给出）")
    ax.plot(df1.dt_s, df1.R1 + df1.R2, "s-", lw=1.8, label="R1 + R2（公式给出）")
    ax.plot(df1.dt_s, df1.rdc, "^-", lw=1.4, ls="--", label="r_dc（先验锚定，恒定）")
    ax.plot(df1.dt_s, df1.step, "k:", lw=1.4, label="实测首点压降 / I")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("采样步长 dt (s)")
    ax.set_ylabel("电阻 (mΩ)")
    ax.set_title("F1b  采样变粗：R0 从 39 膨胀到 208 mΩ，R1+R2 变负")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_f_01_sampling_audit.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 图 F2
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax = axes[0]
    for mode, (label, _) in MODES.items():
        d = df2[df2["mode"] == mode].sort_values("level")
        ax.plot(d.level * 100, d.R0_eff_mOhm, "o-", lw=1.6, color=COLORS[mode], label=label)
    ax.axhline(r0_fresh, color="k", ls=":", lw=1, label="新鲜电池")
    ax.set_xlabel("注入退化量 (%)")
    ax.set_ylabel("初始阶跃内阻 R0_eff (mΩ)")
    ax.set_title("F2a  内阻对三种退化模式几乎不敏感")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for mode, (label, _) in MODES.items():
        d = df2[df2["mode"] == mode].sort_values("level")
        ax.plot(d.level * 100, d.dq_mAh, "s-", lw=1.6, color=COLORS[mode], label=label)
    ax.set_xlabel("注入退化量 (%)")
    ax.set_ylabel("ΔQ(V) 深度 (mAh)")
    ax.set_title("F2b  ΔQ(V) 对 LLI / LAM_neg 高度敏感")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_f_02_feature_families.png", dpi=150)
    plt.close(fig)

    print("\n已生成：figures/fig_f_01..02、reports/phaseF_*.csv")


if __name__ == "__main__":
    main()
