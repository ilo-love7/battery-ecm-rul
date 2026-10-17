"""阶段 D1：PyBaMM 物理仿真复现（候选清单 P2）。

对应博远《候选项目清单》P2 的前半段「复现基准曲线」：

1. **模型谱系**：Thevenin(1 阶 RC) / SPM / DFN 三档保真度在同一工况下的
   端电压曲线，量化「降阶要付多少精度代价」；
2. **倍率能力**：0.5C–4C 放电的容量利用率与极化增长；
3. **快充协议族**：7 条多步 CC 协议的充电时长与温升，复现「快充—发热」耦合。

产物：
- figures/fig_d_01_model_hierarchy.png
- figures/fig_d_02_rate_capability.png
- figures/fig_d_03_fastcharge_heat.png
- reports/phaseD1_sim.csv

运行：
    python scripts/phase_d1_pybamm_sim.py
需要 pybamm（``pip install -r requirements-full.txt``）。
"""
from __future__ import annotations

import json
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

from src.sim.models import STAGE_A_ECM, ocv_from_spm  # noqa: E402
from src.sim.protocols import PROTOCOLS  # noqa: E402
from src.sim.runner import run_cycles, run_single_discharge  # noqa: E402

FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

CHEM = "lfp"          # 电压曲线复现用磷酸铁锂（与 NASA / MIT 同体系）
CHEM_THERMAL = "nmc"  # 产热用 NMC 参数集（含集流体几何与热参数）


def interp_to(V, Q, q_grid):
    """把 (V, Q) 曲线重采样到给定容量栅格，便于跨模型逐点比较。"""
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if Q.size < 5 or Q.max() <= 0:
        return np.full(len(q_grid), np.nan)
    Q = Q / Q.max()
    keep = np.concatenate([[True], np.diff(Q) > 1e-9])
    return np.interp(q_grid, Q[keep], V[keep])


def fit_ecm_to_curve(t, V, Q, i_amp, ocv_soc, ocv_v):
    """用阶段 A 的一阶 RC 结构去拟合高保真模型的放电曲线。

    模型：``V = OCV(SOC) − I·R0 − I·R1·(1 − e^(−t/τ))``，``t`` 为真实秒数，
    因此 τ 与阶段 A 的时间常数可直接比较。
    拟合结果与阶段 A 从 NASA 实测辨识出的参数对照，回答两个问题：
    ① 降阶到 1 阶 RC 会损失多少电压精度；② 实测辨识的内阻处于什么水平。
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    soc = 1.0 - Q / Q.max()
    ocv = np.interp(soc, ocv_soc, ocv_v)

    def resid(p):
        r0, r1, tau = p
        return (ocv - V) - i_amp * (r0 + r1 * (1 - np.exp(-t / max(tau, 1e-3))))

    sol = least_squares(resid, np.array([0.03, 0.02, 200.0]),
                        bounds=([0, 0, 1.0], [0.5, 0.5, 20000.0]))
    r0, r1, tau = sol.x
    fitted = ocv - i_amp * (r0 + r1 * (1 - np.exp(-t / max(tau, 1e-3))))
    rmse = float(np.sqrt(np.mean((fitted - V) ** 2)))
    return {"R0_ohm": float(r0), "R1_ohm": float(r1), "tau_s": float(tau)}, fitted, rmse


def part1_hierarchy():
    """三档模型 + 1 阶 RC 反演的端电压对比。

    对比的关键不是「谁更准」，而是把误差拆成**结构误差**与**参数误差**：
    - 结构误差：同一套参数下 SPM 相对 DFN 的偏差（≈17 mV，可忽略）；
    - 参数误差：同一结构（1 阶 RC）换一组内阻后的偏差（≈290 mV，致命）。
    """
    curves, rows = {}, []
    for kind, label, is_th in (("dfn", "DFN（全阶）", False),
                               ("spm", "SPM（单粒子）", False),
                               ("thevenin", "Thevenin（阶段 A 参数）", True)):
        t, V, Q, el = run_single_discharge(kind, CHEM, 1.0, thevenin=is_th)
        curves[kind] = (t, V, Q)
        rows.append({"model": label, "key": kind, "seconds": el, "n_points": len(V)})
        print(f"[D1] {label:24s} {el:5.2f}s  V=[{V.min():.3f}, {V.max():.3f}]")

    # 用阶段 A 的一阶 RC 结构反演 SPM 曲线
    soc, ocv = ocv_from_spm(CHEM)
    i_amp = 2.3  # Prada2013 额定 2.3 Ah，1C 即 2.3 A
    t_spm, V_spm, Q_spm = curves["spm"]
    fit, fitted, rmse = fit_ecm_to_curve(t_spm, V_spm, Q_spm, i_amp, soc, ocv)
    print(f"[D1] 一阶 RC 反演 SPM：R0={fit['R0_ohm'] * 1000:.1f} mΩ  "
          f"R1={fit['R1_ohm'] * 1000:.2f} mΩ  τ={fit['tau_s']:.0f} s  "
          f"RMSE={rmse * 1000:.1f} mV")

    # 用反演出的参数再跑一次 PyBaMM 的 Thevenin，隔离「结构」与「参数」两种误差
    from src.sim.models import thevenin_params  # 局部导入，避免顶层依赖

    t2, V2, Q2, el2 = run_single_discharge(
        "thevenin", CHEM, 1.0, thevenin=False,
        params_override=thevenin_params(CHEM, r0=fit["R0_ohm"],
                                        r1=max(fit["R1_ohm"], 1e-4),
                                        tau=fit["tau_s"]))
    curves["refit"] = (t2, V2, Q2)
    rows.append({"model": "Thevenin（反演 SPM 参数）", "key": "refit", "seconds": el2,
                 "n_points": len(V2), "R0_ohm": fit["R0_ohm"], "R1_ohm": fit["R1_ohm"]})

    # 以 DFN 为参照，比较各模型在同一放电深度下的电压偏差
    q_grid = np.linspace(0.02, 0.98, 200)
    ref = interp_to(curves["dfn"][1], curves["dfn"][2], q_grid)
    for r in rows:
        V = interp_to(curves[r["key"]][1], curves[r["key"]][2], q_grid)
        r["rmse_vs_dfn_mV"] = float(np.sqrt(np.nanmean((V - ref) ** 2)) * 1000)
        r["max_dev_mV"] = float(np.nanmax(np.abs(V - ref)) * 1000)

    # ---------- 图 d_01 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    styles = {"dfn": ("#2b6cb0", "DFN（全阶）", 2.4),
              "spm": ("#dd6b20", "SPM（单粒子）", 1.4),
              "thevenin": ("#38a169", "Thevenin（阶段 A 辨识参数）", 1.6),
              "refit": ("#805ad5", "Thevenin（反演 SPM 参数）", 1.2)}
    for k, (color, label, lw) in styles.items():
        t, V, Q = curves[k]
        ax.plot(Q / Q.max() * 100, V, color=color, lw=lw, label=label,
                ls="--" if k == "refit" else "-")
    ax.set_xlabel("放电深度 DOD / %")
    ax.set_ylabel("端电压 / V")
    ax.set_title("1C 恒流放电：三档保真度与 1 阶 RC 反演", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    order = ["dfn", "spm", "refit", "thevenin"]
    lab = {"dfn": "DFN\n(参照)", "spm": "SPM", "refit": "ECM\n(反演参数)",
           "thevenin": "ECM\n(阶段 A 参数)"}
    vals = [next(r["rmse_vs_dfn_mV"] for r in rows if r["key"] == k) for k in order]
    colors = ["#2b6cb0", "#dd6b20", "#805ad5", "#38a169"]
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, vals, color=colors, height=0.55)
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.02, y, f"{v:.0f} mV", va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab[k] for k in order], fontsize=8.5)
    ax.set_xlabel("相对 DFN 的电压 RMSE / mV")
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_title("结构误差 vs 参数误差", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_01_model_hierarchy.png")
    plt.close(fig)
    return rows, curves


def part2_rate():
    """倍率能力：不同 C 率下的放电曲线与容量利用率。"""
    rows, curves = [], {}
    for c in (0.5, 1.0, 2.0, 3.0, 4.0):
        t, V, Q, el = run_single_discharge("spm", CHEM, c)
        curves[c] = (V, Q)
        rows.append({"c_rate": c, "cap_Ah": float(Q.max()),
                     "energy_Wh": float(np.trapezoid(V, Q)),
                     "v_min": float(V.min()), "seconds": el})
        print(f"[D1] {c:g}C  容量={Q.max():.3f} Ah  能量={np.trapezoid(V, Q):.2f} Wh")

    base = rows[0]["cap_Ah"]
    for r in rows:
        r["utilization"] = r["cap_Ah"] / base

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, (c, (V, Q)) in enumerate(curves.items()):
        ax.plot(Q, V, lw=1.5, color=cmap(i / (len(curves) - 1)), label=f"{c:g}C")
    ax.set_xlabel("已放电容量 / Ah")
    ax.set_ylabel("端电压 / V")
    ax.set_title("SPM 不同倍率放电（LFP，25 ℃）", fontsize=10)
    ax.legend(frameon=False, fontsize=8, title="倍率", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    cs = [r["c_rate"] for r in rows]
    ax.plot(cs, [r["utilization"] * 100 for r in rows], "o-", color="#2b6cb0", ms=5)
    for c, r in zip(cs, rows):
        ax.annotate(f"{r['utilization'] * 100:.1f}%", (c, r["utilization"] * 100),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("放电倍率 / C")
    ax.set_ylabel("容量利用率（相对 0.5C） / %")
    ax.set_title("倍率越高，可用容量越少", fontsize=10)
    ax.set_ylim(min(r["utilization"] for r in rows) * 100 - 3, 102)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_02_rate_capability.png")
    plt.close(fig)
    return rows


def part3_fastcharge():
    """快充协议族：充电时长与温升（含集总热模型）。

    时长同时记录两个口径：
    - ``t80``：充到 80% 荷电量的时间，Severson 论文用的是这个；
    - ``t_total``：含恒压段的总时长。
    二者差距本身就是结论：t80 能从 45.2 min 压到 16.2 min（2.8 倍），总时长只能从
    85.4 min 压到 56.8 min（1.5 倍）——恒压段吃掉了大部分收益，所以「快充」的真实
    收益要看 t80，而不是插枪到拔枪的时间。
    """
    rows = []
    for p in PROTOCOLS:
        r = run_cycles(p, n_cycles=1, chemistry=CHEM_THERMAL, sei_scale=None, thermal=True)
        t80 = float(r.t80[0]) / 60.0 if np.isfinite(r.t80[0]) else np.nan
        t_tot = float(r.t_charge[0]) / 60.0
        dT = float(r.tmax[0] - 298.15) if np.isfinite(r.tmax[0]) else np.nan
        rows.append({"protocol": p.key, "label": p.label, "nominal_min": p.nominal_minutes,
                     "t80_min": t80, "t_total_min": t_tot, "dT_K": dT,
                     "max_c": p.max_c, "mean_c": p.mean_c, "seconds": r.seconds})
        print(f"[D1] {p.key:11s} 名义 {p.nominal_minutes:5.1f} min | "
              f"充到 80% {t80:5.1f} min | 总时长 {t_tot:5.1f} min | 温升 {dT:5.2f} K")

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    colors = plt.get_cmap("plasma")(np.linspace(0.12, 0.85, len(df)))
    ax.scatter(df["t80_min"], df["dT_K"], s=72, c=colors, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["protocol"], (r["t80_min"], r["dT_K"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=7.5)
    ax.set_xlabel("充到 80% 荷电量的时长 / min")
    ax.set_ylabel("最高温升 ΔT / K")
    ax.set_title("C 率越激进温升越高；place_high 与 base_1c 重合", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["t80_min"], width=0.38, color="#2b6cb0", label="充到 80% 时长")
    ax.bar(x + 0.2, df["t_total_min"], width=0.38, color="#90cdf4", label="含恒压段总时长")
    ax.set_xticks(x)
    ax.set_xticklabels(df["protocol"], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("时长 / min")
    ax.set_title("t80 压 2.8 倍，总时长只压 1.5 倍：恒压段吃掉收益", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "fig_d_03_fastcharge_heat.png")
    plt.close(fig)
    return rows


def main():
    h_rows, _ = part1_hierarchy()
    r_rows = part2_rate()
    f_rows = part3_fastcharge()

    pd.DataFrame(h_rows).to_csv(REP / "phaseD1_sim.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(r_rows).to_csv(REP / "phaseD1_rate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(f_rows).to_csv(REP / "phaseD1_protocols.csv", index=False, encoding="utf-8-sig")

    summary = {
        "hierarchy": h_rows,
        "rate": r_rows,
        "protocols": f_rows,
        "stage_a_ecm": STAGE_A_ECM,
    }
    (REP / "phaseD1_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[D1] 完成，图与表已写入 figures/ reports/")


if __name__ == "__main__":
    main()
