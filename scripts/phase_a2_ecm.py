"""阶段 A 加强版：ECM 参数辨识 + OCV 偏置标定 + 内阻老化轨迹。

相对初版（scripts/phase_a_ecm.py）的改进：
1. OCV 用 20 圈充放电平均 + PCHIP 单调平滑，降低分箱噪声；
2. 增加「OCV 偏置标定」：用训练圈残差拟合 3 阶多项式偏置 δ(SOC)，
   消除充放电平均法在低 SOC 段的系统偏差（这是初版 B0005 复测 RMSE 2.25% 的主因）；
3. 扩展到 4 颗电池（B0005/B0006/B0007/B0018）；
4. 新增内阻老化轨迹：逐圈 R0 与 R_dc(50%) 随 SOH 的演化，为阶段 C 的寿命特征提供物理解释。

验证口径：同工况复测电压 RMSE ≤ 实测量程 2%。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.ecm import ecm  # noqa: E402
from src.parsers import nasa  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

FIG = os.path.join(ROOT, "figures")
REP = os.path.join(ROOT, "reports")
CELLS = ["B0005", "B0006", "B0007", "B0018"]


def smooth_ocv(battery, n_cycles: int = 20):
    sg, og = ecm.build_ocv(battery, n_cycles=n_cycles, n_bins=100)
    good = np.isfinite(og)
    f = PchipInterpolator(sg[good], og[good], extrapolate=True)
    return sg, og, f


def rmse_pct(pred, meas):
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return rmse, rmse / (meas.max() - meas.min()) * 100


def discharge_cycles(battery):
    return [cc for _i, cc in nasa.iter_cycles(battery) if nasa.cycle_type(cc) == "discharge"]


NASA_DIR = Path(os.environ.get("NASA_DIR") or (ROOT / "data" / "raw" / "nasa"))


def cell_file(cell_id: str) -> Path:
    """Locate a NASA .mat file under data/raw/nasa (flat or extracted/)."""
    for p in nasa.all_nasa_files(NASA_DIR):
        if p.stem == cell_id:
            return p
    raise SystemExit(
        f"[错误] 找不到 {cell_id}.mat（搜索目录：{NASA_DIR}）。\n"
        f"       请先按 data/README.md 下载 NASA PCoE 数据集，"
        f"或用环境变量 NASA_DIR 指定路径。"
    )


def process_cell(cell_id: str) -> tuple[dict, dict]:
    path = cell_file(cell_id)
    _name, battery = nasa.load_battery(path)
    sg, og, ocv = smooth_ocv(battery, n_cycles=20)
    ds = discharge_cycles(battery)
    train, test = ds[0], ds[2]

    t, i, V, soc, i_dc = ecm.steady_region(train)
    tt, ii, VV, ssoc, _ = ecm.steady_region(test)
    t_cc, i_cc, V_cc, soc_cc, _ = ecm.cc_region(train)
    tt_cc, ii_cc, VV_cc, ssoc_cc, _ = ecm.cc_region(test)

    r0 = ecm.step_resistance(train)
    sc, rd = ecm.soc_resistance(train, ocv)
    rd_fn = lambda s: np.interp(s, sc, rd)  # noqa: E731

    # 未标定
    pred_tr = ocv(soc_cc) - i_cc * rd_fn(soc_cc)
    pred_te = ocv(ssoc_cc) - ii_cc * rd_fn(ssoc_cc)
    _, pct_tr_raw = rmse_pct(pred_tr, V_cc)
    _, pct_te_raw = rmse_pct(pred_te, VV_cc)

    # OCV 偏置标定（3 阶多项式，只用训练圈残差）
    resid = V_cc - pred_tr
    coef = np.polyfit(soc_cc, resid, 3)
    pred_tr_c = pred_tr + np.polyval(coef, soc_cc)
    pred_te_c = pred_te + np.polyval(coef, ssoc_cc)
    _, pct_tr_cal = rmse_pct(pred_tr_c, V_cc)
    _, pct_te_cal = rmse_pct(pred_te_c, VV_cc)

    row = {
        "cell": cell_id,
        "i_dc_A": i_dc,
        "R0_step_mohm": r0 * 1000,
        "Rdc50_mohm": float(rd_fn(0.5) * 1000),
        "Rpol50_mohm": float((rd_fn(0.5) - r0) * 1000),
        "Rdc80_mohm": float(rd_fn(0.8) * 1000),
        "Rdc20_mohm": float(rd_fn(0.2) * 1000),
        "rmse_train_raw_pct": pct_tr_raw,
        "rmse_test_raw_pct": pct_te_raw,
        "rmse_train_cal_pct": pct_tr_cal,
        "rmse_test_cal_pct": pct_te_cal,
    }

    # ---- 内阻老化轨迹：每 10 圈取一次 ----
    aging = []
    cap0 = nasa.discharge_capacity(ds[0])
    for k in range(0, len(ds), 10):
        cc = ds[k]
        try:
            r0k = ecm.step_resistance(cc)
            sck, rdk = ecm.soc_resistance(cc, ocv)
            capk = nasa.discharge_capacity(cc)
        except Exception:
            continue
        aging.append({"cell": cell_id, "cycle": k + 1, "R0_mohm": r0k * 1000,
                      "Rdc50_mohm": float(np.interp(0.5, sck, rdk) * 1000),
                      "capacity_Ah": capk, "SOH": capk / cap0})
    figs = {"sg": sg, "og": og, "ocv": ocv, "aging": pd.DataFrame(aging),
            "train": (t_cc, i_cc, V_cc, soc_cc, pred_tr_c),
            "test": (tt_cc, ii_cc, VV_cc, ssoc_cc, pred_te_c)}
    return row, figs


def main() -> int:
    rows, all_aging = [], []
    for cell in CELLS:
        r, figs = process_cell(cell)
        rows.append(r)
        all_aging.append(figs["aging"])
        print(f"{cell}: R0={r['R0_step_mohm']:.2f}mΩ  "
              f"复测RMSE 原始 {r['rmse_test_raw_pct']:.2f}% → 标定 {r['rmse_test_cal_pct']:.2f}%")

        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        t, i, V, s, pred = figs["train"]
        ax.plot(s, V, "k.", ms=2, label="实测（辨识圈）")
        ax.plot(s, pred, lw=1.6, color="tab:blue", label="SOC 依赖内阻模型 + OCV 标定")
        ax.set_xlabel("SOC")
        ax.set_ylabel("端电压 / V")
        ax.set_title(f"NASA {cell} — 恒流放电拟合（RMSE {r['rmse_train_cal_pct']:.2f}%）")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig_a2_fit_{cell}.png"))
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REP, "phaseA2_ecm_results.csv"), index=False, encoding="utf-8-sig")

    aging = pd.concat(all_aging, ignore_index=True)
    aging.to_csv(os.path.join(REP, "phaseA2_aging.csv"), index=False, encoding="utf-8-sig")

    # ---- 图：内阻老化轨迹 ----
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for cell, g in aging.groupby("cell"):
        axes[0].plot(g["SOH"] * 100, g["R0_mohm"], "o-", ms=3, lw=1.2, label=cell)
        axes[1].plot(g["SOH"] * 100, g["Rdc50_mohm"], "o-", ms=3, lw=1.2, label=cell)
    axes[0].set_xlabel("SOH / %")
    axes[0].set_ylabel("R0 / mΩ")
    axes[0].set_title("欧姆内阻随老化上升")
    axes[1].set_xlabel("SOH / %")
    axes[1].set_ylabel("R_dc(50%) / mΩ")
    axes[1].set_title("总直流内阻随老化上升")
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_a2_r_aging.png"))
    plt.close(fig)

    # ---- 图：OCV 曲线 ----
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for cell in CELLS:
        path = os.path.join(ROOT, "data", "raw", "nasa", "extracted", f"{cell}.mat")
        _n, b = nasa.load_battery(path)
        sg, og, f = smooth_ocv(b, n_cycles=20)
        ax.plot(sg, f(sg), lw=1.4, label=cell)
    ax.set_xlabel("SOC")
    ax.set_ylabel("OCV / V")
    ax.set_title("OCV-SOC 曲线（20 圈平均 + PCHIP 平滑）")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_a2_ocv_soc.png"))
    plt.close(fig)

    print("\n" + df.round(2).to_string(index=False))
    print(f"\n复测 RMSE 中位：原始 {df['rmse_test_raw_pct'].median():.2f}% → "
          f"标定后 {df['rmse_test_cal_pct'].median():.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
