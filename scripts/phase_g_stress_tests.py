"""阶段 G：受数据阻塞的两个补充实验（数据于 2026-10-18 经 junction 解锁）。

实验一 —— 早期预测窗口（N<100）：
    固定「只使用前 N 圈」（N=20/50/80/100），用 window-relative 特征重训
    ElasticNet / RandomForest / GradientBoosting，测主测试集与二级测试集的中位 APE，
    量化「早测」相对满窗口（N=100）的代价。对应 applicability.md §2.1 / §6 待补项。
    说明：论文口径 xq_* 特征（Q_100−Q_10）结构性依赖第 100 圈，因此 N<100 时
    这三个特征不可用，窗口特征集为 full 去掉全 NaN 列后的 window-relative 子集。

实验二 —— 噪声注入压力测试：
    在 ΔQ(V) 立方体上叠加高斯测量噪声（σ 取多个量级，单位 Ah），重训并重测，
    量化「测量精度劣化」对 RUL 中位 APE 的影响。对应 applicability.md §2.2 / §7.3 待补项。
    这是测量噪声的一阶代理（在已提取的 ΔQ(V) 曲线层注入）；更忠实的测试需回到
    原始 V/I/Q 注入，但本实验足以给出鲁棒性量级结论。σ 同时以「相对信号标准差」
    给出，便于解读。

产物：
    reports/phaseG_window.csv     早期窗口各 N 的中位 APE
    reports/phaseG_noise.csv      噪声各 σ 的中位 APE
    reports/phaseG_results.json   汇总
    figures/fig_g_01_window.png   早期窗口曲线
    figures/fig_g_02_noise.png    噪声鲁棒性曲线
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from src.rul.features import build_features, feature_columns  # noqa: E402
from src.rul.models import (  # noqa: E402
    make_elasticnet,
    make_gbm,
    make_rf,
    metrics,
)

PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SEED = 42


def load_mit():
    d = np.load(PROC / "mit_dq.npz", allow_pickle=True)
    dq, ids, vgrid = d["dq"], d["cell_ids"], d["v_grid"]
    qd = np.load(PROC / "mit_qd.npz", allow_pickle=True)["qd"]
    cells = pd.read_parquet(PROC / "mit_cells.parquet")
    cycles = pd.read_parquet(PROC / "mit_cycles.parquet")
    return dq, qd, ids, vgrid, cells, cycles


def prep(feat: pd.DataFrame):
    """丢掉全 NaN 列（N<100 时 xq_* 会全 NaN），返回 X/y/split/列名。"""
    fcols = [c for c in feature_columns(feat) if not feat[c].isna().all()]
    X = feat[fcols].values.astype(float)
    y = feat["cycle_life"].values.astype(float)
    split = feat["split"].values
    return X, y, split, fcols


def tune_en(X_tr, y_tr, alphas=(0.01, 0.03, 0.1, 0.3),
            l1_ratios=(0.5, 0.9, 1.0)):
    best, best_score = (0.1, 0.5), np.inf
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    for a in alphas:
        for lr in l1_ratios:
            errs = []
            for tri, vai in kf.split(X_tr):
                m = make_elasticnet(alpha=a, l1_ratio=lr)
                m.fit(X_tr[tri], np.log(y_tr[tri]))
                p = np.exp(m.predict(X_tr[vai]))
                errs.append(np.median(np.abs(p - y_tr[vai]) / y_tr[vai]))
            s = float(np.mean(errs))
            if s < best_score:
                best, best_score = (a, lr), s
    return best[0], best[1]


def train_eval(X, y, split):
    tr = split == "train"
    te = split == "test"
    sec = split == "secondary_test"
    out = {}
    a, lr = tune_en(X[tr], y[tr])
    for name, mk in (
        ("ElasticNet", lambda a=a, lr=lr: make_elasticnet(alpha=a, l1_ratio=lr)),
        ("RandomForest", make_rf),
        ("GradientBoosting", make_gbm),
    ):
        m = mk().fit(X[tr], np.log(y[tr]))
        p = np.clip(np.exp(m.predict(X)), 50, 3000)
        out[f"{name}_test"] = float(metrics(y[te], p[te])["median_ape"])
        out[f"{name}_secondary"] = float(metrics(y[sec], p[sec])["median_ape"])
    out["en_alpha"], out["en_l1_ratio"] = a, lr
    return out


def run_window(dq, qd, ids, vgrid, cells, cycles):
    print("[G1] 早期预测窗口实验（N=20/50/80/100）")
    rows = []
    for N in (20, 50, 80, 100):
        # 早期窗口：只观测到前 N 圈，因此 qd 立方体也截断到前 N 圈，
        # 让结构性依赖第 100 圈的 xq_* 特征在 N<100 时正确地变为 NaN。
        feat = build_features(dq, ids, vgrid, cycles, cells,
                              n_cycle_window=N, qd_cube=qd[:, :N, :])
        X, y, split, fcols = prep(feat)
        r = train_eval(X, y, split)
        r.update({"n_window": N, "n_features": len(fcols),
                  "xq_available": int("xq_log_var" in fcols)})
        rows.append(r)
        print(f"  N={N:3d} 特征={len(fcols):2d} xq={'Y' if r['xq_available'] else 'N'} "
              f"GBM(test={r['GradientBoosting_test']:.2f}%, sec={r['GradientBoosting_secondary']:.2f}%) "
              f"RF(test={r['RandomForest_test']:.2f}%, sec={r['RandomForest_secondary']:.2f}%) "
              f"EN(test={r['ElasticNet_test']:.2f}%, sec={r['ElasticNet_secondary']:.2f}%)")
    return rows


def run_noise(dq, qd, ids, vgrid, cells, cycles):
    print("[G2] 噪声注入压力测试（σ 多量级，单位 Ah）")
    sig_finite = np.nanstd(dq[np.isfinite(dq)])
    print(f"  ΔQ(V) 信号标准差（有限值）≈ {sig_finite:.4f} Ah")
    rows = []
    base = 20261018
    for sigma in (0.0, 0.001, 0.003, 0.01, 0.03):
        rng = np.random.default_rng(base + int(round(sigma * 1000)))
        dq_n = (dq + rng.normal(0, sigma, dq.shape)).astype(np.float32)
        qd_n = (qd + rng.normal(0, sigma, qd.shape)).astype(np.float32)
        feat = build_features(dq_n, ids, vgrid, cycles, cells,
                             n_cycle_window=100, qd_cube=qd_n)
        X, y, split, fcols = prep(feat)
        r = train_eval(X, y, split)
        r.update({"sigma_Ah": sigma, "sigma_rel": sigma / sig_finite,
                  "n_features": len(fcols)})
        rows.append(r)
        print(f"  σ={sigma:.4f} Ah (×{sigma / sig_finite:5.1f} 信号σ) "
              f"GBM(test={r['GradientBoosting_test']:.2f}%, sec={r['GradientBoosting_secondary']:.2f}%) "
              f"RF(test={r['RandomForest_test']:.2f}%, sec={r['RandomForest_secondary']:.2f}%) "
              f"EN(test={r['ElasticNet_test']:.2f}%, sec={r['ElasticNet_secondary']:.2f}%)")
    return rows, sig_finite


def fig_window(rows):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors = {"ElasticNet": "#2b6cb0", "RandomForest": "#dd6b20",
              "GradientBoosting": "#38a169"}
    markers = {"test": "o", "secondary": "^"}
    for model, c in colors.items():
        for tag, mk in markers.items():
            ax.plot(df["n_window"], df[f"{model}_{tag}"], marker=mk, lw=1.6,
                    color=c, alpha=0.85 if tag == "test" else 0.55,
                    label=f"{model}（{'主测试' if tag == 'test' else '二级测试'}）")
    ax.axhline(10, ls="--", lw=1, color="#c53030", label="Severson 目标 10%")
    ax.set_xlabel("使用前 N 圈（早期预测窗口）")
    ax.set_ylabel("中位相对误差 / %")
    ax.set_xticks([20, 50, 80, 100])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g_01_window.png")
    plt.close(fig)


def fig_noise(rows):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors = {"ElasticNet": "#2b6cb0", "RandomForest": "#dd6b20",
              "GradientBoosting": "#38a169"}
    for model, c in colors.items():
        ax.plot(df["sigma_rel"], df[f"{model}_test"], marker="o", lw=1.6,
                color=c, label=model)
    ax.axvline(1.0, ls=":", lw=1, color="#718096",
               label="σ = 信号标准差（噪声≈传感器量级）")
    ax.set_xscale("log")
    ax.set_xlabel("注入噪声 σ（相对 ΔQ(V) 信号标准差）")
    ax.set_ylabel("主测试集中位相对误差 / %")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g_02_noise.png")
    plt.close(fig)


def main():
    t0 = time.time()
    dq, qd, ids, vgrid, cells, cycles = load_mit()
    window = run_window(dq, qd, ids, vgrid, cells, cycles)
    noise, sig_finite = run_noise(dq, qd, ids, vgrid, cells, cycles)

    wdf = pd.DataFrame(window)
    ndf = pd.DataFrame(noise)
    wdf.to_csv(REP / "phaseG_window.csv", index=False, encoding="utf-8-sig")
    ndf.to_csv(REP / "phaseG_noise.csv", index=False, encoding="utf-8-sig")
    fig_window(window)
    fig_noise(noise)

    # 相对基准：N=100 / σ=0 的 GBM/RF/EN 主测试误差
    base_gbm = float(wdf.loc[wdf.n_window == 100, "GradientBoosting_test"].iloc[0])
    base_rf = float(wdf.loc[wdf.n_window == 100, "RandomForest_test"].iloc[0])
    base_en = float(wdf.loc[wdf.n_window == 100, "ElasticNet_test"].iloc[0])
    noise_base_gbm = float(ndf.loc[ndf.sigma_Ah == 0.0, "GradientBoosting_test"].iloc[0])
    noise_base_rf = float(ndf.loc[ndf.sigma_Ah == 0.0, "RandomForest_test"].iloc[0])
    noise_base_en = float(ndf.loc[ndf.sigma_Ah == 0.0, "ElasticNet_test"].iloc[0])

    summary = {
        "signal_std_Ah": float(sig_finite),
        "window": wdf.to_dict(orient="records"),
        "noise": ndf.to_dict(orient="records"),
        "baseline_N100_test_median_ape": {
            "GradientBoosting": base_gbm, "RandomForest": base_rf,
            "ElasticNet": base_en},
        "baseline_sigma0_test_median_ape": {
            "GradientBoosting": noise_base_gbm, "RandomForest": noise_base_rf,
            "ElasticNet": noise_base_en},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (REP / "phaseG_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[G] 完成（{summary['elapsed_s']}s）。图与表已写入 figures/ reports/")
    print(f"    基准 N=100 主测试：GBM={base_gbm:.2f}% RF={base_rf:.2f}% EN={base_en:.2f}%")
    print(f"    基准 σ=0 主测试：GBM={noise_base_gbm:.2f}% RF={noise_base_rf:.2f}% EN={noise_base_en:.2f}%")


if __name__ == "__main__":
    main()
