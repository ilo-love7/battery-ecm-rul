"""阶段 D2：物理仿真 × 数据驱动（P2 与 P1 的结合）。

阶段 D1 复现了 PyBaMM 的基准曲线；本脚本回答一个更硬的问题：

    **P1 里那三个 ΔQ(V) 特征，到底编码了什么？**

三条证据链：

1. **退化模式归因**：在 LFP 参数集上分别注入 LLI（锂库存损失）、
   LAM_pos / LAM_neg（正/负极活性物质损失），看每种模式把 ΔQ(V) 曲线
   掰成什么形状，再与 MIT 实测的 ΔQ(V) 形态比对，判断真实电池以哪种模式为主。
2. **协议能否决定寿命**：7 条快充协议分别用「纯 SEI」与「SEI+析锂」两种
   机制跑老化，对比协议间的寿命离散度；再与 MIT 实测中**同一协议**的
   电池寿命离散度对照。
3. **把物理先验喂回 P1**：以三种退化模式的 ΔQ(V) 形状为基，对每颗真实
   电池的 ΔQ(V) 做最小二乘分解，得到 3 个「退化模式系数」，作为可解释
   特征加入 P1 模型，检验主测试集与 batch3 是否改善。

产物：
- figures/fig_d_04_degradation_modes.png
- figures/fig_d_05_protocol_life.png
- figures/fig_d_06_mode_features.png
- reports/phaseD2_*.csv

运行：
    python scripts/phase_d2_sim_rul.py
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
from scipy.stats import spearmanr  # noqa: E402

from src.rul.features import build_features, feature_sets  # noqa: E402
from src.rul.models import make_elasticnet, make_gbm, metrics  # noqa: E402
from src.sim.models import load_params  # noqa: E402
from src.sim.protocols import PROTOCOLS, BY_KEY  # noqa: E402
from src.sim.runner import V_GRID, qd_from_sim, run_cycles, run_single_discharge  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SEED = 42
CHEM = "lfp"
VG = V_GRID[CHEM]
LEVELS = (0.03, 0.06, 0.10)

#: 三种退化模式 -> (中文名, 被改动的 PyBaMM 参数)
MODES = {
    "LLI": ("锂库存损失 LLI", "Initial concentration in negative electrode [mol.m-3]"),
    "LAM_pos": ("正极活性物质损失 LAM_pos", "Positive electrode active material volume fraction"),
    "LAM_neg": ("负极活性物质损失 LAM_neg", "Negative electrode active material volume fraction"),
}
COLORS = {"LLI": "#2b6cb0", "LAM_pos": "#dd6b20", "LAM_neg": "#38a169"}


def degrade_params(mode: str, level: float):
    """按比例削减某个退化模式对应的参数，返回一份新的参数集。"""
    key = MODES[mode][1]
    p = load_params(CHEM)
    p[key] = p[key] * (1.0 - level)
    return p


def smooth(y, w: int = 9):
    """滑动平均，抑制 Q(V) 插值在「初始 IR 突降」处的伪影。

    仿真的放电曲线一开始有个几毫伏的欧姆跳变，把大量容量点压到同一个
    电压上；插值到公共栅格后会形成一条竖直伪影。窗口取 9 点（≈45 mV），
    远小于 LFP 平台段的尺度，不影响形态判断。
    """
    y = np.asarray(y, dtype=float)
    if y.size < w:
        return y
    k = np.ones(w) / w
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(ypad, k, mode="valid")


def partA_modes():
    """三种退化模式对 ΔQ(V) 的影响，与真实数据形态对照。"""
    _, V0, Q0, _ = run_single_discharge("spm", CHEM, 1.0)
    qd_fresh = qd_from_sim(V0, Q0, VG)

    curves, rows = {}, []
    for mode in MODES:
        curves[mode] = []
        for lv in LEVELS:
            _, V, Q, _ = run_single_discharge("spm", CHEM, 1.0,
                                              params_override=degrade_params(mode, lv))
            qd = qd_from_sim(V, Q, VG)
            dq = smooth(qd - qd_fresh)              # 与 P1 同口径：跨循环、同方向
            curves[mode].append(dq)
            rows.append({
                "mode": mode, "level": lv,
                "dq_depth_mAh": float(np.nanmax(np.abs(dq)) * 1000),
                "cap_loss_pct": float((1 - Q.max() / Q0.max()) * 100),
            })
            print(f"[D2] {mode:8s} level={lv:.0%}  ΔQ 深度={rows[-1]['dq_depth_mAh']:6.1f} mAh  "
                  f"容量损失={rows[-1]['cap_loss_pct']:5.1f}%")

    # 真实数据的 ΔQ(V)：第 100 圈 − 第 10 圈
    qd = np.load(PROC / "mit_qd.npz", allow_pickle=True)["qd"]
    dq_real_all = qd[:, 98, :] - qd[:, 8, :]
    dq_real = smooth(np.nanmedian(dq_real_all, axis=0))
    print(f"[D2] 真实 ΔQ(V) 中位曲线：深度 {np.nanmax(np.abs(dq_real)) * 1000:.1f} mAh")

    # 形态比较：各自按最大绝对值归一化后求相关
    def norm(x):
        y = np.asarray(x, dtype=float)
        m = np.nanmax(np.abs(y))
        return y / m if m > 0 else y

    corr = {}
    for mode in MODES:
        c = norm(curves[mode][-1])
        r = float(np.corrcoef(norm(dq_real), c)[0, 1])
        corr[mode] = r
        rows.append({"mode": mode, "level": np.nan, "corr_with_real": r})
        print(f"[D2] 形态相关：{mode:8s} vs 真实  r={r:+.3f}")

    # ---------- 图 d_04 ----------
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), sharey=False)
    for ax, mode in zip(axes, MODES):
        ymax = np.nanmax(np.abs(curves[mode][-1]))
        for i, lv in enumerate(LEVELS):
            ax.plot(VG, curves[mode][i] * 1000, lw=1.4,
                    color=COLORS[mode], alpha=0.35 + 0.28 * i, label=f"−{lv:.0%}")
        ax.plot(VG, norm(dq_real) * ymax * 1000, "k--", lw=1.5, label="MIT 实测（同幅归一）")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"{MODES[mode][0]}\n（与实测形态 r={corr[mode]:+.2f}）", fontsize=9)
        ax.set_xlabel("端电压 / V")
        ax.legend(frameon=False, fontsize=7.5, title="参数削减", title_fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ΔQ(V) / mAh")
    fig.suptitle("三种退化模式把 ΔQ(V) 掰成不同形状——实测形态更像哪一种？",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_04_degradation_modes.png", bbox_inches="tight")
    plt.close(fig)

    return curves, dq_real, pd.DataFrame(rows), corr


def partB_protocol_life():
    """协议能否决定寿命：纯 SEI vs SEI+析锂。"""
    rows = []
    for p in PROTOCOLS:
        for tag, kw in (("SEI", dict(chemistry="nmc", sei_scale=1e3, plating=False)),
                        ("SEI+析锂", dict(chemistry="nmc_plate", sei_scale=1e3, plating=True))):
            n = 30  # 两种机制跑同样圈数，损失率才可比
            r = run_cycles(p, n_cycles=n, **kw)
            ret = float(r.cap[-1] / r.cap[0]) if r.cap[0] > 0 else np.nan
            t80 = float(np.nanmean(r.t80)) / 60.0
            rows.append({"protocol": p.key, "mechanism": tag, "n_cycles": n,
                         "retention": ret, "loss_pct": (1 - ret) * 100,
                         "t80_min": t80, "life_sim": r.life,
                         "seconds": r.seconds})
            print(f"[D2] {p.key:11s} {tag:8s} {n} 圈  保持率={ret * 100:6.2f}%  "
                  f"（损失 {rows[-1]['loss_pct']:5.2f}%）")

    df = pd.DataFrame(rows)

    # 真实数据：同一协议内 vs 全体的寿命离散度
    cells = pd.read_parquet(PROC / "mit_cells.parquet")
    pol_cols = [f"policy_{i}" for i in range(15)]
    cells["pol"] = [tuple(r) for r in cells[pol_cols].values]
    grp = cells.groupby("pol")["cycle_life"].agg(["count", "min", "max", "mean"])
    grp = grp[grp["count"] >= 2]
    within = float(((grp["max"] - grp["min"]) / grp["mean"]).median())
    overall = float((cells["cycle_life"].max() - cells["cycle_life"].min())
                    / cells["cycle_life"].mean())
    print(f"[D2] 真实数据：同协议组内极差/均值 = {within:.2f}，"
          f"全体极差/均值 = {overall:.2f}（{len(grp)} 组、{int(grp['count'].sum())} 颗）")

    # ---------- 图 d_05 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    keys = [p.key for p in PROTOCOLS]
    x = np.arange(len(keys))
    for off, tag, color in ((-0.2, "SEI", "#2b6cb0"), (0.2, "SEI+析锂", "#c53030")):
        sub = df[df["mechanism"] == tag].set_index("protocol").loc[keys]
        ax.bar(x + off, sub["loss_pct"], width=0.38, color=color, label=tag)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("30 圈后的容量损失 / %")
    ax.set_title("两种机制下协议都几乎无差别——「协议→寿命」未复现", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    sei = df[df["mechanism"] == "SEI"].set_index("protocol").loc[keys]["loss_pct"]
    plate = df[df["mechanism"] == "SEI+析锂"].set_index("protocol").loc[keys]["loss_pct"]
    spread_sim = float((sei.max() - sei.min()) / sei.mean())
    spread_sim2 = float((plate.max() - plate.min()) / plate.mean())
    vals = [spread_sim, spread_sim2, within, overall]
    labs = ["仿真\n(纯 SEI)", "仿真\n(SEI+析锂)", "真实·同协议\n组内", "真实·全体\n跨协议"]
    cols = ["#2b6cb0", "#c53030", "#38a169", "#805ad5"]
    ax.bar(range(4), vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("寿命离散度（极差 / 均值）")
    ax.set_title("仿真与实测的寿命离散度对比", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_05_protocol_life.png")
    plt.close(fig)

    return df, {"within_policy": within, "overall": overall,
                "sim_spread_sei": spread_sim, "sim_spread_plating": spread_sim2}


def _tune_en(X, y, alphas=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """训练集内部 5 折 CV 选 ElasticNet 的 alpha（与阶段 C 完全同一套流程）。

    每个实验臂各自调参，避免「基线用旧超参、增强臂用新超参」这种不对等比较。
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best, best_score = alphas[0], np.inf
    for a in alphas:
        errs = []
        for tri, vai in kf.split(X):
            m = make_elasticnet(alpha=a).fit(X[tri], np.log(y[tri]))
            p = np.exp(m.predict(X[vai]))
            errs.append(float(np.median(np.abs(p - y[vai]) / y[vai])))
        s = float(np.mean(errs))
        if s < best_score:
            best, best_score = a, s
    return best


def partC_mode_features(curves):
    """把仿真的三种退化模式当作基，对真实 ΔQ(V) 做分解，再喂回 P1。"""
    # 基函数：取最大退化档的形状，单位化
    basis, names = [], []
    for mode in MODES:
        b = np.asarray(curves[mode][-1], dtype=float)
        m = np.nanmax(np.abs(b))
        basis.append(b / m if m > 0 else b)
        names.append(mode)
    B = np.stack(basis, axis=1)                     # (300, 3)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)

    qd = np.load(PROC / "mit_qd.npz", allow_pickle=True)["qd"]
    dq_real = np.stack([smooth(qd[i, 98, :] - qd[i, 8, :]) for i in range(qd.shape[0])])

    coefs = np.full((dq_real.shape[0], 3), np.nan)
    for i in range(dq_real.shape[0]):
        y = dq_real[i]
        m = np.isfinite(y)
        if m.sum() < 30:
            continue
        c, *_ = np.linalg.lstsq(B[m], y[m], rcond=None)
        coefs[i] = c
    # 归一化成「各模式的相对占比」，消除整体幅度的影响
    tot = np.nansum(np.abs(coefs), axis=1, keepdims=True)
    share = coefs / np.where(tot > 0, tot, np.nan)

    # ---------- 重建特征表 ----------
    d = np.load(PROC / "mit_dq.npz", allow_pickle=True)
    cube, ids, vgrid = d["dq"], d["cell_ids"], d["v_grid"]
    cells = pd.read_parquet(PROC / "mit_cells.parquet")
    cycles = pd.read_parquet(PROC / "mit_cycles.parquet")
    feat = build_features(cube, ids, vgrid, cycles, cells, qd_cube=qd)
    order = cells["cell_id"].tolist()
    feat = feat.set_index("cell_id").loc[order].reset_index()

    for j, nm in enumerate(names):
        feat[f"mode_{nm}"] = share[:, j]
        feat[f"mode_{nm}_abs"] = coefs[:, j]

    y = feat["cycle_life"].values.astype(float)
    _ok = np.isfinite(share[:, 0]) & np.isfinite(y)
    r = float(spearmanr(share[_ok, 0], y[_ok])[0]) if _ok.sum() > 5 else float("nan")
    print(f"[D2] LLI 占比 vs 寿命：Spearman ρ={r:+.3f}"
          f"（占比标准差 {np.nanstd(share[:, 0]):.3f}，区分度低）")
    split = feat["split"].values
    tr, te, sec = split == "train", split == "test", split == "secondary_test"
    p6 = feature_sets(feat)["paper6"]
    mode_cols = [f"mode_{n}" for n in names]

    arms = {
        "P1 基线（论文 6 特征）": p6,
        "P1 + 仿真模式系数": p6 + mode_cols,
        "仅仿真模式系数": mode_cols,
    }
    rows = []
    for arm, cols in arms.items():
        X = feat[cols].values.astype(float)
        alpha = _tune_en(X[tr], y[tr])
        for nm, mk in (("ElasticNet", lambda a=alpha: make_elasticnet(alpha=a)),
                       ("GradientBoosting", make_gbm)):
            m = mk().fit(X[tr], np.log(y[tr]))
            pred = np.clip(np.exp(m.predict(X)), 50, 3000)
            for tag, mask in (("test", te), ("secondary", sec)):
                rows.append({"arm": arm, "model": nm, "eval_set": tag,
                             "n_features": len(cols), "alpha": alpha,
                             **metrics(y[mask], pred[mask])})
        print(f"[D2] {arm:22s} n_feat={len(cols):2d}  alpha={alpha:g}")
    res = pd.DataFrame(rows)
    print("[D2] 仿真模式系数增强（中位相对误差 %）：")
    print(res.pivot_table(index=["arm", "model"], columns="eval_set",
                          values="median_ape").round(2))

    # ---------- 图 d_06 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    conds = [("ElasticNet", "test", "EN·主测试"), ("ElasticNet", "secondary", "EN·batch3"),
             ("GradientBoosting", "test", "GBM·主测试"),
             ("GradientBoosting", "secondary", "GBM·batch3")]
    x = np.arange(len(conds))
    for off, arm, color in ((-0.2, "P1 基线（论文 6 特征）", "#2b6cb0"),
                            (0.2, "P1 + 仿真模式系数", "#dd6b20")):
        vals = [float(res[(res.arm == arm) & (res.model == m) &
                          (res.eval_set == s)]["median_ape"].iloc[0])
                for m, s, _ in conds]
        ax.bar(x + off, vals, width=0.38, color=color, label=arm.replace("P1 ", ""))
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in conds], fontsize=8.5)
    ax.set_ylabel("中位相对误差 / %")
    ax.set_title("加入仿真模式系数：四种组合全部没有改善", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    data = [share[:, j] for j in range(3)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, mode in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[mode])
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([MODES[m][0].split(" ")[0] for m in names], fontsize=8.5)
    ax.set_ylabel("ΔQ(V) 的模式投影系数（按绝对值归一）")
    ax.set_title("124 颗电池的分解结果几乎一致——没有区分度", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_06_mode_features.png")
    plt.close(fig)

    return res, feat, {"llI_spearman": float(r), "modes": names,
                       "share_std": {n: float(np.nanstd(share[:, j]))
                                     for j, n in enumerate(names)}}


def main():
    curves, dq_real, mode_df, corr = partA_modes()
    proto_df, spread = partB_protocol_life()
    aug_df, feat, extra = partC_mode_features(curves)

    mode_df.to_csv(REP / "phaseD2_modes.csv", index=False, encoding="utf-8-sig")
    proto_df.to_csv(REP / "phaseD2_protocols.csv", index=False, encoding="utf-8-sig")
    aug_df.to_csv(REP / "phaseD2_augment.csv", index=False, encoding="utf-8-sig")

    summary = {
        "mode_corr": corr,
        "protocol_spread": spread,
        "augment": aug_df.to_dict(orient="records"),
        "mode_spearman": extra,
    }
    (REP / "phaseD2_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[D2] 完成，图与表已写入 figures/ reports/")


if __name__ == "__main__":
    main()
