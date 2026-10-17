"""阶段 C：RUL（剩余寿命 / 循环寿命）预测建模。

协议完全对齐 Severson 2019：
- 训练集 41 颗（batch1+2 偶数位）、主测试集 43 颗、二级测试集 40 颗（batch3）；
- 只使用前 100 圈数据做「早期寿命预测」；
- 目标：测试集相对误差中位数（median APE）。

模型：
1. ElasticNet（论文同款基线，log 目标 + 标准化）
2. RandomForest
3. GradientBoosting
4. 1D CNN（3 通道 ΔQ(V) + 标量特征，5 个种子集成）

产物：
- reports/phaseC_metrics.csv / phaseC_predictions.csv
- figures/fig_c_*.png
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
from sklearn.model_selection import KFold  # noqa: E402

from src.rul.features import build_features, feature_columns, feature_sets  # noqa: E402
from src.rul.models import (  # noqa: E402
    DQCNN, make_elasticnet, make_gbm, make_mlp, make_rf, metrics,
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


def load_all():
    d = np.load(PROC / "mit_dq.npz", allow_pickle=True)
    cube, ids, vgrid = d["dq"], d["cell_ids"], d["v_grid"]
    qd_cube = None
    qd_path = PROC / "mit_qd.npz"
    if qd_path.exists():
        qd_cube = np.load(qd_path, allow_pickle=True)["qd"]
    else:
        print("[C][警告] 未找到 mit_qd.npz，论文口径 xq_* 特征将不可用；"
              "请先运行 scripts/phase_b_build_dataset.py mit")
    cells = pd.read_parquet(PROC / "mit_cells.parquet")
    cycles = pd.read_parquet(PROC / "mit_cycles.parquet")
    feat = build_features(cube, ids, vgrid, cycles, cells, qd_cube=qd_cube)
    order = cells["cell_id"].tolist()
    feat = feat.set_index("cell_id").loc[order].reset_index()
    return cube, ids, vgrid, cells, feat


def curve_tensor(cube: np.ndarray) -> np.ndarray:
    """(n,100,300) -> (n,3,300)：第 2 圈 / 第 100 圈 / 两者差值。"""
    c2 = cube[:, 0, :]
    c100 = cube[:, -1, :]
    return np.stack([c2, c100, c2 - c100], axis=1).astype(np.float32)


def tune_elasticnet(X, y, alphas=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
                    l1_ratios=(0.1, 0.5, 0.9, 1.0)):
    """在训练集内部做 5 折 CV 同时选 alpha 与 l1_ratio（不外泄测试集）。

    l1_ratio 必须一起搜：固定 0.5 时 EN 只是岭回归与 Lasso 的折中，而论文六特征
    是稀疏信号，l1_ratio→1（纯 Lasso）才能做特征选择。另一份独立复现
    （battery-lifetime-guide）用 `ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0])`，
    同划分下 EN 六特征 5.9%；本项目早期版本固定 l1_ratio=0.5，同配置为 8.33%，
    而两者的随机森林完全一致（8.12% vs 8.2%）——证明数据口径与标签都对，
    差距全在超参搜索空间。详见 docs/results.md 第六节。
    """
    best, best_score = None, np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for a in alphas:
        for lr in l1_ratios:
            errs = []
            for tr, va in kf.split(X):
                m = make_elasticnet(alpha=a, l1_ratio=lr)
                m.fit(X[tr], np.log(y[tr]))
                pred = np.exp(m.predict(X[va]))
                errs.append(np.median(np.abs(pred - y[va]) / y[va]))
            score = float(np.mean(errs))
            if score < best_score:
                best, best_score = (a, lr), score
    return best[0], best[1], best_score


def main():
    cube, ids, vgrid, cells, feat = load_all()
    fcols = feature_columns(feat)
    print(f"[C] 特征数 = {len(fcols)}，样本 = {len(feat)}")

    X = feat[fcols].values.astype(float)
    y = feat["cycle_life"].values.astype(float)
    split = feat["split"].values
    tr, te, sec = split == "train", split == "test", split == "secondary_test"
    print(f"[C] train={tr.sum()} test={te.sum()} secondary={sec.sum()}")

    curves = curve_tensor(cube)
    results, preds = [], {}

    # ---------- 基线：常数预测 ----------
    base = float(np.median(y[tr]))
    for name, mask in (("常数中位数(基线)", te), ("常数中位数(基线)", sec)):
        p = np.full(mask.sum(), base)
        m = metrics(y[mask], p)
        results.append({"model": name, "eval_set": "test" if mask is te else "secondary",
                        **m})

    # ---------- ElasticNet ----------
    alpha, l1r, cv_score = tune_elasticnet(X[tr], y[tr])
    print(f"[C] ElasticNet alpha={alpha} l1_ratio={l1r} (inner CV median APE={cv_score * 100:.2f}%)")
    enet = make_elasticnet(alpha=alpha, l1_ratio=l1r).fit(X[tr], np.log(y[tr]))
    preds["ElasticNet"] = np.exp(enet.predict(X))

    # ---------- RandomForest ----------
    rf = make_rf().fit(X[tr], np.log(y[tr]))
    preds["RandomForest"] = np.exp(rf.predict(X))

    # ---------- GradientBoosting ----------
    gbm = make_gbm().fit(X[tr], np.log(y[tr]))
    preds["GradientBoosting"] = np.exp(gbm.predict(X))

    # ---------- 神经网络基线（MLP） ----------
    mlp = make_mlp().fit(X[tr], np.log(y[tr]))
    preds["MLP"] = np.exp(mlp.predict(X))

    # ---------- 1D CNN ----------
    cnn = DQCNN(n_scalar=X.shape[1], n_seed=1, epochs=600, lr=3e-3, patience=600, n_folds=5)
    cnn.fit(curves[tr], X[tr], y[tr])
    preds["CNN(1D)"] = cnn.predict(curves, X)

    # 物理下限/上限截断：循环寿命不可能 < 50 或 > 3000 圈，
    # 避免个别模型外推出接近 0 的值把相对误差放大到无意义的量级
    for name in preds:
        preds[name] = np.clip(preds[name], 50, 3000)

    for name, p in preds.items():
        for tag, mask in (("test", te), ("secondary", sec)):
            results.append({"model": name, "eval_set": tag, **metrics(y[mask], p[mask])})

    res = pd.DataFrame(results)
    res.to_csv(REP / "phaseC_metrics.csv", index=False, encoding="utf-8-sig")
    print(res.pivot(index="model", columns="eval_set", values="median_ape").round(2))

    # 预测明细
    detail = feat[["cell_id", "batch", "split", "cycle_life"]].copy()
    for name, p in preds.items():
        detail[f"pred_{name}"] = p
    detail.to_csv(REP / "phaseC_predictions.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 1：寿命分布 ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, 2400, 25)
    for tag, lab, color in (("train", "训练集 41", "#2b6cb0"),
                            ("test", "主测试 43", "#dd6b20"),
                            ("secondary_test", "二级测试 40", "#38a169")):
        ax.hist(y[split == tag], bins=bins, alpha=0.55, label=lab, color=color)
    ax.set_xlabel("循环寿命（圈，至 80% 容量）")
    ax.set_ylabel("电池数")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_01_life_dist.png")
    plt.close(fig)

    # ---------- 图 2：ΔQ(V) 老化轨迹 ----------
    life = cells.set_index("cell_id")["cycle_life"]
    order = np.argsort(life.values)
    pick = [ids[order[2]], ids[order[len(order) // 2]], ids[order[-3]]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, cid in zip(axes, pick):
        i = list(ids).index(cid)
        ax.plot(vgrid, cube[i, 0] * 1000, lw=1.4, label="第 2 圈")
        ax.plot(vgrid, cube[i, -1] * 1000, lw=1.4, label="第 100 圈")
        ax.set_title(f"{cid}（寿命 {int(life[cid])} 圈）", fontsize=10)
        ax.set_xlabel("端电压 / V")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ΔQ(V) / mAh")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_02_dq_aging.png")
    plt.close(fig)

    # ---------- 图 3：预测 vs 真实 ----------
    names = list(preds)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        p = preds[name]
        ax.scatter(y[te], p[te], s=26, alpha=0.85, label="主测试集", color="#dd6b20")
        ax.scatter(y[sec], p[sec], s=26, alpha=0.75, marker="^", label="二级测试集", color="#38a169")
        lim = [100, 2400]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xscale("log")
        ax.set_yscale("log")
        m = metrics(y[te], p[te])
        ax.set_title(f"{name}｜中位误差 {m['median_ape']:.1f}%", fontsize=10)
        ax.set_xlabel("真实寿命 / 圈")
        ax.set_ylabel("预测寿命 / 圈")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_04_pred_scatter.png")
    plt.close(fig)

    # ---------- 图 4：模型对比 ----------
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    plot_models = names + ["常数中位数(基线)"]
    sub = res[res["eval_set"] == "test"].drop_duplicates("model").set_index("model").loc[plot_models]
    xpos = np.arange(len(sub))
    ax.bar(xpos - 0.2, sub["median_ape"], width=0.38, label="中位相对误差", color="#2b6cb0")
    ax.bar(xpos + 0.2, sub["mean_ape"], width=0.38, label="平均相对误差", color="#90cdf4")
    ax.axhline(10, ls="--", lw=1, color="#c53030", label="Severson 目标 10%")
    ax.set_ylim(0, 60)
    for x, v in zip(xpos, sub["median_ape"]):
        if v > 60:
            ax.text(x, 56, f"{v:.0f}%", ha="center", fontsize=7, color="#2b6cb0")
    ax.set_xticks(xpos)
    ax.set_xticklabels([s.replace("(", "\n(") for s in sub.index], fontsize=8)
    ax.set_ylabel("相对误差 / %")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_03_model_compare.png")
    plt.close(fig)

    # ---------- 图 5：学习曲线 ----------
    rng = np.random.default_rng(SEED)
    sizes = [10, 15, 20, 25, 30, 35, 41]
    lc = []
    idx_tr = np.where(tr)[0]
    for n in sizes:
        for rep in range(8):
            pick_i = rng.choice(idx_tr, size=n, replace=False)
            for nm, mk in (("ElasticNet", lambda: make_elasticnet(alpha=alpha, l1_ratio=l1r)),
                           ("RandomForest", lambda: make_rf(n_estimators=200))):
                m = mk().fit(X[pick_i], np.log(y[pick_i]))
                p = np.exp(m.predict(X[te]))
                lc.append({"model": nm, "n_train": n,
                           "median_ape": metrics(y[te], p)["median_ape"]})
    lc_df = pd.DataFrame(lc)
    lc_df.to_csv(REP / "phaseC_learning_curve.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for nm, g in lc_df.groupby("model"):
        s = g.groupby("n_train")["median_ape"].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", ms=4, capsize=3, label=nm)
    ax.set_xlabel("训练电池数")
    ax.set_ylabel("主测试集中位相对误差 / %")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_05_learning_curve.png")
    plt.close(fig)

    # ---------- 图 6：RF 特征重要性 ----------
    imp = rf.named_steps["m"].feature_importances_
    top = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh([fcols[i] for i in top][::-1], imp[top][::-1], color="#4a5568")
    ax.set_xlabel("重要性")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_07_importance.png")
    plt.close(fig)

    # ---------- 消融：论文 6 特征 vs 全部特征 ----------
    sets = feature_sets(feat)
    p6 = sets["paper6"]
    Xp = feat[p6].values.astype(float)
    alpha6, l1r6, score6 = tune_elasticnet(Xp[tr], y[tr])
    print(f"[C][paper6] 特征数={len(p6)}，EN alpha={alpha6} l1_ratio={l1r6}，内部CV={score6 * 100:.2f}%")

    abl_rows = []
    for fname, Xf, a_use, l1_use in (("paper6", Xp, alpha6, l1r6), ("full", X, alpha, l1r)):
        for nm, mk in (("ElasticNet", lambda a=a_use, lr=l1_use: make_elasticnet(alpha=a, l1_ratio=lr)),
                       ("RandomForest", make_rf),
                       ("GradientBoosting", make_gbm)):
            m = mk().fit(Xf[tr], np.log(y[tr]))
            p = np.clip(np.exp(m.predict(Xf)), 50, 3000)
            for tag, mask in (("test", te), ("secondary", sec)):
                abl_rows.append({"feature_set": fname, "n_features": Xf.shape[1],
                                 "model": nm, "eval_set": tag, **metrics(y[mask], p[mask])})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(REP / "phaseC_ablation.csv", index=False, encoding="utf-8-sig")
    print("[C] 特征口径消融（中位相对误差 %）：")
    print(abl.pivot_table(index=["feature_set", "model"], columns="eval_set",
                          values="median_ape").round(2))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    models3 = ["ElasticNet", "RandomForest", "GradientBoosting"]
    for ax, tag, ttl in zip(axes, ("test", "secondary"),
                            ("主测试 43 颗", "二级测试 40 颗（batch3）")):
        sub = abl[abl["eval_set"] == tag]
        xpos = np.arange(len(models3))
        for off, fs, color, lab in ((-0.19, "paper6", "#2b6cb0", "论文 6 特征"),
                                    (0.19, "full", "#90cdf4", f"全部 {len(fcols)} 特征")):
            vals = []
            for mdl in models3:
                r = sub[(sub.model == mdl) & (sub.feature_set == fs)]["median_ape"]
                vals.append(float(r.iloc[0]) if len(r) else np.nan)
            ax.bar(xpos + off, vals, width=0.36, color=color, label=lab)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["弹性网", "随机森林", "GBM"], fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("中位相对误差 / %")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_08_ablation.png")
    plt.close(fig)

    # ---------- 补充口径：全量 124 颗 5 折交叉验证 ----------
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for k, (tri, vai) in enumerate(kf5.split(X)):
        for nm, mk in (("ElasticNet", lambda: make_elasticnet(alpha=alpha, l1_ratio=l1r)),
                       ("RandomForest", make_rf),
                       ("GradientBoosting", make_gbm),
                       ("MLP", make_mlp)):
            m = mk().fit(X[tri], np.log(y[tri]))
            cv_rows.append({"fold": k, "model": nm,
                            **metrics(y[vai], np.exp(m.predict(X[vai])))})
        c = DQCNN(n_scalar=X.shape[1], n_seed=1, epochs=500, lr=3e-3, patience=500, n_folds=4)
        c.fit(curves[tri], X[tri], y[tri])
        cv_rows.append({"fold": k, "model": "CNN(1D)",
                        **metrics(y[vai], c.predict(curves[vai], X[vai]))})
        print(f"  [CV] fold {k} 完成", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(REP / "phaseC_cv.csv", index=False, encoding="utf-8-sig")
    cv_sum = cv.groupby("model")["median_ape"].agg(["mean", "std"]).round(2)
    print("[C] 全量 5 折 CV（中位相对误差 %）：")
    print(cv_sum)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    order_cv = cv_sum["mean"].sort_values().index
    ax.barh(order_cv, cv_sum.loc[order_cv, "mean"], xerr=cv_sum.loc[order_cv, "std"],
            color="#2b6cb0", height=0.5, capsize=3)
    ax.set_xlabel("5 折交叉验证中位相对误差 / %（均值±标准差）")
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c_06_cv.png")
    plt.close(fig)

    # ---------- 汇总 JSON ----------
    summary = {
        "n_features": len(fcols),
        "elasticnet_alpha": alpha,
        "elasticnet_l1_ratio": l1r,
        "inner_cv_median_ape": cv_score * 100,
        "paper6_features": p6,
        "paper6_alpha": alpha6,
        "paper6_l1_ratio": l1r6,
        "paper6_inner_cv_median_ape": score6 * 100,
        "ablation": abl.to_dict(orient="records"),
        "metrics": res.to_dict(orient="records"),
        "cv5": cv_sum.reset_index().to_dict(orient="records"),
        "top_features": [fcols[i] for i in top[:10]],
    }
    (REP / "phaseC_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[C] 完成，图与表已写入 figures/ reports/")


if __name__ == "__main__":
    main()
