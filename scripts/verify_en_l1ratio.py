# -*- coding: utf-8 -*-
"""ElasticNet l1_ratio 消融验证：固定 0.5 vs 网格搜索 vs 姊妹仓库配方
====================================================================

背景
----
ECM-RUL 与 battery-lifetime-guide 是两个独立复现（同一 MIT 124 口径、同一篇
Severson 2019），随机森林完全对齐（8.12% vs 8.2%），但弹性网差 3.4 个百分点
（本仓 8.33% vs guide 5.9%）。根因已定位：本仓早期版本固定 l1_ratio=0.5，
而 guide 用 ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0]) 让混合比参与搜索。
代码侧修复已随 619bbe8 提交（phase_c_rul.py 的 tune_elasticnet 改为双参数
内层 CV），但 MIT 原始数据（8.3 GB）不在本地，无法整体重跑 phase_c_rul.py。

本脚本用**最小的受控实验**单独验证这一个变量：
  数据 = 本仓 reports/phaseC_predictions.csv 的 124 颗名单（split 标注 +
         合并后续跑电池后的正确标签，5 颗续跑电池的标签差异已核对）
        × 姊妹仓库缓存的论文六特征快照（mit_features.csv，138 行过滤前口径，
         按名单 join 后恰好 124 颗；特征是各电池前 100 圈的计算，不受
         续跑合并影响）
  三个模型配置（其余全部固定：StandardScaler、训练 41 → 测试 43 官方混合划分）：
    OLD  固定 l1_ratio=0.5，仅内层 5 折 CV 选 alpha（复刻 619bbe8 之前的行为）
    NEW  内层 5 折 CV 同时搜 alpha × l1_ratio（复刻当前 phase_c_rul.py）
    GUIDE 复刻 guide 配方：ElasticNetCV(l1_ratio 网格, cv=KFold(4,shuffle,42))
  对拍：从 guide 的逐颗误差文件反算其中位误差，并逐颗比对预测值。

口径声明（诚实边界）
------------------
- 本脚本验证的是「超参搜索空间」这一个变量，不替代整条管线重跑；
  MIT 原始数据就位后仍需以 phase_c_rul.py 全量重跑为准，本脚本结果作为
  桥梁证据记录于 docs/results.md。
- 特征来自姊妹仓库管线（特征定义与 batch 口径已经在 RF 对齐中得到交叉验证，
  但严格说不等同于本仓 build_features 的输出）。
- 百分比误差一律在原始圈数空间计算（训练在 log 空间，预测后 exp/10^ 还原）。

输出
----
reports/en_l1ratio_verification.csv   三配置 × 两测试集的指标汇总
reports/en_l1ratio_predictions.csv    逐颗预测（三配置 + guide 对拍）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rul.models import SEED, make_elasticnet  # noqa: E402

# 数据路径（可用环境变量覆盖）
ROSTER = ROOT / "reports" / "phaseC_predictions.csv"
FEATURES_CSV = (
    Path(__file__).resolve().parents[1].parent
    / "battery-lifetime-guide"
    / "results"
    / "mit_features.csv"
)
GUIDE_DETAIL = (
    Path(__file__).resolve().parents[1].parent
    / "battery-lifetime-guide"
    / "results"
    / "阶段43f_论文主测试逐颗误差.csv"
)
OUT_DIR = ROOT / "reports"

PAPER6 = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def load_matrix():
    """名单（标签+split）× 六特征快照 → 124×6 矩阵。"""
    roster = {}
    with open(ROSTER, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = ("batch" + str(int(r["batch"])), r["cell_id"][3:])
            roster[key] = {
                "cell_id": r["cell_id"],
                "split": r["split"],
                "cycle_life": float(r["cycle_life"]),
                "old_en_pred": float(r["pred_ElasticNet"]),
            }
    feats = {}
    with open(FEATURES_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            feats[(r["batch"], r["cell_idx"])] = r
    matched = sorted(set(roster) & set(feats))
    if len(matched) != 124:
        raise SystemExit(f"join 后得到 {len(matched)} 颗，预期 124——数据口径可能变了")
    keys, y, split, old_pred, X = [], [], [], [], []
    for k in matched:
        keys.append(k)
        y.append(roster[k]["cycle_life"])
        split.append(roster[k]["split"])
        old_pred.append(roster[k]["old_en_pred"])
        X.append([float(feats[k][c]) for c in PAPER6])
    return keys, np.array(X), np.array(y), np.array(split), np.array(old_pred)


def metrics(y_true, y_pred):
    err = np.abs(y_pred - y_true) / y_true * 100
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "median": float(np.median(err)),
        "mean": float(err.mean()),
        "p90": float(np.percentile(err, 90)),
        "mae_cycles": float(np.abs(y_pred - y_true).mean()),
        "r2": 1 - ss_res / ss_tot,
    }


def fit_old(Xtr, ytr):
    """复刻 619bbe8 之前：仅搜 alpha，l1_ratio 固定 0.5。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best, best_score = None, np.inf
    for a in ALPHAS:
        errs = []
        for tr, va in kf.split(Xtr):
            m = make_elasticnet(alpha=a)
            m.fit(Xtr[tr], np.log(ytr[tr]))
            errs.append(
                float(np.median(np.abs(np.exp(m.predict(Xtr[va])) - ytr[va]) / ytr[va]))
            )
        score = float(np.mean(errs))
        if score < best_score:
            best, best_score = a, score
    m = make_elasticnet(alpha=best)
    m.fit(Xtr, np.log(ytr))
    return m, {"alpha": best, "l1_ratio": 0.5}


def fit_new(Xtr, ytr):
    """复刻当前 phase_c_rul.py：alpha × l1_ratio 双参数内层 CV。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best, best_score = None, np.inf
    for a in ALPHAS:
        for lr in (0.1, 0.5, 0.9, 1.0):
            errs = []
            for tr, va in kf.split(Xtr):
                m = make_elasticnet(alpha=a, l1_ratio=lr)
                m.fit(Xtr[tr], np.log(ytr[tr]))
                errs.append(
                    float(np.median(np.abs(np.exp(m.predict(Xtr[va])) - ytr[va]) / ytr[va]))
                )
            score = float(np.mean(errs))
            if score < best_score:
                best, best_score = (a, lr), score
    m = make_elasticnet(alpha=best[0], l1_ratio=best[1])
    m.fit(Xtr, np.log(ytr))
    return m, {"alpha": best[0], "l1_ratio": best[1]}


def fit_guide(Xtr, ytr):
    """复刻 guide step4_3f：ElasticNetCV，log10 目标，4 折。"""
    m = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            cv=KFold(n_splits=4, shuffle=True, random_state=42),
            random_state=42,
            max_iter=100000,
        ),
    )
    m.fit(Xtr, np.log10(ytr))
    en = m.named_steps["elasticnetcv"]
    return m, {"alpha": float(en.alpha_), "l1_ratio": float(en.l1_ratio_)}


def main():
    keys, X, y, split, old_pred = load_matrix()
    tr = split == "train"
    te = split == "test"
    se = split == "secondary_test"
    print(f"[数据] 124 颗：train {tr.sum()} / test {te.sum()} / secondary {se.sum()}")
    print(f"[数据] 训练集内层 CV 选参，测试集只评估\n")

    configs = {
        "OLD_fixed_l1_0.5": (fit_old, np.exp),
        "NEW_grid_search": (fit_new, np.exp),
        "GUIDE_replica": (fit_guide, lambda p: 10.0**p),
    }

    rows, details = [], {}
    for name, (fit_fn, inv) in configs.items():
        model, hp = fit_fn(X[tr], y[tr])
        print(f"[{name}] 选中 alpha={hp['alpha']:.4g} l1_ratio={hp['l1_ratio']:.2f}")
        pred_all = inv(model.predict(X))  # 全量预测，仅按 split 评估
        p_te, p_se = pred_all[te], pred_all[se]
        m_te, m_se = metrics(y[te], p_te), metrics(y[se], p_se)
        print(
            f"    主测试 43：中位 {m_te['median']:.2f}% | 均值 {m_te['mean']:.2f}% | "
            f"MAE {m_te['mae_cycles']:.0f} 圈 | R² {m_te['r2']:.3f}"
        )
        print(
            f"    二级测试 40：中位 {m_se['median']:.2f}% | 均值 {m_se['mean']:.2f}% | "
            f"MAE {m_se['mae_cycles']:.0f} 圈 | R² {m_se['r2']:.3f}\n"
        )
        rows.append(
            {
                "config": name,
                "alpha": round(hp["alpha"], 6),
                "l1_ratio": hp["l1_ratio"],
                "test_median_pct": round(m_te["median"], 2),
                "test_mean_pct": round(m_te["mean"], 2),
                "test_p90_pct": round(m_te["p90"], 2),
                "test_mae_cycles": round(m_te["mae_cycles"], 1),
                "test_r2": round(m_te["r2"], 3),
                "secondary_median_pct": round(m_se["median"], 2),
                "secondary_mean_pct": round(m_se["mean"], 2),
                "secondary_mae_cycles": round(m_se["mae_cycles"], 1),
                "secondary_r2": round(m_se["r2"], 3),
            }
        )
        details[name] = pred_all  # 全量 124 预测

    # 参考行：本仓旧预测（已入 reports 的 8.33%/6.31% 来源）
    m_old_te = metrics(y[te], old_pred[te])
    m_old_se = metrics(y[se], old_pred[se])
    print(
        f"[旧仓库存档值] 主测试中位 {m_old_te['median']:.2f}% | 二级测试中位 {m_old_se['median']:.2f}%"
        "（与 results.md 记录的 8.33%/6.31% 互为校验）"
    )
    rows.insert(
        0,
        {
            "config": "ARCHIVED_old_forecast",
            "alpha": "",
            "l1_ratio": 0.5,
            "test_median_pct": round(m_old_te["median"], 2),
            "test_mean_pct": round(m_old_te["mean"], 2),
            "test_p90_pct": round(m_old_te["p90"], 2),
            "test_mae_cycles": round(m_old_te["mae_cycles"], 1),
            "test_r2": round(m_old_te["r2"], 3),
            "secondary_median_pct": round(m_old_se["median"], 2),
            "secondary_mean_pct": round(m_old_se["mean"], 2),
            "secondary_mae_cycles": round(m_old_se["mae_cycles"], 1),
            "secondary_r2": round(m_old_se["r2"], 3),
        },
    )

    # guide 逐颗对拍
    if GUIDE_DETAIL.exists():
        gm = {}
        with open(GUIDE_DETAIL, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                gm[("batch" + r["batch"] if not r["batch"].startswith("batch") else r["batch"]), r["cell_idx"]] = r
        te_keys = [k for k, s in zip(keys, split) if s == "test"]
        g_errs, g_preds, g_matched_pred = [], [], []
        guide_te = details["GUIDE_replica"][te]
        for j, k in enumerate(te_keys):
            r = gm.get(k)
            if r is None:
                continue
            g_errs.append(float(r["误差%"]))
            g_preds.append(float(r["预测"]))
            g_matched_pred.append(guide_te[j])
        if g_errs:
            print(
                f"\n[对拍] guide 逐颗文件 {len(g_errs)} 颗：其中位误差 "
                f"{np.median(g_errs):.2f}%（README 声称 5.9%）"
            )
            if g_preds:
                d = np.abs(np.array(g_preds) - np.array(g_matched_pred))
                rel = d / np.maximum(np.array(g_preds), 1e-9)
                print(
                    f"[对拍] 逐颗预测差异：中位相对差 {np.median(rel) * 100:.3f}%，"
                    f"最大 {rel.max() * 100:.2f}%（数量级一致即证明数据与协议对齐）"
                )
    else:
        print("\n[对拍] 未找到 guide 逐颗文件，跳过")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "en_l1ratio_verification.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_DIR / "en_l1ratio_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "batch", "cell_idx", "split", "cycle_life",
                    "pred_OLD", "pred_NEW", "pred_GUIDE", "pred_old_archive"])
        for i, (k, yy, s, op) in enumerate(zip(keys, y, split, old_pred)):
            w.writerow([f"b{k[0][-1]}c{k[1]}", k[0], k[1], s, yy,
                        round(float(details["OLD_fixed_l1_0.5"][i]), 2),
                        round(float(details["NEW_grid_search"][i]), 2),
                        round(float(details["GUIDE_replica"][i]), 2),
                        op])
    print(f"\n[输出] reports/en_l1ratio_verification.csv")
    print(f"[输出] reports/en_l1ratio_predictions.csv")


if __name__ == "__main__":
    main()
