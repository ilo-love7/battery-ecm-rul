# -*- coding: utf-8 -*-
"""LOBO 三折留一批次交叉验证 + bootstrap CI + 特征分布 KS 检验
=================================================================

对应深化方案（严谨 §2.1 / §4）中「审稿视角下含金量最高、工作量最小」的一块。
guide 已有的轮换表（results/mit_rotation.csv）只有 3 个单点中位数、单个模型、
无置信区间；本脚本把它升级为规范协议：

  - 三折纯批次留一（train 两批 → test 剩一批），两个模型版本
    （旧配置 l1_ratio 固定 0.5 / 新配置 alpha×l1_ratio 网格）都跑；
  - 每折**独立**内层 5 折 CV 调参（标准化在 Pipeline 内，只吃折内训练数据，
    无测试折泄漏——这是本实验最常见的泄漏点，方案坑清单第 1 条）；
  - 每折报 中位/均值/P90/最差 cell 的 APE + 中位 APE 的 1000 次 bootstrap CI；
  - 每折对 6 个论文特征做 KS 检验（测试折 vs 折内训练池），定位「哪折难、
    难在哪个特征上」（严谨 §4 Step 2）。

数据口径（与 scripts/verify_en_l1ratio.py 相同，诚实边界见其 docstring）：
  名单/标签/split 来自 reports/phaseC_predictions.csv（合并后续跑电池后的
  正确标签）；六特征来自 battery-lifetime-guide 的 mit_features.csv 快照
  （138 行过滤前口径，按 124 名单 join）。60 维特征未缓存，本实验只覆盖
  paper6 口径；60 维的 LOBO 需要原始数据重跑。

用法：python scripts/lobo_analysis.py
输出：reports/lobo_folds.csv（三折 × 两模型 × 全指标 + KS 检验摘要）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rul.models import SEED, make_elasticnet  # noqa: E402

ROSTER = ROOT / "reports" / "phaseC_predictions.csv"
FEATURES_CSV = (
    ROOT.parent / "battery-lifetime-guide" / "results" / "mit_features.csv"
)
OUT = ROOT / "reports" / "lobo_folds.csv"

PAPER6 = ["log_var", "log_min", "log_mean", "slope", "intercept", "discrepancy"]
ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_BOOT = 1000

CONFIGS = {
    "OLD_fixed_l1_0.5": {"l1_ratios": (0.5,)},
    "NEW_grid_search": {"l1_ratios": (0.1, 0.5, 0.9, 1.0)},
}


def load_matrix():
    roster = {}
    with open(ROSTER, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            b = "batch" + str(int(r["batch"]))
            roster[(b, r["cell_id"][3:])] = {
                "cell_id": r["cell_id"],
                "cycle_life": float(r["cycle_life"]),
            }
    feats = {}
    with open(FEATURES_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            feats[(r["batch"], r["cell_idx"])] = r
    keys = sorted(set(roster) & set(feats))
    if len(keys) != 124:
        raise SystemExit(f"join 后 {len(keys)} 颗，预期 124")
    X, y, batches = [], [], []
    for k in keys:
        X.append([float(feats[k][c]) for c in PAPER6])
        y.append(roster[k]["cycle_life"])
        batches.append(k[0])
    return keys, np.array(X), np.array(y), np.array(batches)


def boot_ci(err, seed=SEED):
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(err, size=len(err), replace=True)) for _ in range(N_BOOT)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fit_tuned(Xtr, ytr, l1_ratios):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best, best_score = None, np.inf
    for a in ALPHAS:
        for lr in l1_ratios:
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
    return m, best


def main():
    keys, X, y, batches = load_matrix()
    print(f"[数据] 124 颗（b1 {np.sum(batches == 'batch1')} / "
          f"b2 {np.sum(batches == 'batch2')} / b3 {np.sum(batches == 'batch3')}）")
    rows = []

    for test_b in ("batch1", "batch2", "batch3"):
        tr = batches != test_b
        te = batches == test_b
        train_batches = "+".join(sorted(set(batches[tr])))
        print(f"\n===== 折：train {train_batches} → test {test_b}（n={te.sum()}）=====")

        # 特征分布漂移：测试折 vs 折内训练池（严谨 §4 Step 2）
        ks_worst = []
        for j, cname in enumerate(PAPER6):
            ks_stat, ks_p = stats.ks_2samp(X[te][:, j], X[tr][:, j])
            ks_worst.append((ks_stat, ks_p, cname))
        ks_worst.sort(reverse=True)
        ks_top = ks_worst[0]
        print(f"  [KS] 漂移最大特征：{ks_top[2]}（D={ks_top[0]:.3f}, p={ks_top[1]:.2e}）；"
              f"全部 6 特征 p<0.05 的有 {sum(1 for s, p, _ in ks_worst if p < 0.05)} 个")

        for name, cfg in CONFIGS.items():
            model, (alpha, l1r) = fit_tuned(X[tr], y[tr], cfg["l1_ratios"])
            pred = np.exp(model.predict(X[te]))
            err = np.abs(pred - y[te]) / y[te] * 100
            lo, hi = boot_ci(err)
            row = {
                "test_batch": test_b,
                "train_batches": train_batches,
                "n_test": int(te.sum()),
                "config": name,
                "alpha": alpha,
                "l1_ratio": l1r,
                "median_ape": round(float(np.median(err)), 2),
                "mean_ape": round(float(err.mean()), 2),
                "p90_ape": round(float(np.percentile(err, 90)), 2),
                "max_ape": round(float(err.max()), 2),
                "ci95_lo": round(lo, 2),
                "ci95_hi": round(hi, 2),
                "ks_top_feature": ks_top[2],
                "ks_top_D": round(float(ks_top[0]), 3),
                "n_features_p05": sum(1 for s, p, _ in ks_worst if p < 0.05),
            }
            rows.append(row)
            print(
                f"  {name:<18} α={alpha:<6g} l1={l1r:.1f} | 中位 {row['median_ape']:5.2f}% "
                f"CI[{row['ci95_lo']:5.2f},{row['ci95_hi']:5.2f}] | 均值 {row['mean_ape']:5.2f}% "
                f"P90 {row['p90_ape']:6.2f}% | 最差 {row['max_ape']:6.2f}%"
            )

    # 与 guide 轮换表对拍（其单点数字：19.3 / 34.3 / 8.4，模型未标注，应为 EN 类）
    print("\n[对拍] guide results/mit_rotation.csv 单点中位（模型未标注）："
          "test b1 19.3% / b2 34.3% / b3 8.4%")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[输出] {OUT.relative_to(ROOT)}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
