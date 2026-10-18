# -*- coding: utf-8 -*-
"""现有结果的统计显著性分析：bootstrap 置信区间 + Wilcoxon 配对检验
========================================================================

背景
----
两个深化方案（智诚 §5.2 / 严谨 §2.1）都要求「所有结论必须带 bootstrap CI 或
多种子重复，禁止只报单点」「改进是否显著用 Wilcoxon 符号秩检验判定」。
本仓库此前的报告只报中位APE单点数字，本脚本把这块补上：

  输入（全部是已有产物，不需要原始数据）：
    reports/phaseC_predictions.csv         5 模型 × 124 颗逐颗预测（官方划分）
    reports/en_l1ratio_predictions.csv     EN 三配置 × 124 颗逐颗预测（受控实验）

  输出：
    reports/phaseC_significance.csv        每个（模型×测试集）的中位 APE +
                                          95% bootstrap CI + 与最佳模型的
                                          Wilcoxon 配对检验 p 值 + 判定标签

方法口径（与方案对齐，写死在代码里）
----------------------------------
- bootstrap：对测试集的逐颗百分比误差有放回重采样 1000 次（种子固定），
  取中位 APE 的 2.5/97.5 分位作为 95% CI。n=43/40，小样本，禁止报裸数字。
- Wilcoxon：配对的是**同一颗 cell 的绝对百分比误差**（paired by cell），
  双侧检验。判定标签遵循智诚 §5.2：p<0.05 且中位改进 ≥1 个百分点 → 「显著」；
  p<0.05 但改进 <1 pp → 「边际」；否则「不显著」。对「哪个模型更好」的
  比较， additionally 报告方向。
- 所有百分比误差在原始圈数空间计算。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRED_MAIN = ROOT / "reports" / "phaseC_predictions.csv"
PRED_EN = ROOT / "reports" / "en_l1ratio_predictions.csv"
OUT = ROOT / "reports" / "phaseC_significance.csv"

N_BOOT = 1000
SEED = 42
MODELS = ["ElasticNet", "RandomForest", "GradientBoosting", "MLP", "CNN(1D)"]
EN_CONFIGS = ["OLD", "NEW", "GUIDE"]  # en_l1ratio_predictions.csv 的列名前缀 pred_*
EN_LABELS = {
    "OLD": "OLD_fixed_l1_0.5（旧配置）",
    "NEW": "NEW_grid_search（新配置）",
    "GUIDE": "GUIDE_replica（guide配方）",
}


def ape(y_true, y_pred):
    return np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.asarray(y_true) * 100


def _fmt_p(p):
    if p == "":
        return "-"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def boot_ci(err, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(err)
    meds = np.empty(n_boot)
    for i in range(n_boot):
        meds[i] = np.median(rng.choice(err, size=n, replace=True))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def wilcoxon_label(err_a, err_b):
    """err_a 相对 err_b 的配对检验。返回 (p, 标签, 差距pp, 方向)。

    标签遵循智诚 §5.2：p<0.05 且 |中位差距| ≥1 pp → 「显著」；
    p<0.05 但 <1 pp → 「边际」；否则「不显著」。方向：a 更差/更优/持平。
    """
    stat, p = stats.wilcoxon(err_a, err_b, alternative="two-sided", zero_method="wilcox")
    m_a, m_b = float(np.median(err_a)), float(np.median(err_b))
    delta = m_b - m_a  # >0 表示 a 更优（误差更低）delta pp
    adelta = abs(delta)
    if p < 0.05 and adelta >= 1.0:
        label = "显著"
    elif p < 0.05:
        label = "边际"
    else:
        label = "不显著"
    direction = "更优" if delta > 1e-9 else ("更差" if delta < -1e-9 else "持平")
    return float(p), label, delta, direction


def load_main():
    rows = list(csv.DictReader(open(PRED_MAIN, encoding="utf-8-sig")))
    data = {}
    for r in rows:
        data.setdefault(r["split"], []).append(r)
    return {k: v for k, v in data.items()}


def main():
    out_rows = []
    data = load_main()

    for split, rows in data.items():
        y = np.array([float(r["cycle_life"]) for r in rows])
        errs = {m: ape(y, [float(r[f"pred_{m}"]) for r in rows]) for m in MODELS}
        meds = {m: float(np.median(e)) for m, e in errs.items()}
        best = min(meds, key=meds.get)
        scope = "（样本内·过拟合诊断）" if split == "train" else ""
        print(f"\n===== {split}（n={len(rows)}）最佳模型：{best} {scope}=====")
        for m in MODELS:
            lo, hi = boot_ci(errs[m])
            if m == best:
                p, label, delta, direction = "", "基准", 0.0, "—"
            else:
                p, label, delta, direction = wilcoxon_label(errs[m], errs[best])
            print(
                f"  {m:<18} 中位 {meds[m]:6.2f}%  CI[{lo:5.2f},{hi:5.2f}]  "
                f"vs {best}: {direction}{delta:+5.2f}pp p={_fmt_p(p)} {label}"
            )
            out_rows.append({
                "analysis": "model_compare",
                "eval_set": split,
                "item": m,
                "n": len(rows),
                "median_ape": round(meds[m], 2),
                "ci95_lo": round(lo, 2),
                "ci95_hi": round(hi, 2),
                "vs_best": best,
                "delta_pp": round(delta, 2),
                "wilcoxon_p": round(p, 6) if p != "" else "",
                "label": label,
                "direction": direction,
            })

    # EN 配置对比（来自受控实验的逐颗预测）
    if PRED_EN.exists():
        rows = list(csv.DictReader(open(PRED_EN, encoding="utf-8-sig")))
        for split in ("test", "secondary_test"):
            sub = [r for r in rows if r["split"] == split]
            if not sub:
                continue
            y = np.array([float(r["cycle_life"]) for r in sub])
            errs = {c: ape(y, [float(r[f"pred_{c}"]) for r in sub]) for c in EN_CONFIGS}
            meds = {c: float(np.median(e)) for c, e in errs.items()}
            best = min(meds, key=meds.get)
            print(f"\n===== EN 配置对比 {split}（n={len(sub)}）=====")
            for c in EN_CONFIGS:
                lo, hi = boot_ci(errs[c])
                if c == best:
                    p, label, delta, direction = "", "基准", 0.0, "—"
                else:
                    p, label, delta, direction = wilcoxon_label(errs[c], errs[best])
                print(
                    f"  {EN_LABELS[c]:<26} 中位 {meds[c]:6.2f}%  CI[{lo:5.2f},{hi:5.2f}]  "
                    f"vs {EN_LABELS[best].split('（')[0]}: {direction}{delta:+5.2f}pp p={_fmt_p(p)} {label}"
                )
                out_rows.append({
                    "analysis": "en_config",
                    "eval_set": split,
                    "item": EN_LABELS[c],
                    "n": len(sub),
                    "median_ape": round(meds[c], 2),
                    "ci95_lo": round(lo, 2),
                    "ci95_hi": round(hi, 2),
                    "vs_best": EN_LABELS[best],
                    "delta_pp": round(delta, 2),
                    "wilcoxon_p": round(p, 6) if p != "" else "",
                    "label": label,
                    "direction": direction,
                })

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[输出] {OUT.relative_to(ROOT)}（{len(out_rows)} 行）")


if __name__ == "__main__":
    main()
