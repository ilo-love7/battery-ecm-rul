"""生成自包含 HTML 数据看板（图片 base64 内嵌，离线可看，无外部依赖）。

用法：python scripts/build_dashboard.py
产物：<项目根>/交付物/数据看板.html
"""
from __future__ import annotations

import base64
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent.parent / "交付物"
OUT.mkdir(parents=True, exist_ok=True)

FIG = ROOT / "figures"
REP = ROOT / "reports"


def img(name: str, caption: str = "", w: str = "100%") -> str:
    p = FIG / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{w}">{cap}</figure>'


def table(df: pd.DataFrame, fmt: dict | None = None, cls: str = "") -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if c in fmt and isinstance(v, (int, float)):
                tds.append(f"<td>{v:.{fmt[c]}f}</td>")
            elif isinstance(v, float):
                tds.append(f"<td>{v:.2f}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def main() -> int:
    metrics = pd.read_csv(REP / "phaseC_metrics.csv")
    piv = metrics.pivot(index="model", columns="eval_set", values="median_ape")
    piv = piv.reindex(columns=["test", "secondary"]).reset_index()
    piv.columns = ["模型", "主测试集 43 颗（中位误差 %）", "二级测试集 40 颗（中位误差 %）"]
    order = ["GradientBoosting", "ElasticNet", "RandomForest", "MLP", "CNN(1D)", "常数中位数(基线)"]
    piv["_o"] = piv["模型"].map({m: i for i, m in enumerate(order)}).fillna(99)
    piv = piv.sort_values("_o").drop(columns="_o")

    cv = pd.read_csv(REP / "phaseC_cv.csv")
    cvsum = cv.groupby("model")["median_ape"].agg(["mean", "std"]).reset_index()
    cvsum.columns = ["模型", "5 折 CV 中位误差均值 %", "标准差"]
    cvsum["_o"] = cvsum["模型"].map({m: i for i, m in enumerate(order)}).fillna(99)
    cvsum = cvsum.sort_values("_o").drop(columns="_o")

    ecm = pd.read_csv(REP / "phaseA2_ecm_results.csv")
    ecm = ecm[["cell", "R0_step_mohm", "Rdc50_mohm", "Rpol50_mohm",
               "rmse_test_raw_pct", "rmse_test_cal_pct"]].copy()
    ecm.columns = ["电池", "R0（mΩ）", "R_dc(50%)（mΩ）", "R_pol(50%)（mΩ）",
                   "复测 RMSE 原始（%）", "复测 RMSE 标定（%）"]

    res = json.loads((REP / "phaseC_results.json").read_text(encoding="utf-8"))
    best = piv.iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>锂电池 ECM 建模 + RUL 寿命预测 · 数据看板</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1a202c;--sub:#5a6472;--line:#e2e6ea;--blue:#1f5fa9;--teal:#0f766e;--amber:#b45309}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",-apple-system,"Segoe UI",sans-serif;line-height:1.65}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 72px}}
header{{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:30px}}
h1{{font-size:26px;margin:0 0 8px;font-weight:600;letter-spacing:.5px}}
.sub{{color:var(--sub);font-size:13.5px}}
.badges{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}
.badge{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px;min-width:150px}}
.badge b{{display:block;font-size:20px;color:var(--blue);font-weight:600}}
.badge span{{font-size:12px;color:var(--sub)}}
section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px 26px;margin-bottom:22px}}
h2{{font-size:16.5px;margin:0 0 6px;font-weight:600}}
h2 .n{{color:var(--blue);margin-right:8px;font-variant-numeric:tabular-nums}}
.lead{{color:var(--sub);font-size:13px;margin:0 0 16px}}
figure{{margin:16px 0 6px;text-align:center}}
figure img{{border:1px solid var(--line);border-radius:8px}}
figcaption{{font-size:12px;color:var(--sub);margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{color:var(--sub);font-weight:500;font-size:12.5px;background:#fafbfc}}
tbody tr:hover{{background:#f8fafc}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.note{{background:#f0f7ff;border-left:3px solid var(--blue);padding:12px 16px;border-radius:0 8px 8px 0;font-size:13px;color:#20364f;margin-top:14px}}
.warn{{background:#fff8ec;border-left:3px solid var(--amber);padding:12px 16px;border-radius:0 8px 8px 0;font-size:13px;color:#5a3d12;margin-top:14px}}
code{{background:#eef1f4;padding:1px 6px;border-radius:4px;font-size:12px;font-family:Consolas,monospace}}
footer{{color:var(--sub);font-size:12px;text-align:center;margin-top:30px}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<header>
<h1>锂电池等效电路建模 + RUL 寿命预测</h1>
<div class="sub">NASA PCoE / MIT-Stanford（Severson 2019）双数据集 · 阶段 A~F 完整交付 · 生成于 {date.today().isoformat()}</div>
<div class="badges">
<div class="badge"><b>124</b><span>MIT 官方口径电池（41/43/40）</span></div>
<div class="badge"><b>{best['主测试集 43 颗（中位误差 %）']:.1f}%</b><span>最优模型主测试中位误差</span></div>
<div class="badge"><b>1.61%</b><span>ECM 复测 RMSE 中位数</span></div>
<div class="badge"><b>34</b><span>NASA 电池 · 2794 次放电</span></div>
</div>
</header>

<section>
<h2><span class="n">01</span>数据与任务</h2>
<p class="lead">两套公开数据、统一预处理成「逐循环标量表 + ΔQ(V) 曲线立方体」，支撑 ECM 参数辨识与早期寿命预测两类任务。</p>
<div class="grid2">
{table(pd.DataFrame([
    {"数据集": "MIT-Stanford", "电池数": "124（官方白名单）", "循环数": "100,501", "用途": "RUL 主基准"},
    {"数据集": "NASA PCoE", "电池数": "34", "循环数": "7,565（放电 2,794）", "用途": "ECM 辨识 + 老化分析"},
]))}
{img("fig_c_01_life_dist.png", "图 1　循环寿命分布：训练集 41 / 主测试 43 / 二级测试 40，寿命跨度 148–2237 圈")}
</div>
</section>

<section>
<h2><span class="n">02</span>ECM 等效电路参数辨识</h2>
<p class="lead">OCV-SOC 用 20 圈充放电平均 + PCHIP 平滑；R0 用放电电流阶跃压降；R_dc(SOC) 由恒流区逐点求取。四颗 NASA 电池复测 RMSE 中位数 1.61%，达到「≤2% 量程」目标值。</p>
{img("fig_a2_ocv_soc.png", "图 2　OCV-SOC 曲线（四颗电池）", "62%")}
{table(ecm, fmt={"R0（mΩ）": 2, "R_dc(50%)（mΩ）": 2, "R_pol(50%)（mΩ）": 2, "复测 RMSE 原始（%）": 2, "复测 RMSE 标定（%）": 2})}
<div class="note">B0005 由初版 2.25% 降至 1.72%，B0006 0.96%、B0018 1.49% 达标；B0007 为 3.29%，源于其前 20 圈存在明显容量再生，OCV 平均被老化状态污染——已在报告中如实标注。</div>
{img("fig_a2_r_aging.png", "图 3　内阻随 SOH 的演化：SOH 从 100% 降到 70%，R0 上升约 5%，R_dc(50%) 上升约 33%")}
</section>

<section>
<h2><span class="n">03</span>RUL 早期寿命预测</h2>
<p class="lead">仅用前 100 圈数据，预测电池跑到 80% 容量的循环数。划分完全复刻 Severson 2019：训练 41 颗、主测试 43 颗、二级测试（batch3）40 颗。</p>
{img("fig_c_04_pred_scatter.png", "图 4　预测 vs 真实（对数坐标）：ElasticNet / GBM / MLP 贴合对角线，CNN 明显向均值收缩")}
{table(piv, fmt={"主测试集 43 颗（中位误差 %）": 2, "二级测试集 40 颗（中位误差 %）": 2})}
<div class="note">GradientBoosting 主测试 6.20%（R²=0.93）最优，但跨批次升到 16.37%；RandomForest 跨批次最稳（11.89%）。ElasticNet 主测试 7.77%，换到 batch3 退化到 28.64%——60 维特征里编码了充电协议，换批次就失效。</div>
{img("fig_c_03_model_compare.png", "图 5　各模型中位/平均相对误差对比（红色虚线为论文 10% 目标）")}
<div class="grid2">
{img("fig_c_05_learning_curve.png", "图 6　学习曲线：训练电池数从 10 增到 41，误差持续下降且未收敛")}
{img("fig_c_06_cv.png", "图 7　全量 124 颗 5 折交叉验证：RF 7.37% 最优")}
</div>
{table(cvsum, fmt={"5 折 CV 中位误差均值 %": 2, "标准差": 2})}
<div class="warn"><b>深度模型的数据饥饿：</b>1D CNN 在 41 颗训练电池下中位误差 32.7%，而在 99 颗训练电池的 5 折 CV 下降到 12.1%。这与 Severson 2019 最终选择弹性净而非深度网络的经验一致——本任务的有效样本量是「电池数」而非「循环数」。</div>
</section>

<section>
<h2><span class="n">04</span>特征与可解释性</h2>
<p class="lead">ΔQ(V) 曲线随老化整体抬升、谷值变浅；随机森林重要性显示容量衰减速率、ΔQ 方差、内阻增长率是主导特征。</p>
{img("fig_c_02_dq_aging.png", "图 8　不同寿命电池的 ΔQ(V)：第 2 圈（蓝）vs 第 100 圈（橙）")}
{img("fig_c_07_importance.png", "图 9　随机森林 Top-15 特征重要性", "72%")}
</section>

<section>
<h2><span class="n">05</span>物理仿真：ΔQ(V) 到底在测什么</h2>
<p class="lead">用 PyBaMM 做三件事：复现模型谱系、量化快充的发热代价、把 ΔQ(V) 拆成可命名的退化模式。</p>
{img("fig_d_01_model_hierarchy.png", "图 10　模型谱系：SPM 相对 DFN 只差 5 mV（结构误差可忽略），而同一 ECM 结构换一组内阻就偏 201 mV（参数误差致命）")}
{img("fig_d_04_degradation_modes.png", "图 11　三种退化模式对 ΔQ(V) 的影响：实测形态与锂库存损失（r=+0.64）、负极活性物质损失（r=+0.58）同向，与正极损失无关")}
{img("fig_d_05_protocol_life.png", "图 12　两种机制下七条协议都几乎无差别（离散度 0.08 / 0.05），比实测全体 2.61 小两个数量级——「协议 → 寿命」没复现出来")}
<div class="warn"><b>仿真增强的结论是否定的：</b>把 ΔQ(V) 分解出的退化模式系数加进预测模型，主测试 8.33%→8.41%、batch3 6.31%→6.69%，四种组合全部没有改善。原因是这些系数在 124 颗电池之间几乎没有差异（LLI 一律占主导），而仿真基形状与实测形态的相关也只有 0.64——信息量远低于 ΔQ(V) 的原始统计量。</div>
</section>

<section>
<h2><span class="n">06</span>第三方辨识的审计：能被测量 ≠ 能被辨识</h2>
<p class="lead">候选清单 P3 由另一个 AI 独立完成。把它的辨识链拆开逐条检验后，三个核心量全部不可辨识；而物理仿真给出的内阻真值（38.6 mΩ）正好可以当标尺。</p>
{img("fig_e_01_selfconsistency.png", "图 13　判据自证：参数翻倍或减半，复算残差仍 ≤ 0.31 mV；只有「拿错电池」才掉到 9.6–12.0 mV", "72%")}
{img("fig_e_02_prior_sensitivity.png", "图 14　r_dc 由 OCV 先验决定：先验 ±0.1 V ⇒ r_dc ±50 mΩ，三颗电池曲线完全重合", "72%")}
{img("fig_f_01_sampling_audit.png", "图 15　采样率决定 R0：真值恒为 38.6 mΩ，dt 加到 18.7 s 后报出 208.1 mΩ（膨胀 5.4 倍），R1/R2 变负", "72%")}
{img("fig_f_02_feature_families.png", "图 16　反过来解释了 P1 的选择：内阻对三种退化模式几乎不敏感（±7% 且方向不一致），ΔQ 深度跨模式差 25 倍", "72%")}
<div class="warn"><b>三线合并的一句话：</b>P1 用 ΔQ(V) 预测寿命；P2 证明 ΔQ(V) 编码的是 LLI + LAM_neg，并给出内阻真值；P3 的辨识链在结构上给不出内阻真值。<b>能被测量不等于能被辨识</b>——同一批电压曲线，ΔQ(V) 能把寿命预测到 6.20%，而「先验锚定 + 自证判据 + 粗采样」的链条可以让同一个内阻报出 65–208 mΩ 之间的任何值。</div>
</section>

<section>
<h2><span class="n">07</span>复现</h2>
<p class="lead">全部结果由以下命令产生，随机种子固定（42），重跑误差波动 &lt;5%。</p>
<p style="font-size:13px">
<code>python scripts/phase_b_build_dataset.py all</code>　构建统一数据集<br>
<code>python scripts/phase_a2_ecm.py</code>　ECM 辨识与老化轨迹<br>
<code>python scripts/phase_c_rul.py</code>　RUL 建模与评估<br>
<code>python scripts/phase_d1_pybamm_sim.py</code>　PyBaMM 模型谱系与快充发热<br>
<code>python scripts/phase_d2_sim_rul.py</code>　退化模式归因与仿真增强对照<br>
<code>python scripts/phase_e_p3_audit.py</code>　P3 辨识审计<br>
<code>python scripts/phase_f_bridge.py</code>　三线桥接（采样审计 / 特征族）<br>
<code>streamlit run src/app/app.py</code>　交互式小工具
</p>
</section>

<footer>数据来源：NASA PCoE Battery Data Set · Severson et al., Nature Energy 2019（MIT-Stanford）｜ 本看板为离线自包含文件，图片已内嵌</footer>
</div></body></html>"""

    path = OUT / "数据看板.html"
    path.write_text(html, encoding="utf-8")
    print(f"已生成：{path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
