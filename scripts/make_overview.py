"""生成项目总览图：三条线如何接成一条证据链。

产出 ``figures/project_overview.png``（1600×760），用于 README 与 LEARN.md 的头部概览。
不需要任何原始数据，纯示意 + 已在报告中定稿的数字。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "figures" / "project_overview.png"

DARK = "#1a202c"
GREY = "#4a5568"
BLUE = "#2b6cb0"
TEAL = "#2c7a7b"
ORANGE = "#c05621"
BG = "#f7fafc"


def box(ax, x, y, w, h, title, lines, edge, face="#ffffff", title_size=13, body_size=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=edge, zorder=3)
    ax.text(x + 0.022, y + h - 0.135, "\n".join(lines), ha="left", va="top",
            fontsize=body_size, color=GREY, linespacing=1.55, zorder=3)


def arrow(ax, x1, y1, x2, y2, label="", color=BLUE, rad=0.0, lx=0.0, ly=0.055):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=17, linewidth=1.7,
            color=color, connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="bottom",
                fontsize=8.8, color=color, style="italic", zorder=3)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 7.6), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "锂电池 ECM 建模与 RUL 寿命预测 —— 一条互相校验的证据链",
            ha="center", va="top", fontsize=17, fontweight="bold", color=DARK)
    ax.text(0.5, 0.925, "NASA PCoE + MIT-Stanford 公开数据 · 只用前 100 圈 · 124 颗电池 · 预测循环寿命 148–2237 圈",
            ha="center", va="top", fontsize=10.5, color=GREY)

    # ---- 数据层：放在最上方作为输入 ----
    box(ax, 0.035, 0.775, 0.93, 0.125,
        "数据层（两份公开数据集，口径先冻结再建模）",
        ["NASA PCoE：B0005 / B0006 / B0007 / B0018 恒流放电 → 辨识 Thevenin 参数（同工况复测 RMSE 中位 1.61%）",
         "MIT-Stanford：124 颗（官方 LoadData.m 筛选顺序）→ 逐循环曲线、寿命标签、ΔQ(V) 立方体"],
        edge="#4a5568", face="#edf2f7", title_size=12, body_size=9.2)

    # ---- 三条线 ----
    box(ax, 0.030, 0.285, 0.295, 0.455,
        "P1 · 数据线",
        ["把「为什么变差」和「还能跑多久」\n做成可复现的代码\n\n"
         "• 二阶 ECM：R0 / R1 / R2 / τ1 / τ2\n"
         "• OCV-SOC 查表 + PCHIP\n"
         "• 60 维宽表 → 论文 6 维两套口径\n"
         "• GBM 主测试 6.20%（R²=0.93）\n"
         "• 5 折 CV 7.37% ± 3.25\n\n"
         "关键判据：\n极化电压 ≈ 40 mV < OCV 不确定度\n→ SOC 依赖的 R1/R2 不可单独辨识"],
        edge=BLUE)

    box(ax, 0.352, 0.285, 0.295, 0.455,
        "P2 · 物理仿真线（标尺）",
        ["PyBaMM：SPM / DFN / Thevenin(反演)\n\n"
         "• 三档对比：等效电路的误差\n"
         "  在参数，不在结构\n"
         "  （相对 DFN 5.16 / — / 17.19 mV）\n"
         "• 注入三种退化模式：\n"
         "  ΔQ(V) ← LLI (r=+0.64)\n"
         "           + LAM_neg (r=+0.58)\n"
         "  LAM_pos 不敏感 (r=−0.07)\n"
         "• t80 45.2 → 16.2 min（2.8 倍）\n"
         "  总时长只压 1.5 倍\n\n"
         "修掉：PyBaMM 24.x/26.x 电流符号\n相反导致充放电静默对调"],
        edge=TEAL)

    box(ax, 0.674, 0.285, 0.295, 0.455,
        "P3 · 审计线",
        ["检验另一条辨识链的三个核心量\n\n"
         "① 判据自证：参数 ×2 / ×0.5，\n"
         "   复算 RMSE 仍 ≤ 0.31 mV\n"
         "② 参数由先验决定：r_dc\n"
         "   先验 ±0.1 V → ±50 mΩ\n"
         "③ τ2 短于观测窗口：\n"
         "   162.4 → 94.3 s（截断 45%）\n\n"
         "→ 三个量全部不可辨识\n\n"
         "修复建议见 docs/audit.md"],
        edge=ORANGE)

    # ---- 箭头：数据层 → 三条线（向下，走三列中线） ----
    arrow(ax, 0.183, 0.770, 0.183, 0.745, "", color=GREY)
    arrow(ax, 0.500, 0.770, 0.500, 0.745, "", color=GREY)
    arrow(ax, 0.821, 0.770, 0.821, 0.745, "", color=GREY)
    for x, lab in ((0.183, "恒流阶跃 + 松弛段"), (0.500, "逐循环数据 + 寿命标签"), (0.821, "待审的辨识链")):
        ax.text(x, 0.755, lab, ha="center", va="center", fontsize=8.6,
                color=GREY, style="italic", zorder=3,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=BG, edgecolor="none"))

    # ---- 三条线之间的横箭头 ----
    arrow(ax, 0.328, 0.510, 0.349, 0.510, "真值标尺\n内阻 38.6 mΩ", color=TEAL, lx=-0.004, ly=0.030)
    arrow(ax, 0.650, 0.510, 0.671, 0.510, "δQ / 内阻\n敏感度对照", color=TEAL, lx=0.006, ly=0.030)

    # ---- 结论带：两条桥 ----
    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.150), 0.93, 0.105,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0, facecolor="#fffaf0", zorder=2,
        )
    )
    ax.text(0.5, 0.222, "两条桥接出来的结论",
            ha="center", va="center", fontsize=10.5, color=ORANGE, fontweight="bold", zorder=3)
    ax.text(0.5, 0.183,
            "P3 → P2：采样率决定 R0 —— dt 从 0.05 s 加到 18.7 s，R0 从 38.6 膨胀到 208.1 mΩ（真值恒定），"
            "它报的 105–107 mΩ 里 63% 是伪影        "
            "P2 → P1：内阻对三种退化模式 ±7% 且方向不一致，ΔQ 深度跨模式差 25 倍 → 这就是用 ΔQ 而不用内阻的原因",
            ha="center", va="center", fontsize=9.0, color=DARK, zorder=3)

    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.012), 0.93, 0.118,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=2.2, edgecolor="#744210", facecolor="#fffff0", zorder=2,
        )
    )
    ax.text(0.5, 0.092, "能被测量 ≠ 能被辨识",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#744210", zorder=3)
    ax.text(0.5, 0.048,
            "同一批电压曲线，用 ΔQ(V) 能把寿命预测到 6.20%；链条错了，同一个内阻可以报出 65–208 mΩ 之间的任何值，"
            "而且每一步都有「看起来达标」的残差作证。",
            ha="center", va="center", fontsize=9.6, color=DARK, zorder=3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=BG)
    print(f"saved -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
