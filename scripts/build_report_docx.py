"""生成项目总结报告（.docx，含封面、图表、表格、页脚页码）。

用法：python scripts/build_report_docx.py
产物：<项目根>/交付物/锂电池ECM建模与RUL预测_项目总结报告.docx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REP = ROOT / "reports"
OUT = ROOT.parent.parent / "交付物"
OUT.mkdir(parents=True, exist_ok=True)

INK = RGBColor(0x1A, 0x20, 0x2C)
BLUE = RGBColor(0x1F, 0x5F, 0xA9)
GREY = RGBColor(0x5A, 0x64, 0x72)
LINE = "D8DDE3"


def set_font(run, name_cn="宋体", name_en="Times New Roman", size=10.5, bold=False,
             color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name_en
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name_cn)
    if color is not None:
        run.font.color.rgb = color


def body(doc, text, size=10.5, first_indent=True, space_after=6, color=INK, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.45
    if first_indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_font(r, size=size, color=color)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.35
    r = p.add_run(text)
    set_font(r, size=size)
    return p


def h1(doc, text, num=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    if num:
        r = p.add_run(f"{num}  ")
        set_font(r, name_cn="黑体", size=15, bold=True, color=BLUE)
    r = p.add_run(text)
    set_font(r, name_cn="黑体", size=15, bold=True, color=INK)
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), "1F5FA9")
    pbdr.append(bottom)
    pPr.append(pbdr)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_font(r, name_cn="黑体", size=11.5, bold=True, color=BLUE)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_together = True
    r = p.add_run(text)
    set_font(r, size=9, color=GREY)
    return p


def add_figure(doc, name, cap, width=15.5):
    p = FIG / name
    if not p.exists():
        return
    doc.add_picture(str(p), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_before = Pt(8)
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    caption(doc, cap)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def add_table(doc, df: pd.DataFrame, widths=None, fmt=None, note=None, font_size=9):
    fmt = fmt or {}
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, c in enumerate(df.columns):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(c))
        set_font(r, name_cn="黑体", size=font_size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        shade(hdr[i], "1F5FA9")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(df.columns):
            v = row[c]
            if isinstance(v, float):
                txt = f"{v:.{fmt.get(c, 2)}f}"
            else:
                txt = str(v)
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(txt)
            set_font(r, size=font_size)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(note)
        set_font(r, size=8.5, color=GREY)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    r._r.append(fld1)
    r._r.append(instr)
    r._r.append(fld2)
    set_font(r, size=9, color=GREY)


def build() -> Path:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    sec = doc.sections[0]
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.6)
    add_page_number(sec)

    # ---------------- 封面 ----------------
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("锂电池等效电路建模\n与 RUL 寿命预测")
    set_font(r, name_cn="黑体", size=30, bold=True, color=INK)
    p.paragraph_format.line_spacing = 1.35

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    r = p.add_run("基于 NASA PCoE 与 MIT-Stanford 公开数据集的完整实现")
    set_font(r, name_cn="宋体", size=13, color=GREY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run("Thevenin 等效电路参数辨识 · ΔQ(V) 特征工程 · 早期循环寿命预测")
    set_font(r, name_cn="宋体", size=10.5, color=GREY)

    for _ in range(7):
        doc.add_paragraph()

    info = [
        ("项目阶段", "阶段 0 / A / B / C / D / E / F"),
        ("数据口径", "NASA PCoE 34 颗 · MIT-Stanford 124 颗（官方白名单）"),
        ("核心指标", "ECM 复测 RMSE 1.61% · RUL 主测试中位误差 6.20%"),
        ("报告日期", "2026 年 10 月"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        cells = t.add_row().cells
        cells[0].width = Cm(3.2)
        cells[1].width = Cm(9.5)
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r0 = p0.add_run(k + "　")
        set_font(r0, size=10, color=GREY)
        p1 = cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        set_font(r1, size=10)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 摘要 ----------------
    h1(doc, "摘要")
    body(doc, "本项目在纯线上条件下完成锂电池的两项核心建模任务：一是基于 NASA PCoE 数据集的 "
              "Thevenin 等效电路模型（ECM）参数辨识，二是基于 MIT-Stanford（Severson 2019）数据集的"
              "电池剩余寿命（RUL）早期预测。全部代码、数据产物、图表与报告均可复现。")
    body(doc, "ECM 部分对 4 颗 NASA 电池（B0005/B0006/B0007/B0018）完成 OCV-SOC 构建、欧姆内阻 R0、"
              "SOC 依赖直流内阻 R_dc(SOC) 与极化电阻 R_pol 的辨识，并给出内阻随 SOH 的老化轨迹。"
              "同工况复测电压 RMSE 分别为 1.72%、0.96%、3.29%、1.49%，中位数 1.61%，"
              "满足「≤实测量程 2%」的目标标准（4 颗中 3 颗达标，B0007 的偏差已在正文说明原因）。")
    body(doc, "RUL 部分严格复刻 Severson 等（Nature Energy, 2019）的数据口径与划分协议：从 140 颗原始电池中"
              "按官方 LoadData.m 筛出 124 颗（训练 41 / 主测试 43 / 二级测试 40），仅使用每颗电池前 100 圈数据，"
              "构建 60 维 ΔQ(V) 与老化特征，对比弹性净、随机森林、梯度提升、神经网络与一维卷积网络五类模型。"
              "梯度提升树在主测试集取得 6.20% 的中位相对误差、R²=0.93；改用论文 6 特征后主测试 8.58%、"
              "二级测试 8.03%，与基准论文的 9.1% 处于同一水平，且跨批次几乎不退化。")
    body(doc, "在两条主线之上补做了物理仿真层（PyBaMM）：以 DFN 为参照对比 SPM 与 Thevenin，证明等效电路模型的"
              "误差主要来自参数而非结构（同结构换参数，误差从 201 mV 降到 17 mV）；通过注入已知退化模式，"
              "把 ΔQ(V) 特征归因到锂库存损失（r=0.64）与负极活性材料损失（r=0.58），并排除正极活性材料损失（r=−0.07）；"
              "并说明在电压上限约束下，七条快充协议的 30 圈容量损失只差 1.7 个百分点——"
              "「协议决定寿命」这条链条在本仿真里没有复现出来。"
              "把仿真分解出的模式系数加进 RUL 模型未带来提升，这一负结果如实记录并分析了原因。")
    body(doc, "最后对候选清单第 3 项（P3，由另一个 AI 独立完成的 ECM 参数辨识）做了审计。"
              "把它的辨识链拆开逐条检验后发现：它的复算判据是自证的（参数翻倍残差不变）、直流内阻由人为指定的"
              "OCV 先验决定（先验 ±0.1 V 即 r_dc ±50 mΩ）、慢时间常数短于观测窗口（不可辨识）；"
              "而借助物理仿真给出的内阻真值可以定量证明，它报的阶跃内阻里有 63% 是 18.7 s 采样间隔造成的伪影。"
              "三条线合并后的结论是：**能被测量不等于能被辨识**。")

    h2(doc, "关键结果一览")
    kpi = pd.DataFrame([
        {"任务": "ECM 复测 RMSE（中位）", "结果": "1.61%", "目标值": "≤2%", "结论": "达标"},
        {"任务": "RUL 主测试中位误差（GBM，60 维）", "结果": "6.20%", "目标值": "≤10%", "结论": "达标"},
        {"任务": "RUL 二级测试中位误差（6 维）", "结果": "8.03%", "目标值": "同量级", "结论": "达标"},
        {"任务": "124 颗全量 5 折 CV（RF）", "结果": "7.37%", "目标值": "—", "结论": "补充验证"},
        {"任务": "官方 124 cell 白名单一致性", "结果": "零偏差", "目标值": "与论文一致", "结论": "达标"},
        {"任务": "仿真模型谱系误差（SPM vs DFN）", "结果": "5.16 mV", "目标值": "定性一致", "结论": "达标"},
        {"任务": "ΔQ(V) 机制归因（LLI 相关）", "结果": "0.64", "目标值": "定性成立", "结论": "达标"},
        {"任务": "仿真增强 RUL（四组对照）", "结果": "全部持平或变差", "目标值": "提升 ≥5%", "结论": "未达标（负结果）"},
        {"任务": "P3 辨识审计：R0 中采样伪影占比", "结果": "63%", "目标值": "可忽略", "结论": "不可辨识"},
        {"任务": "P3 辨识审计：参数 ×2 后的复算残差", "结果": "0.31 mV", "目标值": "应显著变差", "结论": "判据无效"},
    ])
    add_table(doc, kpi, widths=[6.2, 3.2, 3.6, 2.6],
              note="表 1　项目核心指标与目标对照")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 一、背景 ----------------
    h1(doc, "项目背景与目标", "一")
    body(doc, "锂离子电池的容量会随循环不可逆衰减。若能在电池「还健康」的前 100 圈就准确预测它能跑多少圈，"
              "就可以提前安排更换、优化快充策略、降低整包失效风险——这正是电池管理系统（BMS）最想解决的"
              "「早期寿命预测」问题。")
    body(doc, "项目设定三条主线：其一是等效电路建模（ECM），用物理可解释的电阻电容网络刻画电池端电压响应，"
              "为 BMS 的 SOC/SOH 估计提供参数基础；其二是数据驱动的 RUL 预测，用 ΔQ(V) 等老化特征回归循环寿命；"
              "其三是物理仿真（PyBaMM），用机理模型检验前两条线得到的结论。三条线共用同一套物理认识，互相印证："
              "ECM 给出「内阻随老化上升」的解释，RUL 模型把这个物理量当作特征使用，仿真则回答「这些特征在机理上"
              "究竟对应什么」。")

    h2(doc, "工作范围与目标标准")
    scope = pd.DataFrame([
        {"阶段": "0", "内容": "环境搭建与数据获取", "目标标准": "可一键装环境、数据 md5 归档", "结果": "完成"},
        {"阶段": "A", "内容": "ECM 参数辨识", "目标标准": "参数表 + 复测 RMSE ≤2%", "结果": "完成（中位 1.61%）"},
        {"阶段": "B", "内容": "统一预处理", "目标标准": "统一 Parquet + 质量报告", "结果": "完成（124+34 颗）"},
        {"阶段": "C", "内容": "RUL 预测建模", "目标标准": "测试中位误差 ≤10%，≥2 类模型", "结果": "完成（5 类模型，6.20%）"},
        {"阶段": "D", "内容": "创新延伸（可选）", "目标标准": "对照实验表", "结果": "完成（数据效率 + 仿真对照）"},
        {"阶段": "E", "内容": "数据小工具", "目标标准": "端到端展示", "结果": "完成（本地运行）"},
        {"阶段": "F", "内容": "总结交付", "目标标准": "报告 + 演示 + 可复现", "结果": "完成"},
        {"阶段": "G", "内容": "物理仿真（PyBaMM）", "目标标准": "模型谱系 + 机制归因 + 增强实验", "结果": "完成（增强为负结果）"},
    ])
    add_table(doc, scope, widths=[1.6, 4.2, 6.4, 4.2],
              note="表 2　各阶段目标标准与完成情况")

    # ---------------- 二、数据 ----------------
    h1(doc, "数据与预处理", "二")
    body(doc, "项目全程只使用两套公开数据：NASA PCoE 电池老化数据集（34 颗 18650，2 A 恒流充放电，"
              "含容量、内阻、温度）与 MIT-Stanford 的 Severson 2019 数据集（140 颗 A123 磷酸铁锂，"
              "多种快充策略，是早期寿命预测领域的标准基准）。")
    body(doc, "MIT 数据为 MATLAB v7.3（HDF5 封装），必须用 h5py 沿对象引用链逐条读取曲线。"
              "更关键的是「用哪些电池」：镜像含 140 颗，论文实际只用 124 颗。项目没有硬编码电池编号，"
              "而是完整复刻官方 LoadData.m 的处理顺序——把 batch2 中 5 颗续测电池拼回 batch1，"
              "剔除 batch3 的采集异常通道与末端容量异常电池，再剔除 5 颗未跑完的电池，最终得到 41+43+40=124 颗。")
    body(doc, "寿命标签定义为「首次放电容量低于 0.88 Ah（额定 1.1 Ah 的 80%）的循环号」。"
              "这里有一个容易踩的坑：第 1 圈是化成圈，放电容量记为 0，若直接搜索「容量 < 0.88」会把所有电池判成"
              "寿命 1 圈。剔除无效值后，重算标签与数据文件自带的 cycle_life 字段逐颗比对，最大偏差为 0 圈。")

    add_figure(doc, "fig_c_01_life_dist.png",
               "图 1　124 颗电池的循环寿命分布（训练 41 / 主测试 43 / 二级测试 40）")
    body(doc, "统一产物包含两张标量表与一个曲线立方体：mit_cycles.parquet 记录 100,501 条逐循环标量"
              "（容量、内阻、温度、充电时间、SOH），mit_dq.npz 存放 124×100×300 的 ΔQ(V) 曲线，"
              "NASA 侧则输出 7,565 条循环记录（含 2,794 次放电）。所有文件的 md5 与循环数抽查结果写入"
              "docs/data.md。")
    body(doc, "需要说明的是，原计划中的 CALCE 数据集因官网要求签署使用许可、本机未取得授权而缺席。"
              "阶段 A 与阶段 C 的主线结论均由 NASA + MIT 双数据集支撑，不影响结论，但跨数据集泛化的样本"
              "因此少了一套，这是本项目的主要数据局限。")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 三、ECM ----------------
    h1(doc, "等效电路建模与参数辨识", "三")
    h2(doc, "3.1 模型与辨识方法")
    body(doc, "采用一阶/二阶 RC 的 Thevenin 模型：端电压 = OCV(SOC) − I·R0 − I·R1(1−e^(−t/τ1)) − …。"
              "其中 OCV(SOC) 由前 20 圈充放电电压按 SOC 分箱平均、再用 PCHIP 单调插值平滑得到；"
              "R0 取放电电流阶跃（0→2 A）瞬间的压降比 ΔV/ΔI；SOC 依赖的直流内阻 R_dc(SOC) 由恒流区逐点"
              "(OCV−V)/I 分箱求取，极化电阻 R_pol = R_dc − R0。")
    body(doc, "必须诚实指出一个可辨识性限制：NASA 数据是恒流放电而非 HPPC 脉冲，放电起始只有一次电流阶跃，"
              "且极化电压（约 40 mV）小于 OCV 的估计不确定度，因此 R1/C1、R2/C2 的 SOC 依赖无法单独辨识——"
              "二阶全曲线拟合中 R2 会撞上参数上界。项目因此改用「R0 + R_dc(SOC)」这一可辨识组合，"
              "把 RC 时间常数作为整条曲线的集总参数处理。")

    add_figure(doc, "fig_a2_ocv_soc.png", "图 2　四颗 NASA 电池的 OCV-SOC 曲线（20 圈平均 + PCHIP 平滑）", 12.5)

    h2(doc, "3.2 辨识结果与验证")
    ecm = pd.read_csv(REP / "phaseA2_ecm_results.csv")
    ecm_show = ecm[["cell", "R0_step_mohm", "Rdc50_mohm", "Rpol50_mohm",
                    "rmse_test_raw_pct", "rmse_test_cal_pct"]].copy()
    ecm_show.columns = ["电池", "R0 (mΩ)", "R_dc(50%) (mΩ)", "R_pol(50%) (mΩ)",
                        "复测 RMSE 原始 (%)", "复测 RMSE 标定 (%)"]
    add_table(doc, ecm_show, widths=[2.2, 2.8, 3.4, 3.4, 3.4, 3.4],
              note="表 3　ECM 参数辨识结果与同工况复测误差（第 1 次放电辨识，第 3 次放电验证）")
    body(doc, "四颗电池的欧姆内阻高度一致（105–108 mΩ），总直流内阻在 SOC 50% 处为 128–136 mΩ，"
              "极化分量 21–31 mΩ，且随 SOC 降低而上升——低 SOC 下电荷转移阻抗增大，符合电化学直觉。"
              "复测 RMSE 中位数 1.61%，B0005 相比初版的 2.25% 降到 1.72%（改进来自 20 圈 OCV 平均与平滑），"
              "B0006、B0018 分别为 0.96%、1.49%。")
    body(doc, "B0007 的 3.29% 超标需要单独说明：该电池在前 20 圈内出现明显的容量再生现象，"
              "用前 20 圈平均构建的 OCV 被老化状态污染，导致系统偏差。这一现象在 NASA 数据集中有文献记载，"
              "属于数据本身的特性而非模型缺陷；若改用前 5 圈建 OCV 可缓解，但会引入更大的分箱噪声。")

    add_figure(doc, "fig_a2_r_aging.png", "图 3　内阻随 SOH 的老化轨迹（SOH 100%→60%）")
    body(doc, "把辨识从「单圈」扩展到「逐圈」，可以得到一条很有说服力的老化曲线：SOH 从 100% 降到 60% 的过程中，"
              "欧姆内阻 R0 上升约 20–25%，而 SOC 50% 处的总直流内阻上升近 80%。后者的增速远快于前者，"
              "说明老化主要恶化的是极化过程（界面膜增厚、电荷转移变慢），而非单纯的电解液/集流体欧姆阻抗。"
              "这条曲线也直接解释了 RUL 模型为什么把「初始内阻」与「内阻增长率」选为重要特征。")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 四、RUL ----------------
    h1(doc, "RUL 早期寿命预测", "四")
    h2(doc, "4.1 任务定义与协议")
    body(doc, "输入是每颗电池前 100 圈的 ΔQ(V) 曲线与逐循环标量，输出是它跑到 80% 容量时的循环数。"
              "划分严格按官方脚本：训练 41 颗、主测试 43 颗、二级测试 40 颗（batch3，实验批次不同），"
              "按电池分组，不存在循环级信息泄漏。")
    body(doc, "ΔQ(V) = Q_discharge(V) − Q_charge(V)，即同一循环中充、放电容量曲线在同一电压点相减。"
              "实现上有两个细节必须处理正确：一是放电容量自高压端起累积，要换算成「自低压端起算」才能与充电曲线"
              "对齐，否则会得到 ±700 mAh 的伪特征；二是部分批次充电起点在 2.47 V 附近，低压端需要用端点值外推。"
              "修正后 ΔQ(V) 的量级为 0–350 mAh，与文献一致。")

    add_figure(doc, "fig_c_02_dq_aging.png", "图 4　不同寿命电池的 ΔQ(V) 曲线：第 2 圈（蓝）与第 100 圈（橙）")

    h2(doc, "4.2 模型与结果")
    metrics = pd.read_csv(REP / "phaseC_metrics.csv")
    test = metrics[metrics.eval_set == "test"].drop_duplicates("model").set_index("model")
    sec = metrics[metrics.eval_set == "secondary"].drop_duplicates("model").set_index("model")
    rows = []
    for name, label in [("GradientBoosting", "梯度提升树 (GBM)"), ("ElasticNet", "弹性净 (ElasticNet)"),
                        ("RandomForest", "随机森林"), ("MLP", "神经网络 (MLP)"), ("CNN(1D)", "一维卷积网络 (1D CNN)"),
                        ("常数中位数(基线)", "常数中位数（基线）")]:
        rows.append({"模型": label,
                     "主测试中位误差": test.loc[name, "median_ape"],
                     "主测试 RMSE": test.loc[name, "rmse"],
                     "主测试 R²": test.loc[name, "r2"],
                     "二级测试中位误差": sec.loc[name, "median_ape"]})
    res = pd.DataFrame(rows)
    add_table(doc, res, widths=[5.0, 3.2, 2.8, 2.4, 3.2],
              note="表 4　五类模型在主测试集（43 颗）与二级测试集（40 颗）上的表现")

    body(doc, "梯度提升树在主测试集取得 6.20% 的中位相对误差与 0.93 的 R²；改用论文 6 特征后主测试 8.58%、二级测试 8.03%，"
              "是唯一在两个测试集都接近 10% 的模型，因此推荐作为交付主模型。"
              "弹性净（论文同款）主测试 7.77%，但换到 batch3 退化到 28.64%——"
              "线性模型对批次分布漂移更敏感，这个对比本身就是一个有价值的结论。")

    add_figure(doc, "fig_c_04_pred_scatter.png", "图 5　预测值与真实值对比（对数坐标，虚线为理想对角线）")

    h2(doc, "4.3 深度模型为什么没赢")
    body(doc, "一维卷积网络（3 通道 ΔQ 曲线 + 标量特征，5 折内部交叉验证 + 早停 + 集成）在 41 颗训练电池下"
              "中位误差 32.7%，预测值明显向均值收缩。把训练电池数从 41 增加到 99（全量 5 折交叉验证），"
              "同样的网络降到 12.1%。")
    body(doc, "这说明本任务的有效样本量是「电池数」而不是「循环数」——每颗电池只有 100 个循环点，"
              "但电池与电池之间的差异（策略、批次、个体）才是模型要学的东西。41 颗电池不足以支撑一个卷积网络的"
              "参数估计，这与 Severson 等人最终选择弹性净而非深度网络的判断一致。")

    add_figure(doc, "fig_c_05_learning_curve.png", "图 6　学习曲线：训练电池数 10→41，误差持续下降且尚未收敛", 12.5)
    add_figure(doc, "fig_c_06_cv.png", "图 7　全量 124 颗 5 折交叉验证：RF 7.37% 最优，CNN 12.11%", 12.5)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 五、讨论 ----------------
    h1(doc, "讨论：模型看到了什么", "五")
    body(doc, "随机森林的特征重要性排序给出了相当清晰的故事：前五位是 dq_diff_logmin、dq_diff_mean、ir_2、"
              "dq_diff_min、policy_4。也就是说，模型主要依赖三件事——ΔQ(V) 曲线从第 2 圈到第 100 圈整体"
              "下移的幅度、电池初始内阻、以及充电策略本身。前者对应锂库存损失（LLI）与活性材料损失（LAM）的"
              "早期信号，后两者则是「这块电池一开始就被怎么对待」的先验信息。")
    add_figure(doc, "fig_c_07_importance.png", "图 8　随机森林 Top-15 特征重要性", 12.5)

    body(doc, "但这条解释链只到「相关」为止：真实电池里 LLI、LAM_pos、LAM_neg 总是同时发生，观测数据无法把它们拆开，"
              "所以「ΔQ(V) 测的是 LLI」始终是一个未经检验的假设。第六节用物理仿真来检验它。")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 六、物理仿真 ----------------
    h1(doc, "物理仿真：把「ΔQ 在测什么」放进模型里检验", "六")
    body(doc, "第四、五节给出的都是相关性证据：ΔQ(V) 特征有效、模型主要依赖它，但「它究竟测的是锂库存损失还是"
              "活性材料损失」这个问题，仅靠观测数据回答不了——因为真实电池里几种退化模式总是同时发生，无法拆开。")
    body(doc, "物理仿真恰好能做这件事：往模型里注入**已知**的单一退化模式，看哪一种能产生与实测同形的 ΔQ 曲线。"
              "本节用 PyBaMM 搭建三层保真度模型（DFN / SPM / Thevenin），参数集选 Prada2013（A123 2.3 Ah 磷酸铁锂/石墨，"
              "与 MIT-Stanford 数据同化学体系），完成三件事：验证模型谱系、量化快充协议的代价、以及给 ΔQ(V) 做机制归因。"
              "最后把仿真特征加进 RUL 模型，检验它能否提升精度。")

    h2(doc, "6.1 模型谱系：误差在参数，不在结构")
    body(doc, "以 DFN 为参照，三条曲线都在 1C 放电工况下对比。SPM 忽略电解液输运，误差只有 5 mV；"
              "Thevenin 用阶段 A 从 NASA 数据辨识出的参数（R0=0.107 Ω、R1=0.021 Ω、τ=30 s）直接套到 A123 上，"
              "误差高达 201 mV。把 R0/R1/τ 按 SPM 曲线重新反演后（R0=0.030 Ω、R1=0.099 Ω），误差降到 17 mV。")
    body(doc, "这个对照的意义在于：等效电路模型的精度瓶颈通常不是「一阶 RC 太简单」，而是「参数不是这块电池的」。"
              "同一套结构、换一组参数，误差可以差一个数量级——这为阶段 A 为什么必须逐电池辨识提供了独立佐证。")
    hier = pd.DataFrame([
        {"模型": "DFN（全阶，参照）", "RMSE vs DFN / mV": 0.0, "最大偏差 / mV": 0.0, "单次放电耗时": "0.42 s"},
        {"模型": "SPM（单粒子）", "RMSE vs DFN / mV": 5.16, "最大偏差 / mV": 5.38, "单次放电耗时": "0.17 s"},
        {"模型": "Thevenin（阶段 A 参数）", "RMSE vs DFN / mV": 200.95, "最大偏差 / mV": 226.84, "单次放电耗时": "0.02 s"},
        {"模型": "Thevenin（反演 SPM 参数）", "RMSE vs DFN / mV": 17.19, "最大偏差 / mV": 45.24, "单次放电耗时": "0.02 s"},
    ])
    add_table(doc, hier, widths=[5.6, 3.4, 3.0, 3.2],
              note="表 5　三档保真度模型在 1C 放电下相对 DFN 的电压误差")
    add_figure(doc, "fig_d_01_model_hierarchy.png",
               "图 9　左：四种模型的 1C 放电曲线；右：误差拆分为结构误差与参数误差")

    h2(doc, "6.2 倍率能力与快充协议")
    body(doc, "放电倍率越高，可放出的容量越少：0.5C 放出 2.119 Ah，4C 只剩 1.129 Ah，容量利用率从 100% 掉到 53.3%，"
              "能量从 6.66 Wh 掉到 3.40 Wh。这是后面所有快充讨论的物理背景——「充得快」的代价不只在寿命，也在可用容量。")
    rate = pd.DataFrame([
        {"倍率": "0.5C", "可用容量 / Ah": 2.119, "能量 / Wh": 6.665, "容量利用率": "100.0%"},
        {"倍率": "1C", "可用容量 / Ah": 1.940, "能量 / Wh": 6.028, "容量利用率": "91.5%"},
        {"倍率": "2C", "可用容量 / Ah": 1.600, "能量 / Wh": 4.894, "容量利用率": "75.5%"},
        {"倍率": "3C", "可用容量 / Ah": 1.329, "能量 / Wh": 4.026, "容量利用率": "62.7%"},
        {"倍率": "4C", "可用容量 / Ah": 1.129, "能量 / Wh": 3.399, "容量利用率": "53.3%"},
    ])
    add_table(doc, rate, widths=[2.4, 4.0, 3.6, 4.0],
              note="表 6　倍率能力（Prada2013，2.0 V 截止）")

    body(doc, "快充协议的对比给出两个结论。第一，**t80 能压 2.8 倍，总时长只能压 1.5 倍**：1C 恒流的 t80 是 45.2 min、"
              "总时长 85.4 min；4.8C 多步把 t80 压到 16.2 min（−64%），总时长却只降到 56.8 min（−34%）。"
              "差的这一块全在恒压段——恒流越快，恒压越长。所以「插枪到拔枪」的时长不是衡量快充的好指标，"
              "Severson 用 t80 是有道理的。第二，**温升随 C 率单调上升**，从 8.4 K 到 29.9 K（4.8C 最高）。")
    proto = pd.DataFrame([
        {"协议": "1C 恒流（基线）", "t80 / min": 45.2, "总时长 / min": 85.4, "温升 / K": 8.4},
        {"协议": "3C 恒流", "t80 / min": 19.5, "总时长 / min": 60.1, "温升 / K": 23.6},
        {"协议": "3.6C(80%)-3.6C(10%)-3.6C(10%)", "t80 / min": 17.8, "总时长 / min": 58.5, "温升 / K": 26.4},
        {"协议": "4.8C(80%)-3.6C(10%)-3.2C(10%)", "t80 / min": 16.2, "总时长 / min": 56.8, "温升 / K": 29.9},
        {"协议": "6C(60%)-3C(40%)", "t80 / min": 16.2, "总时长 / min": 56.7, "温升 / K": 28.9},
        {"协议": "3C(20%)-1C(80%)", "t80 / min": 37.9, "总时长 / min": 78.1, "温升 / K": 16.3},
        {"协议": "1C(80%)-3C(20%)", "t80 / min": 45.0, "总时长 / min": 85.2, "温升 / K": 8.4},
    ])
    add_table(doc, proto, widths=[6.6, 3.0, 3.2, 2.8],
              note="表 7　七条快充协议的 t80、总时长与温升（t80 指充到 80% 荷电量的时长）")
    body(doc, "表里最后两行值得单独说。`1C(80%)-3C(20%)` 是把 3C 段放在 80% SOC 之后，"
              "而它的每一个数字都与 1C 恒流几乎相同（45.0 vs 45.2 min、85.2 vs 85.4 min、8.4 vs 8.4 K）。"
              "原因是高 SOC 段的端电压一上来就贴着 4.2 V 上限，这段 3C 恒流**一步也没走成**，"
              "整条协议退化为 1C 恒流——两条协议的容量衰减（3.7666% vs 3.7666%）甚至逐位相同。"
              "这说明**名义协议与实际执行轨迹可以是两件完全不同的事**。")
    add_figure(doc, "fig_d_03_fastcharge_heat.png",
               "图 10　左：t80 与温升的权衡（place_high 与 base_1c 重合）；右：t80 与总时长的分组对比")

    h2(doc, "6.3 机制归因：ΔQ(V) 到底在测什么")
    body(doc, "分别注入三种退化模式（锂库存损失 LLI、正极活性材料损失 LAM_pos、负极活性材料损失 LAM_neg），"
              "各取 3%/6%/10% 三档，观察 ΔQ(V) 曲线的深度如何响应。")
    modes = pd.DataFrame([
        {"退化模式": "锂库存损失 LLI", "ΔQ 深度 @10% / mAh": 235.2, "容量损失 @10%": "12.12%", "与实测形态相关": 0.643},
        {"退化模式": "负极活性材料损失 LAM_neg", "ΔQ 深度 @10% / mAh": 229.4, "容量损失 @10%": "11.83%", "与实测形态相关": 0.581},
        {"退化模式": "正极活性材料损失 LAM_pos", "ΔQ 深度 @10% / mAh": 9.5, "容量损失 @10%": "0.007%", "与实测形态相关": -0.066},
    ])
    add_table(doc, modes, widths=[5.6, 4.2, 3.0, 3.4],
              note="表 8　三种退化模式的 ΔQ(V) 响应（相关系数为归一化曲线与 MIT 实测中位曲线的皮尔逊相关）")
    body(doc, "结论很清楚：**ΔQ(V) 主要测的是 LLI 与 LAM_neg，对 LAM_pos 几乎不敏感**（相关 −0.07，ΔQ 深度小一个数量级）。"
              "这与电化学直觉一致——正极容量损失不改变负极的锂库存，也就不会在同一电压点上产生充放电容量差。"
              "MIT 实测的中位 ΔQ 深度是 38.8 mAh，落在仿真 3%–6% 退化对应的区间内，量级自洽。")
    add_figure(doc, "fig_d_04_degradation_modes.png",
               "图 11　三种退化模式各三档的 ΔQ(V) 曲线（灰带为 MIT 实测中位曲线）")

    h2(doc, "6.4 充电协议能不能决定寿命：一条没复现出来的链条")
    body(doc, "把七条协议各跑 30 圈老化，分两种机制：纯 SEI 生长（所有参数集都支持）与 SEI + 不可逆析锂"
              "（只有 OKane2022 带析锂参数，这是让充电协议真正影响寿命的机制）。")
    life = pd.DataFrame([
        {"机制": "纯 SEI 生长", "30 圈容量损失区间": "3.48% – 3.77%", "离散度（极差/均值）": 0.08},
        {"机制": "SEI + 不可逆析锂", "30 圈容量损失区间": "34.88% – 36.62%", "离散度（极差/均值）": 0.05},
        {"机制": "MIT 实测（同一策略内）", "30 圈容量损失区间": "—", "离散度（极差/均值）": 0.33},
        {"机制": "MIT 实测（全部策略）", "30 圈容量损失区间": "—", "离散度（极差/均值）": 2.61},
    ])
    add_table(doc, life, widths=[5.6, 5.2, 5.4],
              note="表 9　仿真与实测的协议间寿命离散度对比")
    body(doc, "析锂把容量损失的绝对水平放大了十倍（3.5% → 35%），却**没有放大协议之间的差异**："
              "七条协议的 t80 相差 2.8 倍，30 圈容量损失只差 1.7 个百分点（离散度 0.05），"
              "比实测的 2.61 小两个数量级。**「协议 → 寿命」这条链条在本仿真里没有复现出来。**")
    body(doc, "原因可以指认。第一，所有协议最终都要把同样的电量充进去（都充到 4.2 V、再恒压到 C/20），"
              "而在本参数集下不可逆析锂的累积量对充电速率不敏感，只与充入的总电量有关。"
              "第二，电压上限把激进协议的恒流段截断，实际轨迹趋同——6.1 节已经看到 `1C(80%)-3C(20%)` "
              "干脆退化为 1C 恒流。第三，30 圈的窗口太短，协议差异需要更长时间才显现。")
    body(doc, "按定安计划的预期，这一节本该给出「充电协议显著影响寿命」的证据。结果相反，如实记录。"
              "不过这个负结果反过来支持了 RUL 一侧的判断：MIT 数据里充电时长与寿命的 Spearman 相关只有 −0.05，"
              "时间本来就不是那条通道；能被时间指标区分开的东西，仿真也区分不开。"
              "真正的信息在 ΔQ(V) 按电压分辨的形状里——这也正是第 6.3 节已经证明的事。")
    add_figure(doc, "fig_d_05_protocol_life.png",
               "图 12　左：七条协议在两种机制下的 30 圈容量损失（两种机制下都几乎重合）；右：离散度四方对比")

    h2(doc, "6.5 仿真增强为什么没提升精度")
    body(doc, "既然仿真能给出机制归因，最自然的想法是：把仿真分解出的模式系数（LLI/LAM_pos/LAM_neg 占比）"
              "作为新特征加进 RUL 模型。做法是以三种模式的仿真曲线为基，对每颗电池前 100 圈的 ΔQ 曲线做最小二乘分解，"
              "得到三个系数并归一化为占比。每个实验臂各自 5 折交叉验证选正则强度，保证对等比较。")
    aug = pd.DataFrame([
        {"实验臂": "P1 基线（论文 6 特征）", "ElasticNet 主测试": 8.33, "ElasticNet 二级测试": 6.31,
         "GBM 主测试": 8.58, "GBM 二级测试": 8.03},
        {"实验臂": "P1 + 仿真模式系数（9 维）", "ElasticNet 主测试": 8.41, "ElasticNet 二级测试": 6.69,
         "GBM 主测试": 9.63, "GBM 二级测试": 8.31},
        {"实验臂": "仅仿真模式系数（3 维）", "ElasticNet 主测试": 23.02, "ElasticNet 二级测试": 43.13,
         "GBM 主测试": 19.81, "GBM 二级测试": 34.43},
    ])
    add_table(doc, aug, widths=[6.2, 2.9, 3.0, 2.7, 2.9],
              note="表 10　仿真特征增强的对照实验（中位相对误差 / %，越低越好）")
    body(doc, "四组对照全部持平或变差（8.33→8.41、6.31→6.69、8.58→9.63、8.03→8.31），没有一组达到"
              "预设的「提升 ≥5%」标准。只用三个模式系数则完全不可用（19.8%–43.1%）。这是一个**负结果**，"
              "原因有三层：")
    for txt in [
        "模式系数在 124 颗电池之间几乎没有差异——LLI 占比的标准差只有 0.018，LAM_pos 0.011，LAM_neg 0.027。"
        "一个几乎不变化的特征不可能提供预测信息。",
        "仿真基与实测形态的相关只有 0.64（LLI），基函数本身就不够准，分解出的系数带有系统性偏差。",
        "ΔQ(V) 的原始统计量（xq_log_var / min / mean）已经是对这条曲线的最优压缩。仿真系数是「有物理含义的"
        "低维投影」，信息量严格少于原始曲线，用它替代或补充都只会丢信息。",
    ]:
        bullet(doc, txt)
    body(doc, "所以这一节的定位需要说清楚：**仿真解释了 ΔQ(V) 为什么有效，但替代不了它。**"
              "它的价值在机制归因（6.3 节）与「名义协议 ≠ 实际轨迹」这类工程判断（6.2 节），"
              "而不在给预测模型加特征。把负结果如实写出来，比强行调参凑一个「提升 6%」更有价值——"
              "后者往往只是换了个划分方式。")
    add_figure(doc, "fig_d_06_mode_features.png",
               "图 13　左：四组对照的分组柱状；右：124 颗电池的模式系数分布——几乎无区分度")


    # ---------------- 七、P3 审计 ----------------
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    h1(doc, "第三方 ECM 辨识的审计：能被测量不等于能被辨识", "七")
    body(doc, "候选清单第 3 项（P3）是一份由另一个 AI 独立完成的 ECM 参数辨识，交付了一套完整的辨识链："
              "松弛段双指数拟合定 τ1/τ2 与 R1/R2、用先验 OCV(0.5)=3.86 V 锚定直流内阻、按幅值比例分解出 R0:R1:R2、"
              "再反推出 OCV 曲线、最后用零阶保持离散化仿真回原工况算残差。自报复算 RMSE ≤ 8.7 mV，"
              "按它自己定的「≤ 2% 量程」判据是达标的。")
    body(doc, "本项目没有重做它，而是把这条链拆开逐条检验。三个核心量——R0/R1/R2 的分解、直流内阻 r_dc、"
              "慢时间常数 τ2——**全部不可辨识**。完整推导见 docs/audit.md，这里给出三组关键证据。")

    h2(doc, "7.1 判据自证：参数翻倍，残差不变")
    body(doc, "P3 的 OCV 不是独立测出来的，而是从参数反推的：OCV(t) := V_meas(t) + I·ΣRᵢ(1−e^(−t/τᵢ))。"
              "这条定义的作用就是「让模型精确复现测量值」，于是再仿真回去时残差只剩离散化误差，"
              "与参数对不对无关。用同一套流程做三组扰动即可验证：")
    sc = pd.DataFrame([
        {"参数集": "P3 原参数", "B0005 / mV": 0.12, "B0006 / mV": 0.16, "B0007 / mV": 0.15},
        {"参数集": "全部参数 ×2", "B0005 / mV": 0.24, "B0006 / mV": 0.31, "B0007 / mV": 0.31},
        {"参数集": "全部参数 ×0.5", "B0005 / mV": 0.06, "B0006 / mV": 0.08, "B0007 / mV": 0.08},
        {"参数集": "换另一颗电池的模型", "B0005 / mV": 9.63, "B0006 / mV": 8.68, "B0007 / mV": 11.96},
    ])
    add_table(doc, sc, widths=[5.0, 3.4, 3.4, 3.4],
              note="表 11　自洽性检验：给定任意参数，按 P3 方式重建 OCV 再仿真回去的残差")
    body(doc, "内阻翻倍或减半，RMSE 都压在 0.31 mV 以内；只有「拿错电池」才会掉到 mV 级。"
              "判据阈值是 50 mV（2% 量程），**任何一组数都能通过**。"
              "同一个机制还留下一个指纹：反推 OCV 的幅度随参数线性放大（×0.5 时 0 mV，×2 时 23.5–62.9 mV），"
              "说明这条 OCV 曲线本身就是参数的函数。")
    add_figure(doc, "fig_e_01_selfconsistency.png",
               "图 14　左：三组参数下的复算残差（几乎不变）；右：三组参数反推出的 OCV 曲线——低 SOC 处出现非物理的回落")

    h2(doc, "7.2 参数由先验决定，τ2 由窗口决定")
    body(doc, "P3 的直流内阻完全由第 ② 步决定：r_dc = (OCV50_prior − V(SOC=0.5)) / I。分子里的人为先验是 3.86 V，"
              "分母是恒流值。把它从 3.76 扫到 3.96 V（±0.1 V），三颗电池的 r_dc 从 115.5 / 116.6 / 116.0 mΩ "
              "一路涨到 214.8 / 215.9 / 216.6 mΩ——**先验动 1 mV，r_dc 动 0.5 mΩ**。"
              "P3 自述的不确定度是 ±25 mΩ，低估了 4 倍。")
    body(doc, "更关键的是：三颗电池在任何先验下曲线都完全重合。所以它报的「三颗电池 r_dc 一致"
              "（165.2 / 166.2 / 166.3 mΩ）」不是辨识结论，而是同一个先验被复读了三次。")
    body(doc, "慢时间常数的问题更直接。B0006 的松弛窗口只有 77.7 s（8 个采样点），却报出 τ2 = 95.0 s——"
              "**时间常数比观测窗口还长**。把 B0005 的窗口砍到 45%，τ2 从 162.4 s 掉到 94.3 s（−42%）。"
              "τ2 的点估计没有意义，只有「τ2 ≳ 窗口长度」这个下界是有信息的。")
    add_figure(doc, "fig_e_02_prior_sensitivity.png",
               "图 15　先验 OCV(0.5) 扫描：r_dc 与先验严格线性，且三颗电池曲线重合")

    h2(doc, "7.3 用物理仿真当标尺：采样率决定 R0")
    body(doc, "前两节说明参数不可辨识，这一节给出量级。物理仿真能提供「真值」：把 P3 的整条流程原样套到 "
              "DFN 的合成放电曲线上，只改采样间隔，其余一切不动。")
    samp = pd.DataFrame([
        {"采样间隔 dt": "0.05 s", "R0 / mΩ": 38.6, "R1+R2 / mΩ": -3.7, "r_dc / mΩ": 34.8, "首点压降/I / mΩ": 38.6},
        {"采样间隔 dt": "0.5 s", "R0 / mΩ": 51.5, "R1+R2 / mΩ": -16.6, "r_dc / mΩ": 34.8, "首点压降/I / mΩ": 51.2},
        {"采样间隔 dt": "2 s", "R0 / mΩ": 74.1, "R1+R2 / mΩ": -39.3, "r_dc / mΩ": 34.8, "首点压降/I / mΩ": 71.2},
        {"采样间隔 dt": "5 s", "R0 / mΩ": 101.8, "R1+R2 / mΩ": -67.0, "r_dc / mΩ": 34.8, "首点压降/I / mΩ": 90.5},
        {"采样间隔 dt": "18.7 s（NASA 实际）", "R0 / mΩ": 208.1, "R1+R2 / mΩ": -173.6, "r_dc / mΩ": 34.5, "首点压降/I / mΩ": 105.0},
    ])
    add_table(doc, samp, widths=[4.6, 2.8, 2.8, 2.6, 3.4],
              note="表 12　同一段合成曲线的 R0 随采样间隔的变化（真值恒为 38.6 mΩ）")
    body(doc, "三件事同时发生：R0 从 38.6 膨胀到 208.1 mΩ（5.4 倍），R1、R2 变成负数（双指数拟合失去物理意义，"
              "只剩数学补偿），而 r_dc 恒定在 34.8 mΩ（因为它被先验锚定，与采样无关）。")
    body(doc, "有一个数字可以直接对上：P3 报的「阶跃内阻」在 B0005/6/7 上是 107.3 / 105.9 / 107.5 mΩ，"
              "而合成曲线在 dt = 18.7 s 下算出来是 105.0 mΩ——几乎逐位重合。"
              "这说明它测到的「阶跃内阻」里，真值只占 38.6 mΩ，**其余 63% 是采样伪影**。")
    add_figure(doc, "fig_f_01_sampling_audit.png",
               "图 16　采样间隔从 0.05 s 加到 18.7 s：R0 单调膨胀、R1/R2 变负、r_dc 不动")

    h2(doc, "7.4 反过来，这解释了为什么用 ΔQ(V) 而不用内阻")
    body(doc, "审计到这里其实已经回答了本文第二节那个悬而未决的问题：为什么全部特征都建在 ΔQ(V) 上，"
              "而不建在更「物理」的内阻上。在同一套 LFP 参数集上分别注入三种退化模式，同时看两个观测量：")
    fam = pd.DataFrame([
        {"退化模式": "锂库存损失 LLI", "R0 @3% → @10%": "37.6 → 35.8 mΩ", "内阻变化": "−7.1%", "ΔQ 深度 @3% → @10%": "70.6 → 235.2 mAh"},
        {"退化模式": "正极活性材料损失 LAM_pos", "R0 @3% → @10%": "38.7 → 38.9 mΩ", "内阻变化": "+0.8%", "ΔQ 深度 @3% → @10%": "2.6 → 9.5 mAh"},
        {"退化模式": "负极活性材料损失 LAM_neg", "R0 @3% → @10%": "39.3 → 41.0 mΩ", "内阻变化": "+6.2%", "ΔQ 深度 @3% → @10%": "68.8 → 229.4 mAh"},
    ])
    add_table(doc, fam, widths=[4.8, 3.6, 2.4, 4.4],
              note="表 13　三种退化模式下内阻与 ΔQ(V) 的敏感度对照")
    body(doc, "内阻对三种退化模式几乎不敏感（全在 ±7% 以内），而且**方向还不一致**（LLI 让它降、LAM_neg 让它升）；"
              "ΔQ(V) 深度跨模式差 25 倍，且随退化程度单调放大。拿内阻当特征，等于用一个分不出退化模式的量"
              "去区分退化模式——这就是 ΔQ(V) 成为主特征的物理原因。")
    add_figure(doc, "fig_f_02_feature_families.png",
               "图 17　左：内阻对三种退化模式几乎不敏感；右：ΔQ 深度跨模式差 25 倍")

    h2(doc, "7.5 三条线合并后的一句话")
    body(doc, "**P1 用 ΔQ(V) 预测寿命；P2 证明 ΔQ(V) 编码的是 LLI 与 LAM_neg，并给出了内阻的真值；"
              "P3 的辨识链在结构上给不出内阻真值。** 三条线拼在一起的结论是一句更朴素的话："
              "**能被测量不等于能被辨识**。")
    body(doc, "同一批电压曲线，用对了观测量（ΔQ(V)）可以把寿命预测到 6.20% 的中位误差；"
              "用错了链条（先验锚定 + 自证判据 + 粗采样），同一个内阻可以报出 65 mΩ 到 208 mΩ 之间的任何值，"
              "而且每一步都有「看起来达标」的残差作证。这也解释了为什么第四节要强调特征口径——"
              "口径与辨识链条，比模型结构更能决定结论的上限。")
    body(doc, "修复建议已在 docs/audit.md 第七节列出，核心两条：判据必须留独立验证集（另一圈、另一 SOC、另一倍率），"
              "以及不要用单点 OCV 先验锚定内阻（改用多倍率稳态压差或脉冲序列的电压差分解）。")

    # ---------------- 八、局限 ----------------
    h1(doc, "局限与后续工作", "八")
    for txt in [
        "CALCE 数据集缺席：跨数据集泛化只做了 batch1/2 → batch3 的批次迁移，未做跨化学体系验证。",
        "深度模型未调优到可用：受 41 颗训练电池限制，CNN 的结构搜索空间未充分展开；若有更多电池，"
        "可尝试迁移学习或更强的数据增广。",
        "ECM 的可辨识性限制仍在：要拿到真正的 R1/C1、R2/C2 SOC 依赖，需要 HPPC 脉冲或 EIS 数据。"
        "第六节用 PyBaMM 生成仿真放电曲线反演 ECM 参数，但那只能验证参数标定流程，替代不了实测脉冲。",
        "仿真老化的绝对量级不可信：SEI 溶剂扩散系数需要放大 1000 倍标定才能落到真实寿命量级，"
        "因此仿真只用于比较协议之间的相对差异，不能当作寿命绝对值预测。",
        "析锂参数集与目标化学体系不一致：只有 OKane2022（NMC811/SiOx）带析锂参数，套用到 A123 磷酸铁锂上"
        "是「机制对了、参数不对」，这是仿真增强失败的三个原因之一。",
        "充电协议的名义时长不等于实际轨迹：恒流段被 4.2 V 上限截断，`1C(80%)-3C(20%)` 里的 3C 段一步也没走成，"
        "整条协议退化为 1C 恒流。七条协议的 30 圈容量损失只差 1.7 个百分点（离散度 0.05，实测 2.61），"
        "「协议 → 寿命」这条链条没有复现出来；30 圈的窗口也偏短。",
        "P3 的辨识结果不能当参数用：R0 里 63% 是 18.7 s 采样间隔造成的伪影，r_dc 由 OCV 先验决定，τ2 短于观测窗口。"
        "本项目做了审计并给出修复建议，但没有替 P3 重做一遍辨识。",
        "Streamlit 小工具只在本地验证：部署到公网需要用户自己的 Streamlit Cloud 账号。",
    ]:
        bullet(doc, txt)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------- 九、结论 ----------------
    h1(doc, "结论与交付物", "九")
    body(doc, "项目按计划完成了从数据获取、统一预处理、ECM 参数辨识、RUL 建模到交付的全部主线："
              "ECM 复测 RMSE 中位 1.61%，RUL 主测试中位误差 6.20%（论文口径 8.58%，对标论文 9.1%），两项均达到目标标准。"
              "过程中修正了初版的三个实质性错误——MIT 寿命标签口径、124 颗白名单筛选、ΔQ(V) 的容量起点换算，"
              "并把这些坑写进了工作记录，便于后续复现。")
    body(doc, "在此之上补做了物理仿真层：用 PyBaMM 的三档模型证明「等效电路误差在参数不在结构」，"
              "用退化模式注入证明「ΔQ(V) 主要测 LLI 与 LAM_neg、对 LAM_pos 不敏感」，"
              "并量化了七条快充协议的时间与热代价（t80 压 2.8 倍、总时长只压 1.5 倍）。"
              "仿真特征未能提升 RUL 精度、协议差异也未能拉开寿命，这两个负结果都已如实记录。")
    body(doc, "最后一节是对第三方 P3 辨识的审计。三条线合并后的结论可以压成一句话："
              "**能被测量不等于能被辨识**——同一批电压曲线，用 ΔQ(V) 能把寿命预测到 6.20%，"
              "用「先验锚定 + 自证判据 + 粗采样」的链条，同一个内阻可以报出 65–208 mΩ 之间的任何值。"
              "这条结论比任何单项指标都更能说明「特征口径」与「辨识链条」为什么决定了一个工作的上限。")
    body(doc, "交付物包括：可复现代码仓库（解析器 / ECM / 特征 / 模型 / 仿真 / 审计 / 小工具）、统一数据集、"
              "全部目标图表、五份技术小结（method / results / sim / audit / worklog）、离线数据看板，"
              "以及本报告与配套汇报演示。")
    h2(doc, "复现命令")
    for cmd, desc in [
        ("python scripts/phase_b_build_dataset.py all", "构建统一数据集（MIT 需先放好 3 个 .mat）"),
        ("python scripts/phase_a2_ecm.py", "ECM 辨识与内阻老化轨迹"),
        ("python scripts/phase_c_rul.py", "RUL 建模、评估与出图（约 4 分钟）"),
        ("python scripts/phase_d1_pybamm_sim.py", "PyBaMM 模型谱系 / 倍率 / 快充协议（约 15 秒）"),
        ("python scripts/phase_d2_sim_rul.py", "退化模式归因 / 协议寿命 / 仿真增强（约 40 秒）"),
        ("python scripts/phase_e_p3_audit.py", "P3 辨识审计：自洽性 / 先验敏感性 / 可辨识性（约 40 秒）"),
        ("python scripts/phase_f_bridge.py", "三线桥接：采样率审计 + 内阻/ΔQ 特征族（约 90 秒）"),
        ("python scripts/build_dashboard.py", "生成离线 HTML 数据看板"),
        ("streamlit run src/app/app.py", "启动交互式数据小工具"),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Cm(0.6)
        r = p.add_run(cmd)
        set_font(r, name_en="Consolas", size=9.5, color=BLUE)
        r2 = p.add_run("　—　" + desc)
        set_font(r2, size=9.5, color=GREY)

    path = OUT / "锂电池ECM建模与RUL预测_项目总结报告.docx"
    doc.save(str(path))
    return path


if __name__ == "__main__":
    p = build()
    print(f"已生成：{p}  ({p.stat().st_size / 1024:.0f} KB)")
    sys.exit(0)
