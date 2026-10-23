<div align="center">

# 锂电池等效电路建模与 RUL 寿命预测

**能被测量 ≠ 能被辨识 —— 基于公开电池数据的三阶段可复现研究**

[![CI](https://github.com/ilo-love7/battery-ecm-rul/actions/workflows/ci.yml/badge.svg)](https://github.com/ilo-love7/battery-ecm-rul/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-NASA%20PCoE%20%7C%20MIT--Stanford-blueviolet)](data/README.md)

<p align="center">
  <a href="./README.md">English</a> · <b>简体中文</b>
</p>

</div>

---

用 NASA PCoE 与 MIT-Stanford 公开数据，把「电池为什么变差」和「还能跑多久」做成一套可复现的代码。

**一个项目，三个阶段**：P1 ECM 辨识与 RUL 预测（复现）· P2 PyBaMM 物理仿真（复现 + 创新）· P3 第三方辨识的审计（把前两条线接上）

## 结果速览

| 任务 | 指标 | 结果 | 参考 |
|---|---|---|---|
| ECM 参数辨识 | 同工况复测电压 RMSE（4 颗 NASA 电池） | **1.61%**（中位）<br>0.96 / 1.49 / 1.72 / 3.29 % | 实测量程 2% |
| 早期寿命预测 | 主测试集 43 颗中位相对误差 | **6.20%**（GBM，R²=0.93） | Severson 2019：9.1% |
| 跨批次泛化 | 二级测试集 40 颗（batch3）中位相对误差 | **8.03%**（论文 6 特征，GBM） | 同配置主测试 8.58% |
| 全量交叉验证 | 124 颗 5 折 CV 中位相对误差 | **7.37% ± 3.25**（RF） | — |
| 物理仿真归因 | ΔQ(V) 形态与 MIT 实测的相关 | **+0.64**（锂库存损失）<br>+0.58（负极活性物质损失） | 见 [docs/sim.md](docs/sim.md) |
| 第三方辨识审计 | P3 报的 R0 中采样伪影占比 | **63%**（真值 38.6 mΩ，报 105–107 mΩ） | 见 [docs/audit.md](docs/audit.md) |

只使用每颗电池**前 100 圈**的数据，预测它跑到 80% 容量时的循环数（148–2237 圈，中位 736.5 圈）。

> 上表第一行与论文同为官方划分，但特征与模型配置不同（本项目 60 维 + GBM，论文 6 特征 + ElasticNet），仅作量级参照。同配置的严格对标见下文「特征口径消融」。

---

## 这个项目怎么读

候选清单里的三项工作（P1 ECM/RUL、P2 物理仿真、P3 第三方辨识）原本分散推进，本仓库把它们收拢成
**一个项目的三个阶段**，逐层提供证据：

| 阶段 | 做什么 | 关键产出 | 文档 |
|---|---|---|---|
| **P1** | NASA 上辨识 Thevenin 参数；MIT 上前 100 圈预测寿命 | 复测 RMSE 1.61%；RUL 中位误差 6.20% | [docs/method.md](docs/method.md) · [docs/results.md](docs/results.md) |
| **P2** | PyBaMM 建 DFN/SPM/Thevenin，注入已知退化模式 | ΔQ(V) 归因 LLI+LAM_neg（r 0.64/0.58）；内阻真值 38.6 mΩ | [docs/sim.md](docs/sim.md) |
| **P3** | 审计第三方辨识链，并用 P2 当真值标尺 | 判据自证、参数由先验决定、τ2 不可辨识 | [docs/audit.md](docs/audit.md) |

三阶段合并的结论：**能被测量 ≠ 能被辨识**。三条线的衔接如下。

## 三条主线的衔接

<table>
<tr>
<td width="50%"><img src="figures/fig_a2_fit_B0005.png" alt="ECM 拟合"></td>
<td width="50%"><img src="figures/fig_c_02_dq_aging.png" alt="ΔQ(V) 老化信号"></td>
</tr>
<tr>
<td align="center"><b>主线一 · ECM</b>：Thevenin 等效电路拟合恒流放电曲线，<br>给出 OCV(SOC)、R0、R_dc(SOC) 等物理可解释参数</td>
<td align="center"><b>主线二 · RUL</b>：ΔQ(V) 曲线的整体错位量<br>是早期老化最灵敏的信号</td>
</tr>
</table>

前两条线共用同一套逐循环数据：ECM 解释「为什么变差」，RUL 回答「还能跑多久」。
第三条线（PyBaMM 物理仿真）负责把两者接上——用含明确机制的电化学模型说明
**ΔQ(V) 究竟测的是什么**，见 [第三节](#三物理仿真pybamm)。

第四条线（[第四节](#四第三方-ecm-辨识的审计p3)）是一次反向检验：把第三方独立完成的 P3 ECM 辨识
拿过来审计，用第三条线给出的内阻真值当标尺。结论是 **P3 报的参数不可辨识**，
这反过来解释了为什么第一条线与第二条线必须分工——**能被测量 ≠ 能被辨识**。

---

## 一、ECM 参数辨识（NASA PCoE）

**模型**：Thevenin 一阶 RC

$$V_t = \mathrm{OCV}(\mathrm{SOC}) - I\,R_0 - I\,R_1\left(1 - e^{-t/\tau_1}\right)$$

**辨识流程**

1. **OCV(SOC)**：取前 20 圈充放电电压按 SOC 分箱平均（抵消充放电极化滞回），再用 PCHIP 单调插值平滑。
2. **OCV 偏置标定**：用训练圈残差拟合三阶多项式 δ(SOC)，消除「充放电平均法」在低 SOC 段的系统偏差。
3. **R0**：放电电流阶跃（0 → 2 A）瞬间的 ΔV/ΔI，毫秒级，几乎不受 OCV 误差污染。
4. **R_dc(SOC)**：恒流区逐 SOC 分箱计算 (OCV − V)/I；减去 R0 得到极化分量 R_pol。
5. **验证**：第 1 次放电辨识，第 3 次放电同工况复测，计算电压 RMSE。

**结果**

| 电池 | R0 (mΩ) | R_dc(50%) (mΩ) | R_pol(50%) (mΩ) | 复测 RMSE |
|---|---|---|---|---|
| B0005 | 107.35 | 128.50 | 21.15 | 1.72% |
| B0006 | 105.92 | 130.45 | 24.54 | **0.96%** |
| B0007 | 107.58 | 128.27 | 20.68 | 3.29% ⚠️ |
| B0018 | 105.05 | 135.68 | 30.63 | 1.49% |

> ⚠️ **B0007 未达标（3.29%）**：该电池前 20 圈存在明显容量再生，OCV 估计被污染。这是数据本身的特性，不做隐藏。

**一个可迁移的结论**：SOH 从 100% 降到 70%，总直流内阻上升约 **33%**，而欧姆内阻只涨 **5%** —— 老化主要恶化的是极化过程，不是欧姆阻抗。

**可辨识性边界**：NASA 是恒流放电（非 HPPC 脉冲），一次电流阶跃下极化电压仅约 40 mV，小于 OCV 的估计不确定度，因此 R1/C1、R2/C2 的 SOC 依赖**不可单独辨识**。二阶拟合会让 R2 撞上参数上界——这是数据决定的，不是模型缺陷。

---

## 二、早期寿命预测（MIT-Stanford）

**任务**：输入前 100 圈的 ΔQ(V) 曲线 + 逐循环标量，输出寿命（至 80% 容量）。

**特征（两套口径）**

| 组 | 内容 | 维度 |
|---|---|---|
| 论文 ΔQ(V) | `xq_log_var` / `xq_log_min` / `xq_log_mean`（Q₁₀₀ − Q₁₀ 的 log 统计） | 3 |
| 滞回统计 | 第 2 圈 / 第 100 圈 / 两者差值的 min、var、log\|min\|、mean | 12 |
| 滞回定点 | 3.0 / 3.2 / 3.4 / 3.5 V 处的容量差 | 12 |
| 老化标量 | 容量、内阻、充电时间（首末值、比值、斜率、截距）、温度、库仑效率 | 18 |
| 充电策略 | 15 维快充协议参数 | 15 |

`feature_sets()` 提供 `full`（60 维）与 `paper6`（论文 6 特征）两套列名，用于消融对比。

**主协议结果**（严格按官方划分：训练 41 / 主测试 43 / 二级测试 40，60 维特征）

| 模型 | 主测试中位误差 | 平均误差 | RMSE (圈) | R² | 二级测试中位误差 |
|---|---|---|---|---|---|
| **GradientBoosting** | **6.20%** | 12.87% | **105.1** | **0.93** | 16.37% |
| ElasticNet（论文同款） | 7.77% | 21.27% | 369.7 | 0.11 | 28.64% |
| MLP | 7.69% | 26.01% | 378.7 | 0.06 | 24.71% |
| RandomForest | 8.12% | 16.22% | 158.9 | 0.84 | **11.89%** |
| 1D CNN | 32.67% | 38.95% | 527.4 | −0.82 | 57.09% |
| 常数中位数（基线） | 17.14% | 29.54% | 431.6 | −0.22 | 45.33% |

![模型对比](figures/fig_c_03_model_compare.png)

![预测散点](figures/fig_c_04_pred_scatter.png)

> **统计口径提示（[docs/results.md](docs/results.md) 2.6）**：n=43 的样本量下，上表前四名模型的差距大多不具统计显著性——GBM 6.20% 对 EN 7.77% 的 Wilcoxon 配对检验 p=0.93，bootstrap 95% CI 宽达 5–9 个百分点。排名是描述性的；只有 RandomForest 显著劣于 GBM（p=0.006）、1D CNN 断崖式失败（p=1.8e-07）、二级测试上 RF 显著优于 GBM（p=0.0012）这三条是推断性结论。

**特征口径消融：6 维 vs 60 维**

同一划分、同一模型，只换特征集：

| 模型 | 主测试（60 维） | 主测试（6 维） | batch3（60 维） | batch3（6 维） |
|---|---|---|---|---|
| ElasticNet | 7.77% | 8.33% | 28.64% | **6.31%** |
| GradientBoosting | **6.20%** | 8.58% | 16.37% | **8.03%** |
| RandomForest | 8.12% | 9.38% | 11.89% | **8.22%** |

![特征口径消融](figures/fig_c_08_ablation.png)

**三个可以带走的判断**

1. **特征口径决定跨批次泛化**。60 维里混进了编码充电协议的滞回特征：同分布 6.20% 很好看，跨批次 16.37% 就露馅。换成论文 6 维，8.58% → 8.03%，几乎不退化；弹性网更是从 28.64% 降到 6.31%（**4.5 倍**）。维度不是越多越好，关键是维度里有没有编码「环境」而不只是「电池」。
2. **小样本下树模型最稳**。GBM 主测试 6.20%（R²=0.93），RF 跨批次 11.89%，两者各守一头。ElasticNet 在 60 维上 R² 只有 0.11——中位数好看、方差极大，不适合交付。
3. **深度模型要的是「电池数」，不是「循环数」**。同一个 1D CNN，41 颗训练电池时 32.7%，99 颗（5 折 CV）时降到 12.1%。每颗电池只有 100 个循环点，模型真正要学的是电池之间的差异，41 颗撑不起卷积网络的参数估计。

**补充协议**：全量 124 颗 5 折交叉验证（按电池分组，无循环级泄漏）

| 模型 | 中位误差（均值 ± 标准差） |
|---|---|
| RandomForest | **7.37% ± 3.25** |
| GradientBoosting | 7.46% ± 1.66 |
| ElasticNet | 7.86% ± 0.99 |
| MLP | 9.56% ± 1.08 |
| 1D CNN | 12.11% ± 1.50 |

---

## 三、物理仿真（PyBaMM）

前两节都是数据驱动的：ECM 的参数是拟合出来的，ΔQ(V) 的特征是统计出来的。这一节用
[PyBaMM](https://pybamm.org) 把「为什么」补上。完整方法与局限见 [docs/sim.md](docs/sim.md)。

**① 模型谱系：误差主要来自参数，不是结构**

1C 恒流放电，以 DFN 为参照：

| 模型 | 相对 DFN 电压 RMSE | 单次求解 |
|---|---|---|
| DFN（全阶，电解液输运 + 颗粒扩散） | 0（参照） | 0.42 s |
| SPM（单粒子） | **5.2 mV** | 0.17 s |
| Thevenin（用 SPM 反演参数） | 17.2 mV | 0.02 s |
| Thevenin（用阶段 A 辨识参数） | **201.0 mV** | 0.02 s |

**结构降阶只花 5–17 mV，参数错配要花 200 mV。** 阶段 A 从 NASA B0005 辨识的 R0 = 107 mΩ
是老化态的值，而 Prada2013 新电池只有约 30 mΩ——ECM 没错，是参数不能跨电池状态搬运。

![模型谱系](figures/fig_d_01_model_hierarchy.png)

**② 快充：t80 能压 2.8 倍，总时长只能压 1.5 倍**

7 条多步恒流协议的仿真结果：1C 的 t80 是 **45.2 min**、总时长 85.4 min；4.8C 多步把 t80
压到 **16.2 min**（−64%），总时长却只降到 56.8 min（−34%）——差的这一块全在恒压段。
所以「插枪到拔枪」的时长不是衡量快充的好指标，Severson 用 t80 是有道理的。

温升从 8.4 K 升到 29.9 K（`sev_4_8` 最高）。另外有个副产品：把 3C 段放到 80% SOC 之后
（`1C(80%)-3C(20%)`），端电压一上来就贴着 4.2 V，这段恒流**一步也没走成**，整条协议退化为
1C 恒流——两条协议的容量衰减、温升、t80 逐位相同。

![快充协议](figures/fig_d_03_fastcharge_heat.png)

**③ ΔQ(V) 到底测什么**

在 LFP 参数集上分别注入三种退化模式，看 ΔQ(V) 被掰成什么形状，再与 MIT 实测比对：

| 退化模式 | 与实测形态的相关 |
|---|---|
| 锂库存损失 LLI | **r = +0.64** |
| 负极活性物质损失 LAM_neg | **r = +0.58** |
| 正极活性物质损失 LAM_pos | r = −0.07 |

**实测的 ΔQ(V) 主要由「锂库存损失 + 负极活性物质损失」决定，正极几乎不参与**——这正是
LFP 电池的化学特性。ΔQ(V) 之所以能预测寿命，是因为它同时编码了「损失多少」和「损失在哪」。

![退化模式](figures/fig_d_04_degradation_modes.png)

**④ 一个诚实的负结果**

把仿真拆出的三种模式系数当作特征喂回预测模型，**没有任何改善**：

| 特征组合 | 主测试 43 颗 | batch3 40 颗 |
|---|---|---|
| P1 基线（论文 6 特征，EN） | 8.33% | **6.31%** |
| P1 + 仿真模式系数（EN） | 8.41% | 6.69% |
| P1 基线（GBM） | 8.58% | 8.03% |
| P1 + 仿真模式系数（GBM） | 9.63% | 8.31% |

原因是模式系数在 124 颗电池之间几乎没有差异（LLI 占比标准差仅 0.018），而仿真基与实测形态
的相关只有 0.64——**物理可解释性不等于统计有效性**。按项目验收口径（提升 ≥5%），这一项未达标，
如实记录。

> **一句话总结**：仿真解释了 ΔQ(V) 为什么有效，但替代不了它。

---

## 四、第三方 ECM 辨识的审计（P3）

候选清单第 3 项（P3）是一份由**另一个 AI 独立完成**的 ECM 参数辨识。它交付了一套完整的
辨识链：松弛段双指数拟合 → 先验 OCV 锚定 r_dc → 幅值比例分解 R0:R1:R2 → 反推 OCV → 仿真复算。
复算 RMSE ≤ 8.7 mV，看起来通过了「≤2% 量程」的验收。

本项目没有重做它，而是把它的链条拆开逐条检验。**结论是它报的三个核心量都不可辨识**，
完整推导见 [docs/audit.md](docs/audit.md)。

**① 判据自证：参数翻倍，残差不变**

P3 的 OCV 是从参数反推出来的（`OCV := V_meas + I·ΣRᵢ(1−e^(−t/τᵢ))`），再用同一组参数仿真回去，
残差只剩离散化误差：

| 参数集 | B0005 | B0006 | B0007 |
|---|---|---|---|
| P3 原参数 | 0.12 mV | 0.16 mV | 0.15 mV |
| 参数 ×2 | 0.24 mV | 0.31 mV | 0.31 mV |
| 参数 ×0.5 | 0.06 mV | 0.08 mV | 0.08 mV |
| 换另一颗电池的模型 | 9.63 mV | 8.68 mV | 11.96 mV |

内阻翻倍或减半，RMSE 都压在 0.31 mV 以内。**这个判据对参数没有任何约束力。**

**② 参数由先验决定**

`r_dc_mid = (OCV50_prior − V(SOC=0.5)) / I`。先验从 3.76 扫到 3.96 V（±0.1 V），
三颗电池的 r_dc 从 115.5/116.6/116.0 一路涨到 214.8/215.9/216.6 mΩ——**先验动 1 mV，r_dc 动 0.5 mΩ**。
P3 自述的不确定度 ±25 mΩ，低估了 4 倍。而三颗电池在任何先验下曲线都重合，
所以它报的「三颗电池 r_dc 一致」不是辨识结论，是同一个先验被复读了三次。

**③ τ2 是窗口的函数**

B0006 的松弛窗口只有 77.7 s，却报出 τ2 = 95.0 s——时间常数比观测窗口还长。
把 B0005 的窗口砍到 45%，τ2 从 162.4 s 掉到 94.3 s。

![可辨识性](figures/fig_e_03_identifiability.png)

**④ 采样率决定 R0：P2 当标尺**

把 P3 的流程原样套到 PyBaMM 的 DFN 合成曲线上，只改采样间隔：

| dt | R0 / mΩ | R1+R2 / mΩ | 首点压降/I |
|---|---|---|---|
| 0.05 s | **38.6**（真值） | −3.7 | 38.6 |
| 2 s | 74.1 | −39.3 | 71.2 |
| 18.7 s（NASA 实际间隔） | **208.1** | −173.6 | 105.0 |

采样越粗，R0 膨胀 5.4 倍、R1/R2 变负。P3 报的「阶跃内阻」105.9–107.5 mΩ 与合成曲线在
dt=18.7 s 下的 105.0 mΩ 几乎逐位重合——**它测到的东西里有 63% 是采样伪影**。

![采样审计](figures/fig_f_01_sampling_audit.png)

**⑤ 反过来，这解释了 P1 为什么不用内阻**

| 退化模式 | R0 变化（3%→10%） | ΔQ 深度（3%→10%） |
|---|---|---|
| 锂库存损失 LLI | 37.6 → 35.8 mΩ（−7.1%） | 70.6 → 235.2 mAh |
| 正极活性材料损失 LAM_pos | 38.7 → 38.9 mΩ（+0.8%） | 2.6 → 9.5 mAh |
| 负极活性材料损失 LAM_neg | 39.3 → 41.0 mΩ（+6.2%） | 68.8 → 229.4 mAh |

内阻对三种退化模式几乎不敏感且方向不一致；ΔQ(V) 跨模式差 25 倍。
**这就是 P1 全部特征建在 ΔQ(V) 上的物理原因。**

![特征族](figures/fig_f_02_feature_families.png)

> **三线合并的一句话**：P1 用 ΔQ(V) 预测寿命；P2 证明 ΔQ(V) 编码的是 LLI + LAM_neg，并给出内阻真值；
> P3 的辨识链在结构上给不出内阻真值。**能被测量 ≠ 能被辨识。**

---

## 五、数据口径：124 颗是怎么来的

MIT-Stanford 三个 batch 里并不是每颗电池都能进论文口径。本项目**不硬编码编号**，而是完整复刻官方 `LoadData.m`（见 [`reference/LoadData.m`](reference/LoadData.m)）的处理顺序：

| batch | 处理 | 保留 |
|---|---|---|
| batch1 | 46 颗，先把 batch2 的 5 颗续测电池拼接回来，再剔除 5 颗未跑完的电池 | 41 |
| batch2 | 5 颗续测（batch1 电池的后半程）拼回 batch1 后，其余保留 | 43 |
| batch3 | 删第 38 颗（channel 46 采集异常）→ 删末端容量 > 0.885 Ah → 删噪声电池 [3, 40, 41] | 40 |
| **合计** | | **124 = 41 + 43 + 40** |

**寿命标签**：首次 `QDischarge < 0.88 Ah` 的循环号。第 1 圈是化成圈、容量记为 0，**必须跳过**，否则所有电池都会被判成 1 圈。重算标签与文件自带 `cycle_life` 逐颗比对，最大偏差 **0 圈**。

> 口径不对齐，后面所有指标都不可比。这一步是整个项目的地基。

---

## 用途 / 你能用它做什么

六件具体的事，每件都有可直接运行的入口（仅 CPU；单元测试不需要原始数据集）：

| 用途 | 真实入口（脚本/模块路径） |
|---|---|
| **在 NASA PCoE 电池上辨识 Thevenin ECM 参数** —— OCV(SOC)、R0、R_dc(SOC)、同工况复测 RMSE | `python scripts/phase_a2_ecm.py` · `src/ecm/ecm.py` |
| **用前 100 圈预测早期循环寿命（MIT-Stanford）** —— 60 维 / 论文 6 维两套口径、五类模型 | `python scripts/phase_c_rul.py` · `src/rul/features.py`、`src/rul/models.py` |
| **跑 PyBaMM 物理仿真** —— DFN/SPM/Thevenin 谱系、快充协议、退化模式归因 | `python scripts/phase_d1_pybamm_sim.py` · `python scripts/phase_d2_sim_rul.py` · `src/sim/runner.py` |
| **审计第三方 ECM 辨识链（P3）** —— 用 P2 给出的内阻真值当标尺 | `python scripts/phase_e_p3_audit.py` · `python scripts/phase_f_bridge.py` |
| **构建统一数据集，或交互式查看结果** | `python scripts/phase_b_build_dataset.py all` · `streamlit run src/app/app.py` |
| **重新生成交付物**（Word 报告 + 离线 HTML 看板） | `python scripts/build_report_docx.py` · `python scripts/build_dashboard.py` |

典型用途（Typical uses）：(1) 公开电池数据上 ECM 辨识与早期寿命预测的**基线复现**；
(2) **真值标尺** —— PyBaMM 一侧给出已知内阻（38.6 mΩ），任何辨识流程都可以拿它检验；
(3) **审计模板** —— 判断第三方辨识报出的参数哪些能过敏感性检验、哪些不能；
(4) **教学 / 上手材料** —— 讲清「能被测量 ≠ 能被辨识」的区别，并把负面结果原样保留。

可复现性：`python -m pytest tests/ -q` 跑 **9 个测试**，全部用合成数据、约 2 秒跑完，**不需要原始数据集** ——
它们守住 ECM 参数恢复、ΔQ(V) 两套口径、以及对外报告指标的定义。完整流水线在数据下载完成后完全离线
（NASA PCoE + MIT-Stanford 约 8.6 GB，下载与放置方式见 [`data/README.md`](data/README.md)），
PyBaMM 相关步骤另需 `requirements-full.txt`。CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)，
workflow `CI`）只有一个 **ubuntu-latest / Python 3.11** 的 job：对 `src` + `scripts` 跑 `compileall`、
跑 9 个单元测试、再做一次 import 冒烟测试。Windows 不在 CI 覆盖范围内，本地请在 Windows 上自行跑一遍测试命令。

---

## 六、快速开始

```bash
git clone https://github.com/ilo-love7/battery-ecm-rul.git
cd battery-ecm-rul
pip install -r requirements.txt          # 基础依赖
# pip install -r requirements-full.txt   # 追加 torch / streamlit
```

**准备数据**：按 [`data/README.md`](data/README.md) 下载 NASA PCoE 与 MIT-Stanford 数据集，放到 `data/raw/`。也可用环境变量指定：

```bash
export MIT_DIR=/path/to/mit      # Windows: set MIT_DIR=D:\data\mit
export NASA_DIR=/path/to/nasa
```

**复现**

```bash
# 0) 单元测试（合成数据，不需要原始数据集，约 2 秒）
python -m pytest tests/ -q

# 1) 统一数据集：原始 .mat → Parquet / NPZ（MIT 8.3 GB 约 20 分钟）
python scripts/phase_b_build_dataset.py all

# 2) ECM 参数辨识 + 内阻老化轨迹（约 1 分钟）
python scripts/phase_a2_ecm.py

# 3) RUL 建模与评估（约 4 分钟，CPU）
python scripts/phase_c_rul.py

# 4) PyBaMM 物理仿真（约 1 分钟；需 pip install -r requirements-full.txt）
python scripts/phase_d1_pybamm_sim.py
python scripts/phase_d2_sim_rul.py

# 5) P3 审计与三线桥接（约 2 分钟）
python scripts/phase_e_p3_audit.py
python scripts/phase_f_bridge.py

# 5) 重新生成交付物（Word 报告 + HTML 看板）
python scripts/build_report_docx.py
python scripts/build_dashboard.py

# 6) 交互式小工具
streamlit run src/app/app.py
```

或使用 Makefile：`make test && make data && make ecm && make rul && make sim && make audit`。
`tests/` 里的 9 个用例全部用合成数据，覆盖 ECM 参数恢复（真值可复现）、ΔQ(V) 两种口径的量纲与端点、
以及对外报告指标的同口径定义——不需要下载 8.6 GB 原始数据也能验证核心算法没被改坏。

**产物**

| 路径 | 内容 |
|---|---|
| `reports/phaseC_metrics.csv` | 五类模型 × 两个测试集的完整指标 |
| `reports/phaseC_cv.csv` | 5 折交叉验证逐折结果 |
| `reports/phaseC_predictions.csv` | 每颗电池 5 个模型的预测值 |
| `reports/phaseA2_ecm_results.csv` | ECM 参数与复测 RMSE |
| `reports/phaseD1_*.csv` / `phaseD2_*.csv` | 仿真谱系、协议、退化模式、仿真增强对照 |
| `reports/phaseE_*.csv` / `phaseF_*.csv` | P3 审计（参数、自洽残差、先验扫描、可辨识性）与桥接（采样审计、特征族） |
| `figures/` | 全部结果图 |
| `data/processed/` | 可重建的中间数据集（已 git 忽略） |

---

## 七、仓库结构

```
.
├── src/
│   ├── parsers/          nasa.py（MATLAB v7）/ mit.py（v7.3 HDF5 + 官方 124 颗口径）
│   ├── ecm/              Thevenin 模型、OCV-SOC、最小二乘辨识
│   ├── features/         dq.py：ΔQ(V) 特征工程
│   ├── rul/              features.py 60 维宽表 + 两套口径 / models.py 五类模型
│   ├── sim/              PyBaMM 封装：models.py 模型与参数 / protocols.py 快充协议 / runner.py 实验与曲线抽取
│   └── app/              app.py：Streamlit 小工具
├── tests/                test_core.py：9 个合成数据用例（ECM 参数恢复 / ΔQ 口径 / 指标定义）
├── scripts/
│   ├── phase_b_build_dataset.py   原始数据 → 统一 Parquet / NPZ
│   ├── phase_a2_ecm.py            ECM 辨识 + OCV 标定 + 内阻老化
│   ├── phase_c_rul.py             RUL 建模、评估、出图
│   ├── phase_d1_pybamm_sim.py     PyBaMM 模型谱系、倍率、快充发热
│   ├── phase_d2_sim_rul.py        退化模式归因 + 仿真增强对照
│   ├── phase_e_p3_audit.py        P3 审计：自洽性、先验敏感性、可辨识性
│   ├── phase_f_bridge.py          三线桥接：采样率审计 + 内阻/ΔQ 特征族
│   ├── build_dashboard.py         生成离线 HTML 看板
│   └── build_report_docx.py       生成 Word 总结报告
├── docs/
│   ├── worklog.md        阶段推进、踩坑清单、关键决策
│   ├── data.md           数据集与口径
│   ├── method.md         方法细节
│   ├── results.md        完整指标与结论
│   ├── sim.md            物理仿真：谱系、快充、ΔQ 归因、负结果
│   ├── audit.md          P3 审计 + 三线合并：能被测量 ≠ 能被辨识
│   └── sibling_projects.md  与 battery-lifetime-guide 的分工、口径对照与合并代价
├── 交付物/               Word 项目总结报告 + 离线 HTML 数据看板
├── data/                 数据目录约定（原始数据不入库）
├── figures/              结果图（README.md 为图库索引）
├── reports/              指标表、小结
├── reference/            官方 LoadData.m（仅作口径对照）
├── requirements.txt      核心依赖
├── requirements-full.txt 含 torch / streamlit / pybamm
└── environment.yml       conda 环境
```

---

## 八、已知局限

- **CALCE 缺席**：官网需签署许可，本项目未纳入。ECM 用 NASA、RUL 用 MIT-Stanford，两条线各自有独立数据支撑。
- **深度模型欠拟合**：1D CNN 在 41 颗训练电池下误差 43%，这是小样本的固有限制，已用 5 折 CV 对照说明，不做粉饰。
- **单批次 ECM**：ECM 只在 NASA 4 颗电池上做同工况复测，未做跨温度、跨倍率验证。
- **B0007 超标**：前 20 圈容量再生污染 OCV，复测 RMSE 3.29%，如实保留。
- **学习曲线未饱和**：训练电池从 10 增到 41，误差从约 13% 降到约 8.7%，增加电池数仍有收益。
- **仿真参数未标定到真实寿命量级**：析锂机制的 30 圈容量损失达 35%，只能比较协议间的相对排序，绝对寿命无意义。
- **仿真的化学体系与实测不完全一致**：ΔQ 归因用 LFP（Prada2013），产热与析锂用 NMC（Chen2020 / OKane2022），跨体系只能比形态不能比数值。
- **充电协议的名义时长 ≠ 实际轨迹**：恒流段被 4.2 V 上限截断，`1C(80%)-3C(20%)` 里的 3C 段一步也没走成，整条协议退化为 1C 恒流。7 条协议的 30 圈容量损失只差 1.7 个百分点（离散度 0.05，实测 2.61），「协议 → 寿命」这条链条在本仿真里没有复现出来。
- **P3 的辨识结果不能当参数用**：它报的 R0（65–86 mΩ）里 63% 是 18.7 s 采样间隔的伪影，r_dc 由 OCV 先验决定，τ2 短于观测窗口。审计与修复建议见 [docs/audit.md](docs/audit.md)。

---

## 九、参考与致谢

- Severson, K. A. et al. *Data-driven prediction of battery cycle life before capacity degradation.* **Nature Energy** 4, 383–391 (2019).
- NASA PCoE. *Battery Data Set*, NASA Ames Prognostics Center of Excellence.
- MIT-Stanford. *Battery Cycling Dataset* (<https://data.matr.io/1/>).
- `reference/LoadData.m` 版权归 MIT-Stanford 数据集原作者，此处仅用于复刻数据筛选口径。
- [battery-lifetime-guide](https://github.com/ilo-love7/battery-lifetime-guide)：同一数据集的另一份独立复现，本仓库的 EN 结果与其对照（见 `docs/sibling_projects.md`）。

## 兄弟仓库：同一领域的另一份复现

[battery-lifetime-guide](https://github.com/ilo-love7/battery-lifetime-guide) 用同一批公开数据（NASA PCoE + MIT-Stanford）
复现同一篇 Severson 2019，形态是**带「事故现场」的学习攻略**（step1→6 线性脚本 + FAQ + 面试卡）；
本仓库形态是**自洽的证据链**（ECM 辨识 + PyBaMM 仿真 + 第三方辨识审计）。两者**有意保持为两个仓库**，
分工、口径对照与「为什么不合并」见 [docs/sibling_projects.md](docs/sibling_projects.md)。

> 两边的随机森林在 60 维口径下接近（8.12% vs 8.2%）；ElasticNet 的差距（8.33% vs 5.9%）已经受控实验分解：
> l1_ratio 是否参与搜索只值 0.14 pp，主因是两条管线的 paper6 特征数值定义差异——详见 docs/results.md 2.2。

## License

[MIT](LICENSE)　·　版本记录见 [CHANGELOG.md](CHANGELOG.md)　·　引用方式见 [CITATION.cff](CITATION.cff)（GitHub 页面右侧 "Cite this repository"）
