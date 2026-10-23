# Changelog

本文件按 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式记录本项目的版本变动。
详细的阶段推进、踩坑清单与关键决策见 [`docs/worklog.md`](docs/worklog.md)。

## [1.1.0] - 2026-10-23

把 ECM 拟合的**数值稳定性 / 初值敏感性 / 参数可辨识性**做成离线回归测试 + 已入库基线，让「参数漂移超阈值即失败」成为 CI 里的事实；同时补齐 Windows CI leg 与两处 README 板块。

### Added

- **`src/ecm/stability.py`（422 行，新增）**：ECM 拟合的离线诊断层（只用合成数据 + 仓内已提交产物，**不需要** NASA/MIT 原始数据集）。
  - `classify_fit()` 把一次 `fit_rc` 结果分类：`ok` / `underdetermined`（样本数 ≤ 自由参数个数）/ `no-excitation`（I≡0 或 V 无变化）/ `bad-fit`（残差 RMS 超过量程 5%）/ `bound-hit` / `solver-failure`。存在的理由：求解器对**没有信息量**的输入仍会报 `status=1`（"gtol satisfied"）——空数组、零激励、n=5（恰好等于参数个数）实测都会「收敛」，这类静默成功必须被单独标出来，而不是当成拟合成功。
  - `rel_drift()` 相对参数漂移、`branch_set()` / `branch_sets_allclose()` 无序列支路集合、`max_abs_voltage_diff()` 预测一致性、`solver_failure_reason()` 把「求解失败」与「非法输入」分成两类。
  - 阈值依据写在常量注释里（不是随手拍）：`PARAM_REL_DRIFT_MAX = 1e-6`（合成拟合无噪声、求解器确定性，`xtol=ftol=gtol=1e-12`、double eps = 2.2e-16；1e-6 比 eps 高约 10 个数量级，留出 CI(Linux) 与本地 Windows 之间 scipy/BLAS 的差异余量，同时远低于对外参数的 4–5 位有效数字——1e-6 相对 = 1e-4 mΩ）；`PRED_MAX_ABS_V = 1e-9`（实测两解预测差约 1e-13 V）；`RESID_FRACTION_OF_SPAN_MAX = 0.05`（与项目自身「复测 RMSE ≤ 量程 2%」验收口径同源，此处放松到 5%，只用于区分「解释不了数据」与「正常拟合」，不用于评判精度）。
- **`tests/test_ecm_stability.py`（233 行，25 个用例项）**：初值敏感性（7 组初值，含贴上下界、量级错、`τ1 > τ2` 的坏初值）、数值稳定性（畸形输入的报错语义、标量恒流、零激励、欠定、异常值、单位量级错）、可辨识性（合成真值恢复 + 支路置换对称）。**测试总数 9 → 34**。
- **`tests/regression_baseline.json`（8137 B，已入库基线）**：测试**读取**基线并断言漂移在阈值内（比较逻辑 `compare_to_baseline()`）。同时把 `reports/phaseA2_ecm_results.csv` 的真实数据数字快照进来（NASA 4 颗：R0 105.05–107.58 mΩ、复测 RMSE 中位 **1.6052%** 原始 / **1.6289%** 标定后），使真实数据侧的漂移也能离线核对。
- **`scripts/phase_h_ecm_stability.py`（21 行，新增）**：基线入口——直接跑即重新生成基线，`--check` 只比对、漂移超阈值退出码 1。**不触碰任何既有脚本的产物路径**（`reports/`、`figures/`、`data/processed/` 均未变）。
- **CI 新增 `windows-latest` leg（`syntax & imports (windows)`）**：与既有 ubuntu job 同样步骤；既有 `check` job 的名字与内容一个字未动（不改既有 required check 的名字）。
- README 新增 `## What you can do with this toolkit` / `## 用途 / 你能用它做什么`（用途-入口表 + 典型用途 + 可复现性），以及文件末尾 `## Related repositories` / `## 关联仓库`（7 仓库互链表）。

### Changed

- **`fit_rc()` 增加可选初值 `p0`**：默认仍取 `_params(order)`（`r0_fixed` 时取历史默认），**不传时行为与旧版逐位一致**；传入初值才使「初值敏感性」可测。
- **`fit_rc()` 增加输入校验**（`_validate_1d()` / `_as_p0()`）：空数组 / NaN / inf / 长度不匹配 / 非 1-D / `order` 非 1|2 / `p0` 长度错 → 抛 `ValueError`，消息明确写 "invalid input, not a solver failure"，与「求解器未收敛」（`res.status <= 0`，由 `solver_failure_reason()` 翻译）分属两类。改之前同一批输入是静默垃圾：空数组返回初值 + `status=1`（"converged"）+ `rmse = nan`。

### Fixed

- **新增输入校验曾拒绝「标量恒流」，与既有用法冲突**：`tests/test_core.py` 的合成放电把 `i_mag` 写成标量（`CAP_AH * 3600.0 / 3600.0`，即常数 1C），第一版校验一律拒绝 0-D 输入，会把这种既有公开用法误判成畸形输入。现改为 `_validate_1d("i_mag", ..., broadcast_to=t.shape[0])`：**标量恒流按时间轴广播**为常数数组；`t` / `V_meas` / `soc` 仍不接受标量。回归用例 `test_scalar_constant_current_is_still_accepted` 守着这条。

### 如实记录的边界（不做粉饰）

- **`fit_rc` 的可辨识性边界**：2-RC 的参数**向量**只在「支路置换」意义下可辨识——`V = OCV − I[R0 + R1(1−e^(−t/τ1)) + R2(1−e^(−t/τ2))]` 对 `(R1,τ1) ↔ (R2,τ2)` 不变。实测从 `τ1 > τ2` 的初值出发会收敛到**置换后的标注** `[R0, R2_true, τ2_true, R1_true, τ1_true]`，残差 RMS **7.1e-17 V**，与真值解的 **5.9e-17 V** 同量级——两者在电压上完全不可区分。**单条恒流放电无法恢复带标签的 R1/R2 与 τ1/τ2**，能恢复的只有无序列 `{(R1,τ1),(R2,τ2)}`。这是模型结构决定的，不靠"给拟合加 τ1<τ2 约束"来假装解决。回归用例 `test_swapped_branch_order_is_a_permutation_not_a_wrong_answer` 固化这条对称性。
- 基线记录的是**历史上各初值实际收敛到的标注**；任何标注变化（例如"修掉"上述对称性）都会让漂移门禁变红，必须作为一次有意识的决定连同基线一起复核，而不是静默通过。
- 过程中如实记下的一处自伤：`truth_recovery` 的支路比较最初用浮点逐位 `==`，因末位 1e-16 差异把默认初值的成功恢复误判成 `False`，已改为容差比较 `branch_sets_allclose()`。

---

## [Unreleased]

### 已知分歧与待办

- **ElasticNet 六特征与另一份独立复现（battery-lifetime-guide）差 2.4 pp**（本仓存档 8.33% vs 对方 5.9%）。
  **2026-10-18 受控实验（`scripts/verify_en_l1ratio.py`）已修正早期归因**：早期认为差距全在
  `l1_ratio` 是否参与搜索，实验证明那只是小项——同一份特征快照上固定 0.5 与网格搜索只差
  0.14 pp（6.18% → 6.04%）。**差距主因是两条管线的 paper6 特征数值定义差异**：同一旧配置换特征来源
  即从 6.18% 跳到 8.33%，且误差结构完全不同（mean 34.8% / R² −12.9 vs 23.3% / 0.62）。
  全量重跑（数据已就绪，8.27 GB 落在 `D:\项目\02_数据集\01_国际公开集\MIT_MATR-1\原始数据\`，仓库内 `data/raw/mit` 是其 junction）预期落在 ~6.2% 一线；剩余差距需特征级 diff 定位。
- 两仓主测试 43 颗成员已核对**完全相同**（官方 test_ind 混合划分逐颗对上），寿命标签仅 3 颗
  续跑电池差 1 圈（可忽略）；「RF 8.12% vs 8.2% 完全一致」是不同特征集的巧合接近，
  同口径有效对照为 9.38%（本仓 paper6 RF）vs 8.2%，同样差约 1.2 pp，与 EN 同向。
- 新增 `scripts/verify_en_l1ratio.py`：跨仓受控实验（本仓 124 颗名单与 split × guide 六特征快照，
  三配置 × 两测试集），产物 `reports/en_l1ratio_verification.csv`、`reports/en_l1ratio_predictions.csv`。
- 新增 `docs/sibling_projects.md`：两仓分工、口径对照与合并代价。

### 变更

- `docs/results.md` 第 2.2 节、`docs/sibling_projects.md` 第二节、`README.md` 兄弟仓库段：
  把「差距全在 l1_ratio 搜索空间」「RF 一致证明四条链都对」的旧归因，替换为受控实验的三段式结论
  （l1_ratio 是小项、特征管线是主项、测试集成员已核对一致）。
- `scripts/phase_c_rul.py`：`tune_elasticnet()` 的文档字符串同步修正，注明其方法论正确但
  不是跨仓差距的解释（2026-10-18）。
- 新增 `scripts/stats_significance.py`：对已有逐颗预测补统计口径——bootstrap 1000 次
  95% CI + Wilcoxon 配对检验（判定：p<0.05 且中位差距 ≥1 pp 为显著）。产物
  `reports/phaseC_significance.csv`（21 行）。核心发现：n=43 下 GBM 6.20% vs EN 7.77%
  **不显著**（p=0.93，CI 宽 5–9 pp），此前模型排名只是描述性的；RF 显著劣于 GBM
  （p=0.006）、CNN 断崖失败显著（p=1.8e-07）、batch3 上 RF 显著优于 GBM（p=0.0012）；
  EN l1_ratio 改进主测试上不显著（p=0.92）。
- `docs/results.md` 新增 2.6 节记录上述显著性分析；`README.md` 主协议表下补统计口径提示。
- 新增 `scripts/lobo_analysis.py` + `reports/lobo_folds.csv`：三折纯批次 LOBO（每折独立内层调参）× 两模型版本 + 每折 6 特征 KS 漂移检验（严谨 §2.1/§4）。核心发现：batch2 折最难（中位 24.4%，全部 6 特征分布显著偏移，log_var D=0.77）且与 b1 折 CI 不重叠——跨批次难度差异是统计事实；b2 折新配置自己选回 l1_ratio=0.5，l1_ratio 在三折上没有一次超过 0.3 pp 的影响；与 guide 轮换表方向一致（b2 最难）但幅度不同（特征/标签版本差异）。写入 results.md 2.7。
- 新增 `docs/applicability.md`（适用边界说明书，严谨方案 D3 旗舰交付物）：7 节——模型概览与基准性能、输入要求硬性前提（6 条）、性能承诺 🟢🟡🔴 分级表（11 行）+ 5 条明确不适用场景、已知失效模式 F1–F6（现象/归因/机理）、使用前自检清单（7 项）、边界外替代建议、局限性声明（7 条）。原则：每条边界只引用已验证数字，噪声鲁棒性与温度两类未测定项明确标注「未测定」，不做推测填空。

### 变更（2026-10-18，阶段 G 补做此前数据阻塞的实验）

- 新增 `scripts/phase_g_stress_tests.py` + `reports/phaseG_window.csv` / `phaseG_noise.csv` / `phaseG_results.json` + `figures/fig_g_01_window.png` / `fig_g_02_noise.png`：MIT 原始数据（8.27 GB，junction 接入）就位后，补做两个此前被阻塞的实验（复用 `phase_b` 立方体与 `build_features`，在 ΔQ(V) 曲线层注入噪声的一阶代理）。
  - **早期预测窗口 N<100**（§2.8.1）：同分布主测试前 20 圈 GBM 7.85%（vs 满窗口 100 圈 6.20%）几乎免费；但跨批次二级测试前 20 圈 GBM 36.30%（vs 16.37%）代价大——早测只建议同分布内做。N<100 时论文 `xq_*` 特征（依赖第 100 圈）正确地不可用，特征集退化为 57 维。
  - **噪声注入压力测试**（§2.8.2）：σ≤0.003 Ah（≈真实传感器量级，0.8× 信号σ）主测试误差几乎不变（GBM 6.20%→6.36%）；即便极端 σ=0.03 Ah（≈8× 信号σ）GBM 仍 <10%、RF<11%。测量精度劣化不构成可用性问题。
  - 数字为单官方划分点估计中位 APE（未重算 bootstrap CI），用作量级结论。
- `docs/results.md` 新增 §2.8（早期窗口 + 噪声鲁棒性两节 + 两张表 + 产物清单）。
- `docs/applicability.md`：§2.1（循环数）、§2.2（容量测量精度）由「未测定」改为实测结论并指向 §2.8（保留旧「未测定」曾被标注的事实）；§3 性能承诺表新增 4 行（早期窗口 ×2、噪声 ×2）；§6 早测建议、§7.3 测量精度局限同步闭环。
- `data/processed/phaseB_mit_stats.json`：重跑 `phase_b_build_dataset.py mit`（junction 接入后）刷新耗时统计。

---

## [1.0.0] - 2026-10-17

一个项目的三个阶段：**P1** ECM 辨识与 RUL 预测（复现）→ **P2** PyBaMM 物理仿真（复现 + 创新）→ **P3** 第三方辨识的审计（把前两条线接上）。

### P1 — ECM 辨识与 RUL 预测

- NASA PCoE 四颗电池（B0005/6/7/18）上辨识 Thevenin 二阶模型，同工况复测电压 RMSE 中位 **1.61%**（0.96 / 1.49 / 1.72 / 3.29 %）。
- MIT-Stanford 124 颗电池、仅用前 100 圈数据预测循环寿命（148–2237 圈）：
  - 主测试集中位相对误差 **6.20%**（60 维特征 + GBM，R²=0.93），优于论文同划分的 9.1%；
  - 论文口径 6 特征下 **8.03%**（batch3）/ 8.58%（主测试），与论文 9.1% 一致或更好；
  - 全量 5 折 CV 中位相对误差 7.37% ± 3.25（RF）。
- 特征口径消融证明：60 维里混进的滞回特征导致跨批次退化（6.20% → 16.37%），收紧到论文口径后几乎不退化。
- 1D CNN 对照：41 颗训练 32.7% / 99 颗训练 12.1%，说明深度模型受限于电池数而非结构。

### P2 — PyBaMM 物理仿真

- SPM / DFN / Thevenin(反演) 三档模型对比：相对 DFN 的 RMSE 5.16 / — / 17.19 mV，**等效电路的误差在参数不在结构**。
- 七条快充协议仿真：t80 从 45.2 min 压到 16.2 min（2.8 倍），总时长只从 85.4 压到 56.8 min（1.5 倍），温升 8.4 → 29.9 K。
- 退化模式注入把 ΔQ(V) 归因到**锂库存损失（r=+0.64）与负极活性材料损失（r=+0.58）**，对正极活性损失（r=−0.07）不敏感，与 MIT 实测形态吻合。
- 两个如实记录的负结果：① 把仿真分解出的模式系数作为新特征加入 RUL 模型，四组对照全部持平或变差；② 七条协议的 30 圈容量损失只差 1.7 个百分点（离散度 0.05，实测 2.61），「协议 → 寿命」未复现——电压上限把激进协议的恒流段截断，名义协议与实际轨迹是两件事。

### P3 — 第三方辨识的审计与三线合并

对另一个 AI 独立完成的 ECM 辨识链逐条检验，三个核心量全部不可辨识：

| 问题 | 证据 |
|---|---|
| 判据自证 | OCV 由同一套参数反推，参数 ×2 / ×0.5 后复算 RMSE 仍 ≤ 0.31 mV（仅换电池才 8.7–12.0 mV） |
| 参数由先验决定 | r_dc = (OCV50_prior − V(SOC=0.5))/I，先验 ±0.1 V ⇒ r_dc ±50 mΩ，三颗电池曲线完全重合 |
| τ2 由窗口决定 | B0006 松弛窗仅 77.7 s 却报 τ2 = 95 s；B0005 截到 45% 时 162.4 → 94.3 s |

两条桥把三条线接上：

- **P3 → P2**：把 P3 流程套到 DFN 合成曲线上只改采样间隔，R0 从 38.6 mΩ 膨胀到 208.1 mΩ（真值恒定），R1/R2 变负；P3 报的阶跃内阻 105–107 mΩ 与 dt = 18.7 s 下的 105.0 mΩ 逐位重合 ⇒ **63% 是采样伪影**。
- **P2 → P1**：三种退化模式下内阻变化全在 ±7% 内且方向不一致，ΔQ 深度跨模式差 25 倍 ⇒ **这就是 P1 用 ΔQ(V) 而不用内阻的物理原因**。

合并结论：**能被测量 ≠ 能被辨识**。

### 工程与仓库

- 20 条踩坑记录（`docs/worklog.md`），其中两条是实质 bug：
  - PyBaMM 24.x 放电电流取负、26.x 取正，旧代码用电流符号判充放电会静默对调（t80 失效）→ 改为按电压走向判断；
  - `6C(60%)` 在 5 Ah NMC 上把端电压充到 4.68 V 触发电压上限事件 → 所有恒流段带 `or until V_max`。
- 26 张成果图 + 图库索引（`figures/README.md`，每张图标注生成脚本与对应结论）。
- 统一数据层（NASA + MIT → Parquet/NPZ），复刻官方 `LoadData.m` 的 124 颗筛选口径。
- Makefile 一键复现（`make all`），CI 做语法编译与 import 冒烟测试。
- 交付：Word 项目总结报告（17 图 / 13 表）、离线 HTML 数据看板、交互式小工具。

[1.1.0]: https://github.com/ilo-love7/battery-ecm-rul/releases/tag/v1.1.0
[1.0.0]: https://github.com/ilo-love7/battery-ecm-rul/releases/tag/v1.0.0
