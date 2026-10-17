# 图库索引

所有图片由脚本生成，**不要手工修改**。重新生成见根目录 `Makefile`：`make ecm` / `make rul` / `make sim` / `make audit`。

命名规则：`fig_<阶段>_<内容>.png`，阶段编号与 `scripts/` 前缀一致。

例外：`project_overview.png` 是项目总览图（不按阶段命名），由 `scripts/make_overview.py` 生成，用于 README 与 LEARN.md 头部，展示「数据 → P1 数据线 / P2 物理仿真标尺 / P3 审计线」的三层证据链。

| 前缀 | 阶段 | 脚本 |
|---|---|---|
| `a0` | 数据解析验证 | `validate_nasa.py` |
| `a2` | ECM 参数辨识 | `phase_a2_ecm.py` |
| `c` | RUL 寿命预测 | `phase_c_rul.py` |
| `d` | PyBaMM 物理仿真 | `phase_d1_pybamm_sim.py` / `phase_d2_sim_rul.py` |
| `e` | P3 辨识审计 | `phase_e_p3_audit.py` |
| `f` | 三线桥接 | `phase_f_bridge.py` |

---

## 阶段 A0：数据验证

| 文件 | 内容 | 用途 |
|---|---|---|
| `fig_a0_discharge_vt.png` | NASA B0005 单次放电的电压-时间曲线 | 确认 `.mat`（MATLAB v7）解析与字段对齐正确，是整条链路的第一道自检 |

## 阶段 A2：ECM 参数辨识

| 文件 | 内容 | 对应结论 |
|---|---|---|
| `fig_a2_ocv_soc.png` | 四颗电池的 OCV–SOC 曲线（20 圈平均 + PCHIP） | 充放电平均抵消极化滞回；PCHIP 保单调、不过冲 |
| `fig_a2_fit_B0005.png` | B0005 辨识参数的正向仿真 vs 实测 | 复测 RMSE **1.72%** |
| `fig_a2_fit_B0006.png` | B0006 | **0.96%**（四颗中最好） |
| `fig_a2_fit_B0007.png` | B0007 | **3.29%**（唯一超标，容量再生污染 OCV） |
| `fig_a2_fit_B0018.png` | B0018 | **1.49%** |
| `fig_a2_r_aging.png` | R0 与 R_dc(50%) 随 SOH 的归一化演化 | SOH 100→70：R0 **+5%**、R_dc **+33%**，老化主要恶化极化项 |

## 阶段 C：RUL 寿命预测

| 文件 | 内容 | 对应结论 |
|---|---|---|
| `fig_c_01_life_dist.png` | 训练 41 / 主测试 43 / 二级测试 40 的寿命分布 | 148–2237 圈，中位 736.5 圈，跨度大故用中位误差 |
| `fig_c_02_dq_aging.png` | 三颗不同寿命电池的第 2 / 100 圈曲线 | 长寿命电池曲线几乎不动，短寿命的谷明显变深 |
| `fig_c_03_model_compare.png` | 五模型 + 基线的主测试误差对比 | GBM **6.20%** 最优（R²=0.93） |
| `fig_c_04_pred_scatter.png` | 预测 vs 真实（对数坐标，含 batch3） | CNN 明显向均值收缩；EN 有极端外推点 |
| `fig_c_05_learning_curve.png` | 训练电池数 10→41 与误差的关系 | 30 颗后走平，41 颗接近这批数据的收益边界 |
| `fig_c_06_cv.png` | 全量 124 颗 5 折交叉验证 | RF **7.37%**、GBM 7.46%、CNN 12.11% |
| `fig_c_07_importance.png` | 随机森林 Top-15 特征重要性 | 首位是 `xq_log_mean`（论文 ΔQ 口径） |
| `fig_c_08_ablation.png` | 论文 6 维 vs 全部 60 维，主测试与 batch3 对比 | **跨批次 6 维全面胜出**：EN 6.31% vs 28.64% |

## 阶段 D：PyBaMM 物理仿真（P2）

| 文件 | 内容 | 对应结论 |
|---|---|---|
| `fig_d_01_model_hierarchy.png` | DFN / SPM / Thevenin 三档保真度的 1C 放电曲线，及各自相对 DFN 的电压 RMSE | **结构降阶只花 5–17 mV，参数错配要花 201 mV**；ECM 的问题不在结构在参数 |
| `fig_d_02_rate_capability.png` | 0.5C–4C 放电曲线与容量利用率 | 4C 下可用容量只剩 0.5C 的 **53.3%** |
| `fig_d_03_fastcharge_heat.png` | 7 条快充协议的 t80 / 总时长 / 温升 | t80 从 45.2 压到 16.2 min（**2.8 倍**），总时长只从 85.4 降到 56.8 min（1.5 倍）；`sev_4_8` 温升最高 29.9 K；`place_high` 与 `base_1c` 重合（3C 段被电压上限全部截断） |
| `fig_d_04_degradation_modes.png` | LLI / LAM_pos / LAM_neg 三种退化模式对 ΔQ(V) 的影响，与 MIT 实测形态叠加 | 实测与 **LLI（r=+0.64）、LAM_neg（r=+0.58）** 同向，与 LAM_pos（r=−0.07）无关 |
| `fig_d_05_protocol_life.png` | 纯 SEI vs SEI+析锂两种机制下的协议容量损失，及仿真/实测离散度对照 | 两种机制下协议都几乎无差别（离散度 0.08 / 0.05），比实测的 2.61 小两个数量级——**「协议 → 寿命」没复现出来** |
| `fig_d_06_mode_features.png` | 仿真模式系数加入 P1 后的四组对照 + 124 颗电池的模式分解分布 | **仿真先验没有带来改善**（四种组合全部持平或变差），模式系数标准差仅 0.018 |

## 阶段 E：P3 辨识审计

| 文件 | 内容 | 对应结论 |
|---|---|---|
| `fig_e_01_selfconsistency.png` | 自洽性检验：参数 ×2 / ×0.5 / 换电池三组下的复算 RMSE（上半幅），以及三组参数反推出的 OCV 曲线（下半幅） | 参数翻倍或减半，RMSE 仍 ≤ 0.31 mV；只有换电池才掉到 8.7–12.0 mV → **判据对参数无约束力**。低 SOC 处 OCV 回落 17.6 / 59.5 mV，是 R(SOC) 被误记进 OCV |
| `fig_e_02_prior_sensitivity.png` | 先验 OCV(0.5) 从 3.76 扫到 3.96 V，三颗电池的 r_dc 与 R0 | r_dc 从 115 mΩ 线性涨到 215 mΩ（先验 1 mV → 0.5 mΩ），三条曲线**完全重合** → 参数由先验决定，不是辨识出来的 |
| `fig_e_03_identifiability.png` | τ2 对拟合窗口长度的敏感性（真实数据截断）+ bootstrap 分布 | B0005 截到 45% 时 τ2 从 162 s 掉到 94 s；B0006 窗口仅 77.7 s 却报 95 s → **τ2 是窗口的函数** |

## 阶段 F：三线桥接

| 文件 | 内容 | 对应结论 |
|---|---|---|
| `fig_f_01_sampling_audit.png` | 把 P3 流程套到 DFN 合成曲线上，采样间隔从 0.05 s 加到 18.7 s，看 R0/R1/R2 与 r_dc 怎么变 | R0 从 38.6 膨胀到 **208.1 mΩ**（真值恒为 38.6），R1+R2 变负，r_dc 恒定 → **采样率决定 R0**，P3 的阶跃内阻里 63% 是伪影 |
| `fig_f_02_feature_families.png` | 三种退化模式下，内阻（a）与 ΔQ 深度（b）的敏感度对照 | 内阻变化全在 ±7% 内且方向不一致；ΔQ 跨模式差 25 倍 → **这就是 P1 用 ΔQ 而不用内阻的原因** |

---

## 图与文档的对应关系

| 文档 | 引用图 |
|---|---|
| `README.md` | `fig_a2_fit_B0005`、`fig_c_02_dq_aging`、`fig_c_03_model_compare`、`fig_c_04_pred_scatter`、`fig_c_08_ablation`、`fig_d_01`、`fig_d_03`、`fig_d_04`、`fig_e_03`、`fig_f_01`、`fig_f_02` |
| `docs/results.md` | `fig_a2_fit_B0005`、`fig_a2_r_aging`、`fig_c_01`～`fig_c_08` |
| `docs/method.md` | 无（纯方法说明） |
| `docs/sim.md` | `fig_d_01`～`fig_d_06` |
| `docs/audit.md` | `fig_e_01`～`fig_e_03`、`fig_f_01`～`fig_f_02` |
| `交付物/数据看板.html` | 由 `scripts/build_dashboard.py` 内嵌，不依赖本目录 |
