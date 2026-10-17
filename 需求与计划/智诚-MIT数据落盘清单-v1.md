# MIT（Severson 2019）数据落盘清单 v1.0

> 编制：智诚-科研导师 ｜ 时间：2026-10-10 20:10 ｜ 状态：待 002 号核查
> 校验口径：四要素（文件名 + 字节数 + cell 数 + 抽样解析校验 + 路径），全部为本机实测

---

## 一、落盘总览

| 项目 | 实测结果 |
|---|---|
| 数据文件 | 3 个 batch `.mat`（MATLAB v7.3 / HDF5 格式） |
| 落盘总字节数 | 8,269,341,808 字节（≈ 7.70 GiB） |
| 实测 cell 总数 | 140 |
| 本机路径 | `/home/z/.cache/kagglehub/datasets/rickandjoe/mit-battery-degradation-dataset/versions/1/` |
| 机器可读明细 | `mit_inventory.json`（随本清单同目录交付） |

## 二、分文件四要素明细

| # | 文件名 | 字节数 | cell 数 | 抽样解析校验 |
|---|---|---|---|---|
| 1 | 2017-05-12_batchdata_updated_struct_errorcorrect.mat | 3,025,320,241 | 46 | summary 含 1189 圈；末圈 QDischarge=1.0262 Ah；cycle1 V 数组可读（首索引存在 0 填充，属该文件已知偏移） |
| 2 | 2017-06-30_batchdata_updated_struct_errorcorrect.mat | 2,007,331,155 | 48 | summary 326 圈；首圈 QDischarge=1.0681 Ah → 末圈 0.8265 Ah（退化趋势合理）；cycle1 V 均值 3.145 V |
| 3 | 2018-04-12_batchdata_updated_struct_errorcorrect.mat | 3,236,690,412 | 46 | summary 1008 圈；首圈 1.0666 Ah → 末圈 0.8804 Ah；cycle1 V 均值 3.127 V |

**抽样口径**：每文件取 cell 1，解析 summary（QDischarge 首末圈容量）+ cycle 1 放电 V 数组（点数/均值）。三文件均可正常按 HDF5 引用链解引用，字段 I/Qc/Qd/Qdlin/T/Tdlin/V/t/discharge_dQdV 齐全。

## 三、数据来源与溯源说明（002 号核查重点）

1. **官方源现状**：`data.matr.io/1` 旧版平台 API 已失效（所有 `/api/v1/file/*` 路径返回 301 至前端页面，四个 batch 直链均不可用；本机 HEAD 实测确认）。TRI 门户已改版为静态站点，Severson 数据集页面仍指向旧 SPA 入口。
2. **本清单数据来源**：Kaggle 镜像 `rickandjoe/mit-battery-degradation-dataset`（版本 1，下载经 MD5 校验通过）。文件名与官方 batch 命名一致（`*_batchdata_updated_struct_errorcorrect.mat`），属社区广泛使用的勘误整合版。
3. **与论文口径的差异（需核查确认）**：
   - 本镜像**缺少 batch 2017-05-09**；
   - 实测 cell 数 140，与论文 124 cell 口径不一致（差异可能来自 errorcorrect 版包含论文剔除的电芯，或官方批次划分口径不同）；
   - **影响评估**：Severson 复现基线（前 100 圈特征 → RUL 预测）主要使用 124 cell 的 train/test 划分，缺 2017-05-09 batch 会影响复现覆盖度。若 002 核查判定不可接受，替代路径：Kaggle 补下该 batch 镜像，或向博远求助检索官方替代渠道。
4. **解析技术要点**：文件为 MATLAB v7.3（HDF5），必须用 `h5py`（`scipy.io.loadmat` 直接报 NotImplementedError）；H5MAT 结构数组经引用链解引用（`h5py.ref_dtype`），索引存在 0/1 偏移，属已知特性非数据损坏。

## 四、结论与建议

- **落盘完整性**：3 个文件字节完整、MD5 校验通过（下载器内建）、抽样解析全部成功 → **可作为阶段 B（RUL 建模）的工作数据底座**。
- **遗留风险一项**：缺 batch 2017-05-09 及 cell 数与论文口径差异，请 002 核查给出"可用/需补源"结论。
- NASA / CALCE 两套数据不受影响，此前预研结论（《数据格式预研与环境配置 v1》）仍然有效；NASA 的 loadmat 路径与 MIT 的 h5py 路径需在阶段 0 脚本中分开实现。
