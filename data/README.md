# data/

原始数据体积大（NASA ≈ 200 MB，MIT-Stanford ≈ 8.3 GB），不纳入版本控制。
本目录只保留**目录约定**与**处理产物的校验记录**，数据需按下方说明自行下载。

## 目录约定

```
data/
├── raw/                      # 原始数据（git 忽略）
│   ├── nasa/                 # NASA PCoE：B0005.mat, B0006.mat, ...
│   └── mit/                  # MIT-Stanford：2017-05-12_batchdata_updated_struct_errorcorrect.mat
│                             #               2017-06-30_batchdata_updated_struct_errorcorrect.mat
│                             #               2018-04-12_batchdata_updated_struct_errorcorrect.mat
└── processed/                # 预处理产物（git 忽略，可由脚本重建）
    ├── mit_cells.parquet     # 124 颗电池 × 元信息 / 寿命标签 / 充电策略
    ├── mit_cycles.parquet    # 100,501 条逐循环记录
    ├── mit_dq.npz            # 滞回曲线立方体 [124, 100, 300]（Qd − Qc）
    ├── mit_qd.npz            # 放电容量曲线立方体 [124, 100, 300]（论文 ΔQ 用）
    ├── nasa_cells.parquet
    ├── nasa_cycles.parquet
    ├── mit_md5.json          # 原始文件 md5（入库，便于校验）
    └── phaseB_*_stats.json   # 数据质量统计（入库）
```

## 1. NASA PCoE — Battery Data Set

- 页面：<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- 下载 `Battery_Data_Set.zip`，解压后把 `1. BatteryAgingARC-FY08Q4/` 下的
  `B0005.mat`、`B0006.mat`、`B0007.mat`、`B0018.mat` 放入 `data/raw/nasa/`。
- 格式：MATLAB v7，`scipy.io.loadmat` 可直接读取。

## 2. MIT-Stanford — Severson et al. 2019

- 页面：<https://data.matr.io/1/>（Nature Energy 论文配套数据，公开下载）
- 需要 3 个 batch 文件，放入 `data/raw/mit/`：

  | 文件 | 说明 |
  |---|---|
  | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | batch1 |
  | `2017-06-30_batchdata_updated_struct_errorcorrect.mat` | batch2（含 5 颗 batch1 续测电池） |
  | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | batch3（另一批实验条件） |

- 格式：MATLAB **v7.3（HDF5）**，必须用 `h5py` 逐引用读取，`scipy.io.loadmat` 会失败。
- 官方筛选用 `reference/LoadData.m`，本项目 `src/parsers/mit.py` 完整复刻其处理顺序，
  得到 **124 颗**（train 41 / test 43 / secondary_test 40）。

## 3. CALCE（未使用）

CALCE 数据集需签署许可协议，本项目未纳入；ECM 用 NASA、RUL 用 MIT-Stanford，
两条主线各自有独立数据支撑，不影响结论。

## 指定其它路径

若数据不放在默认位置，可用环境变量覆盖：

```bash
export MIT_DIR=/path/to/mit          # Windows: set MIT_DIR=D:\data\mit
export NASA_DIR=/path/to/nasa
```
