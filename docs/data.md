# 数据说明

本项目使用两个公开数据集，分别支撑两条主线。原始数据不入库，按 [`../data/README.md`](../data/README.md) 自行下载。

| 数据集 | 用途 | 规模 | 格式 |
|---|---|---|---|
| NASA PCoE Battery Data Set | ECM 参数辨识 | 34 颗 18650，7,565 条循环记录 | MATLAB v7（`scipy.io.loadmat`） |
| MIT-Stanford（Severson 2019） | RUL 寿命预测 | 124 颗 A123 LFP，100,501 条循环记录 | MATLAB v7.3 / HDF5（`h5py`） |

---

## 一、NASA PCoE

**来源**：NASA Ames Prognostics Center of Excellence
**电池**：B0005 / B0006 / B0007 / B0018（本项目使用）及其余 30 颗（仅统计，不参与建模）
**工况**：室温恒流充电（CC-CV 4.2 V）→ 恒流放电（约 2 A）至 2.7 V / 2.5 V / 2.2 V

**文件结构**（以 `B0005.mat` 为例）

```
mat["B0005"]                     # battery struct
└── cycle                        # 1×N struct array
    ├── type                     # 'charge' | 'discharge' | 'impedance'
    ├── ambient_temperature      # °C
    ├── time                     # 采样时刻（s）
    └── data
        ├── Voltage_measured     # V
        ├── Current_measured     # A
        ├── Temperature_measured # °C
        ├── Current_load / Voltage_load
        └── Capacity             # 放电容量（Ah），仅放电圈有
```

**已知坑**：部分放电圈的 `Capacity` 字段为空数组。`src/parsers/nasa.py` 会回退到电流积分（`trapz`）计算容量，而不是直接抛 `IndexError`。

**本项目用法**：取 B0005/B0006/B0007/B0018 各前若干圈做 ECM 辨识；第 1 次放电辨识参数，第 3 次放电做同工况复测。

---

## 二、MIT-Stanford（Severson et al. 2019）

**来源**：<https://data.matr.io/1/>（论文配套数据）
**电芯**：A123 Systems APR18650M1A，标称 1.1 Ah，LFP/石墨
**寿命定义**：容量衰减到 80%（即 0.88 Ah）时的循环数，范围 **148–2237 圈**，中位 **736.5 圈**

**文件结构**（v7.3，必须用 `h5py`）

```
/batch
  ├── summary                    # 每颗电池一行的汇总表（对象引用数组）
  │   ├── cycle_life             # 官方寿命标签
  │   ├── policy_readable        # 充电策略描述
  │   ├── QDischarge / QCharge   # 逐循环容量（Ah）
  │   ├── IR                     # 内阻（Ω）
  │   ├── Tmax / Tavg / Tmin     # 温度
  │   └── chargetime             # 充电时长
  └── cycles                     # 每颗电池的原始曲线（对象引用 → 逐循环 group）
```

`summary` 与 `cycles` 都是 HDF5 对象引用数组，必须逐条 `f[ref][()]` 解引用，直接索引会抛
`TypeError: Accessing a group is done with bytes or str`。

### 2.1 官方 124 颗口径

论文只用 124 颗。筛选顺序完整复刻官方 `LoadData.m`（见 `reference/LoadData.m`）：

| batch | 处理 | 保留 |
|---|---|---|
| batch1 | 46 颗，先把 batch2 的 5 颗续测电池拼接回来，再剔除 5 颗未跑完的电池 | 41 |
| batch2 | 5 颗续测（batch1 电池的后半程）拼回 batch1 后，其余保留 | 43 |
| batch3 | 删第 38 颗（channel 46 采集异常）→ 删末端容量 > 0.885 Ah → 删噪声电池 [3, 40, 41] | 40 |
| **合计** | | **124** |

划分与 `LoadData.m` 一致：

| split | 数量 | 来源 | 用途 |
|---|---|---|---|
| `train` | 41 | batch1 | 训练 |
| `test` | 43 | batch2 | 主测试 |
| `secondary_test` | 40 | batch3（另一批实验条件） | 跨批次泛化测试 |

> 为什么不硬编码 barcode：一旦官方口径调整，硬编码编号会静默失效；按规则重算才能保证可比。

### 2.2 寿命标签

```python
def cycle_life_from_qd(qd, eol=0.88):
    qd = np.asarray(qd, dtype=float)
    valid = np.isfinite(qd) & (qd > 0)          # 关键：跳过化成圈
    below = np.where(valid & (qd < eol))[0]
    if below.size == 0:
        return float(len(qd) + 1)
    return float(below[0] + 1)
```

**第 1 圈是化成圈，`QDischarge` 记为 0**。若不跳过，`0 < 0.88` 恒成立，所有电池都会被判成寿命 1 圈。
加上 `qd > 0` 过滤后，重算标签与文件自带 `cycle_life` 逐颗比对，**最大偏差 0 圈**（124/124 一致）。

---

## 三、统一产物

`scripts/phase_b_build_dataset.py` 把两个数据集归一化为 4 个文件（`data/processed/`，已 git 忽略）。

### `mit_cells.parquet`（124 行）

| 列 | 说明 |
|---|---|
| `cell_id` | 电池编号，如 `b1c0` |
| `batch` / `split` | 1/2/3；train/test/secondary_test |
| `cycle_life` | 寿命标签（圈） |
| `n_cycles` | summary 长度 |
| `q0` | 首圈容量 |
| `policy_0..14` | 15 维充电策略参数 |

### `mit_cycles.parquet`（100,501 行）

逐循环 `QDischarge` / `QCharge` / `IR` / `Tmax` / `Tavg` / `Tmin` / `chargetime` / `SOH`。

### `mit_dq.npz`

| 键 | 形状 | 内容 |
|---|---|---|
| `dq` | `(124, 100, 300)` | **滞回**曲线立方体：Qd(V) − Qc(V)，第 2–101 圈 × 2.0–3.5 V 共 300 点，单位 Ah |
| `v_grid` | `(300,)` | 电压栅格 |
| `cell_ids` | `(124,)` | 与 `dq` 第一维对应 |
| `cycles` | `(124, 100)` | 每个窗口对应的实际循环号 |

### `mit_qd.npz`

结构与 `mit_dq.npz` 完全一致，但存的是**放电容量曲线本身**（自 3.5 V 高压端起算），来源是数据集自带的 `Qdlin` 字段重采样。

论文口径的 ΔQ(V) = Q₁₀₀(V) − Q₁₀(V) 由这个文件计算；`mit_dq.npz` 只用于滞回口径的对照特征。**两者不可互换**——滞回口径是同一循环内充放电相减，论文口径是跨循环同方向相减，量级差两个数量级。

### `nasa_cycles.parquet` / `nasa_cells.parquet`

7,565 行逐循环记录（放电 2,794 次）与 34 行电池汇总（首末容量、EOL 圈）。

---

## 四、数据质量处理

| 问题 | 影响面 | 处理 |
|---|---|---|
| MIT 第 1 圈为化成圈（QDischarge=0、曲线仅 2 点） | 全部 124 颗 | 特征窗口从第 2 圈起 |
| batch2 充电起点约 2.47 V（上一圈放到 2.0 V 后静置回升） | 43 颗 | ΔQ 低压端用端点值常值外推 |
| ΔQ 覆盖不足（充电起点 > 2.6 V 或未到 3.45 V） | 12,400 个循环中 1 个 | 该循环置 NaN，不参与统计 |
| NASA 部分放电圈 `Capacity` 为空 | 少量 | 回退电流积分 |
| 极端离群预测 | 少量 | 预测值截断到 [50, 3000] 圈，仅用于绘图与指标计算 |

---

## 五、完整性校验

`data/processed/mit_md5.json` 记录了原始文件 md5，可用于确认下载一致：

| 文件 | 字节数 | MD5 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3,025,320,241 | `1d30ad79d3980c2b2c6cb47783ba27eb` |
| `2017-06-30_batchdata_updated_struct_errorcorrect.mat` | 2,007,331,155 | `f0430a8506ed9e4e326c8579d6be58b4` |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3,236,690,412 | `a57522980f1bd823c9fe90b142e5b8b7` |

```bash
# Windows
certutil -hashfile "2017-05-12_batchdata_updated_struct_errorcorrect.mat" MD5
# macOS / Linux
md5sum 2017-05-12_batchdata_updated_struct_errorcorrect.mat
```

> 注：官方 `data.matr.io` 目前 301 跳转失效，可从 HuggingFace 镜像 `bsebench-org/severson-2019-raw` 获取同一批文件，md5 与上表一致。
