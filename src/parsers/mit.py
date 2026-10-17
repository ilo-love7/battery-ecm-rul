"""MIT-Stanford (Severson et al. 2019) 数据解析器。

数据为 MATLAB v7.3（HDF5 封装），须用 h5py 读取。

关键点：
1. 每个 batch 文件顶层 `batch` 组下，`cycles` / `summary` / `cycle_life` / `policy`
   都是 (N,1) 的 object reference 数组，需逐元素解引用；
2. `summary` 组内的字段是普通 float 数组 (1,n)，读取很快；`cycles` 组内字段是
   (n,1) 的引用数组，每个引用指向一段曲线（V/I/t/Qc/Qd/T），逐条读取较慢，
   因此只读前 n_curves 圈；
3. 官方口径（LoadData.m）需要做三件事：
   - batch2 的 [8,9,10,16,17]（1-based）是 batch1 前 5 颗电池的续测，要拼接回去；
   - batch3 先删第 38 个（channel 46 采集异常），再删末端容量 > 0.885 的电池，
     再删噪声电池 [3,40,41]；
   - 合并后删除 [9,11,13,14,23]（这 5 颗未跑完），最终 124 颗。
4. 寿命标签：首次 QDischarge < 0.88 Ah 的循环序号（1-based），否则 n_cycles+1。
5. 训练/测试划分完全复刻 LoadData.m：
   test_ind = [1:2:(numBat1+numBat2), 84]（43 颗），train 为其余 41 颗，
   batch3 的 40 颗为 secondary test。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import h5py
import numpy as np

BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

SUMMARY_FIELDS = ["cycle", "QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]
CURVE_FIELDS = ["V", "I", "t", "Qc", "Qd", "T", "Qdlin"]

# 官方拼接长度（batch2 -> batch1 前 5 颗）
ADD_LEN = [661, 980, 1059, 207, 481]
BATCH2_MERGE_IDX = [7, 8, 9, 15, 16]  # 0-based，对应 LoadData.m 的 [8:10,16:17]


@dataclass
class CellMeta:
    """一颗电池的元信息（不含曲线），用于官方口径筛选与划分。"""

    cell_id: str
    batch: int
    file: str
    index: int  # 在所属 batch 文件中的 0-based 序号
    merged: Optional[List[tuple]] = None  # [(batch, index), ...] 续测片段
    cycle_life: float = float("nan")
    n_cycles: int = 0
    policy: Optional[np.ndarray] = None
    summary: Dict[str, np.ndarray] = field(default_factory=dict)
    split: str = ""

    @property
    def segments(self) -> List[tuple]:
        segs = [(self.batch, self.index)]
        if self.merged:
            segs += self.merged
        return segs


def _deref_scalar(f: h5py.File, ref) -> np.ndarray:
    return np.asarray(f[ref][()]).ravel()


def read_summary(f: h5py.File, index: int) -> Dict[str, np.ndarray]:
    """读取第 index 颗电池的 summary（逐循环标量），返回 dict of 1-D array。"""
    batch = f["batch"]
    s = f[batch["summary"][index, 0]]
    out = {}
    for k in SUMMARY_FIELDS:
        if k in s:
            out[k] = np.asarray(s[k][()]).ravel().astype(float)
    return out


def read_cycle_life(f: h5py.File, index: int) -> float:
    batch = f["batch"]
    return float(_deref_scalar(f, batch["cycle_life"][index, 0])[0])


def read_policy(f: h5py.File, index: int) -> np.ndarray:
    batch = f["batch"]
    return _deref_scalar(f, batch["policy"][index, 0]).astype(float)


def read_curves(f: h5py.File, index: int, n_curves: int) -> Dict[str, List[np.ndarray]]:
    """读取第 index 颗电池前 n_curves 圈的曲线（V/I/t/Qc/Qd/T）。"""
    batch = f["batch"]
    g = f[batch["cycles"][index, 0]]
    n = g["V"].shape[0]
    n = min(n, n_curves)
    out: Dict[str, List[np.ndarray]] = {k: [] for k in CURVE_FIELDS}
    for c in range(n):
        for k in CURVE_FIELDS:
            ref = g[k][c, 0]
            out[k].append(np.asarray(f[ref][()]).ravel().astype(np.float32))
    return out


def cycle_life_from_qd(qd: np.ndarray, eol: float = 0.88) -> float:
    """首次 QDischarge < eol 的 1-based 循环号；未达到则 n+1。

    注意：第 1 圈是化成/初始循环，QDischarge 记为 0（无有效放电容量），
    必须从第 2 圈起搜索，否则所有电池都会被判成寿命 = 1。
    """
    qd = np.asarray(qd, dtype=float)
    valid = np.isfinite(qd) & (qd > 0)
    below = np.where(valid & (qd < eol))[0]
    if below.size == 0:
        return float(len(qd) + 1)
    return float(below[0] + 1)


def build_official_cell_list(data_dir: str) -> List[CellMeta]:
    """按 LoadData.m 复刻官方 124 颗电池清单与 train/test 划分。"""
    metas: List[CellMeta] = []
    with h5py.File(os.path.join(data_dir, BATCH_FILES[1]), "r") as f1:
        n1 = f1["batch"]["cycles"].shape[0]
        b1 = []
        for i in range(n1):
            b1.append(
                CellMeta(
                    cell_id=f"b1c{i}",
                    batch=1,
                    file=BATCH_FILES[1],
                    index=i,
                    cycle_life=read_cycle_life(f1, i),
                    policy=read_policy(f1, i),
                    summary=read_summary(f1, i),
                )
            )
    with h5py.File(os.path.join(data_dir, BATCH_FILES[2]), "r") as f2:
        n2 = f2["batch"]["cycles"].shape[0]
        b2_all = []
        for i in range(n2):
            b2_all.append(
                CellMeta(
                    cell_id=f"b2c{i}",
                    batch=2,
                    file=BATCH_FILES[2],
                    index=i,
                    cycle_life=read_cycle_life(f2, i),
                    policy=read_policy(f2, i),
                    summary=read_summary(f2, i),
                )
            )

    # 续测片段拼接到 batch1 前 5 颗
    for k, src in enumerate(BATCH2_MERGE_IDX):
        tgt = b1[k]
        seg = b2_all[src]
        tgt.merged = [(2, src)]
        for key in SUMMARY_FIELDS:
            if key in tgt.summary and key in seg.summary:
                tgt.summary[key] = np.concatenate([tgt.summary[key], seg.summary[key]])
        tgt.cycle_life = cycle_life_from_qd(tgt.summary["QDischarge"])
    b2 = [m for i, m in enumerate(b2_all) if i not in BATCH2_MERGE_IDX]

    with h5py.File(os.path.join(data_dir, BATCH_FILES[3]), "r") as f3:
        n3 = f3["batch"]["cycles"].shape[0]
        b3_all = []
        for i in range(n3):
            b3_all.append(
                CellMeta(
                    cell_id=f"b3c{i}",
                    batch=3,
                    file=BATCH_FILES[3],
                    index=i,
                    cycle_life=read_cycle_life(f3, i),
                    policy=read_policy(f3, i),
                    summary=read_summary(f3, i),
                )
            )
    b3 = [m for i, m in enumerate(b3_all) if i != 37]  # 删 channel 46
    b3 = [m for m in b3 if not (m.summary["QDischarge"][-1] > 0.885)]  # 末端容量异常
    drop = {2, 39, 40}
    b3 = [m for i, m in enumerate(b3) if i not in drop]  # 噪声电池

    combined = b1 + b2 + b3
    keep = [i for i in range(len(combined)) if i not in {8, 10, 12, 13, 22}]
    combined = [combined[i] for i in keep]

    num_b12 = len(combined) - len(b3)  # 84 = batch1(41) + batch2(43)
    for i, m in enumerate(combined):
        m.cycle_life = cycle_life_from_qd(m.summary["QDischarge"])
        m.n_cycles = len(m.summary["QDischarge"])
        if i < num_b12:
            # LoadData.m: test_ind = [1:2:(numBat1+numBat2), 84]（1-based）
            one_based = i + 1
            m.split = "test" if (one_based % 2 == 1 or one_based == num_b12) else "train"
        else:
            m.split = "secondary_test"
    return combined


def iter_cells_with_curves(
    data_dir: str, metas: List[CellMeta], n_curves: int = 150
) -> Iterator[tuple]:
    """按需逐颗读取曲线，产出 (meta, curves)。最多同时打开一个文件。"""
    cache: Dict[int, h5py.File] = {}
    try:
        for m in metas:
            per_seg = []
            for b, idx in m.segments:
                if b not in cache:
                    cache[b] = h5py.File(os.path.join(data_dir, BATCH_FILES[b]), "r")
                per_seg.append(read_curves(cache[b], idx, n_curves))
            merged = {k: [] for k in CURVE_FIELDS}
            for seg in per_seg:
                for k in CURVE_FIELDS:
                    merged[k].extend(seg[k])
            for k in CURVE_FIELDS:
                merged[k] = merged[k][:n_curves]
            yield m, merged
    finally:
        for f in cache.values():
            f.close()
