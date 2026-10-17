"""阶段 B：三套数据统一预处理 → 统一 Parquet / NPZ。

用法：
    python scripts/phase_b_build_dataset.py mit      # MIT-Stanford（124 cell 官方口径）
    python scripts/phase_b_build_dataset.py nasa     # NASA PCoE（34 cell）

产物（data/processed/）：
    mit_cells.parquet    每颗电池一行：cell_id/batch/split/cycle_life/n_cycles/policy
    mit_cycles.parquet   每循环一行：Qd/Qc/IR/Tmax/Tavg/Tmin/chargetime/SOH
    mit_dq.npz           ΔQ(V) 曲线立方体 [n_cell, n_cycle, 300] + 栅格 + cell_id
    nasa_cycles.parquet  每循环一行：type/capacity/SOH
    nasa_cells.parquet   每颗电池一行：首圈容量、循环数、EOL 循环
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.dq import V_GRID, dq_curve, dq_stats, qd_from_qdlin  # noqa: E402
from src.parsers import mit, nasa  # noqa: E402

MIT_DIR = Path(os.environ.get("MIT_DIR") or (ROOT / "data" / "raw" / "mit"))
NASA_DIR = Path(os.environ.get("NASA_DIR") or (ROOT / "data" / "raw" / "nasa"))
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DQ_START, DQ_END = 1, 101  # 使用第 2~101 圈（0-based 1..100，共 100 圈）


def _require_dir(path: Path, hint: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"[错误] 找不到数据目录：{path}\n"
            f"       {hint}\n"
            f"       也可以用环境变量指定，例如：export MIT_DIR=/path/to/mit"
        )


def build_mit() -> None:
    t0 = time.time()
    _require_dir(
        MIT_DIR,
        "请把 3 个 MIT-Stanford batch 文件（*.mat，v7.3）放到 data/raw/mit/，"
        "下载地址见 data/README.md",
    )
    metas = mit.build_official_cell_list(MIT_DIR)
    print(f"[MIT] 官方口径电池数 = {len(metas)}")

    cell_rows = []
    cycle_rows = []
    dq_cube, qd_cube, dq_cells, dq_cycles, dq_stats_rows = [], [], [], [], []
    n_bad = 0
    n_curve_missing = 0

    for k, (meta, curves) in enumerate(
        mit.iter_cells_with_curves(MIT_DIR, metas, n_curves=DQ_END + 1)
    ):
        n = len(curves["V"])
        qd = meta.summary["QDischarge"]
        qc = meta.summary["QCharge"]
        qd0 = qd[1] if len(qd) > 1 and qd[1] > 0 else np.nan

        pol = meta.policy if meta.policy is not None else np.full(15, np.nan)
        row = {
            "cell_id": meta.cell_id,
            "batch": meta.batch,
            "split": meta.split,
            "cycle_life": meta.cycle_life,
            "n_cycles": meta.n_cycles,
            "q0": qd0,
        }
        for j in range(15):
            row[f"policy_{j}"] = float(pol[j]) if j < len(pol) else np.nan
        cell_rows.append(row)

        for c in range(len(qd)):
            cycle_rows.append(
                {
                    "cell_id": meta.cell_id,
                    "cycle": c + 1,
                    "QDischarge": qd[c],
                    "QCharge": qc[c] if c < len(qc) else np.nan,
                    "IR": meta.summary["IR"][c] if c < len(meta.summary["IR"]) else np.nan,
                    "Tmax": meta.summary["Tmax"][c] if c < len(meta.summary["Tmax"]) else np.nan,
                    "Tavg": meta.summary["Tavg"][c] if c < len(meta.summary["Tavg"]) else np.nan,
                    "Tmin": meta.summary["Tmin"][c] if c < len(meta.summary["Tmin"]) else np.nan,
                    "chargetime": meta.summary["chargetime"][c]
                    if c < len(meta.summary["chargetime"])
                    else np.nan,
                    "SOH": qd[c] / qd0 if np.isfinite(qd0) else np.nan,
                }
            )

        cube = np.full((DQ_END - DQ_START, len(V_GRID)), np.nan, dtype=np.float32)
        qcube = np.full((DQ_END - DQ_START, len(V_GRID)), np.nan, dtype=np.float32)
        for j, c in enumerate(range(DQ_START, DQ_END)):
            if c >= n:
                n_curve_missing += 1
                continue
            Vc, Ic = curves["V"][c], curves["I"][c]
            Qcc, Qdc = curves["Qc"][c], curves["Qd"][c]
            dq = dq_curve(Vc, Ic, Qcc, Qdc)
            if np.isnan(dq).all():
                n_bad += 1
                continue
            cube[j] = dq
            # 论文口径所需的原始放电容量曲线：优先用数据集自带的 Qdlin（官方已插值）
            qcube[j] = qd_from_qdlin(curves["Qdlin"][c])
            st = dq_stats(dq)
            dq_stats_rows.append({"cell_id": meta.cell_id, "cycle": c + 1, **st})
        dq_cube.append(cube)
        qd_cube.append(qcube)
        dq_cells.append(meta.cell_id)
        dq_cycles.append(np.arange(DQ_START + 1, DQ_END + 1))

        if (k + 1) % 20 == 0:
            print(f"  ...{k + 1}/{len(metas)} cells, {time.time() - t0:.1f}s", flush=True)

    cells = pd.DataFrame(cell_rows)
    cycles = pd.DataFrame(cycle_rows)
    cells.to_parquet(OUT / "mit_cells.parquet", index=False)
    cycles.to_parquet(OUT / "mit_cycles.parquet", index=False)
    np.savez_compressed(
        OUT / "mit_dq.npz",
        dq=np.stack(dq_cube).astype(np.float32),
        v_grid=V_GRID,
        cell_ids=np.array(dq_cells),
        cycles=np.stack(dq_cycles),
    )
    np.savez_compressed(
        OUT / "mit_qd.npz",
        qd=np.stack(qd_cube).astype(np.float32),
        v_grid=V_GRID,
        cell_ids=np.array(dq_cells),
        cycles=np.stack(dq_cycles),
    )
    pd.DataFrame(dq_stats_rows).to_parquet(OUT / "mit_dq_stats.parquet", index=False)

    stats = {
        "n_cells": int(len(cells)),
        "n_cycles_rows": int(len(cycles)),
        "split_counts": cells["split"].value_counts().to_dict(),
        "batch_counts": cells["batch"].value_counts().to_dict(),
        "cycle_life_min": float(cells["cycle_life"].min()),
        "cycle_life_median": float(cells["cycle_life"].median()),
        "cycle_life_max": float(cells["cycle_life"].max()),
        "dq_shape": list(np.stack(dq_cube).shape),
        "qd_shape": list(np.stack(qd_cube).shape),
        "qd_all_nan_cycles": int(np.isnan(np.stack(qd_cube)).all(axis=2).sum()),
        "dq_all_nan_cycles": int(n_bad),
        "dq_missing_cycles": int(n_curve_missing),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "phaseB_mit_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("[MIT] 完成：", json.dumps(stats, ensure_ascii=False))


def build_nasa() -> None:
    t0 = time.time()
    _require_dir(
        NASA_DIR,
        "请把 NASA PCoE 的 B0005/B0006/B0007/B0018.mat 放到 data/raw/nasa/，"
        "下载地址见 data/README.md",
    )
    files = nasa.all_nasa_files(NASA_DIR)
    if not files:
        raise SystemExit(
            f"[错误] {NASA_DIR} 下没有找到 .mat 文件。\n"
            f"       期望：data/raw/nasa/*.mat 或 data/raw/nasa/extracted/*.mat"
        )
    print(f"[NASA] 电池文件数 = {len(files)}")
    all_rows, cell_rows = [], []
    for p in files:
        cell_id, battery = nasa.load_battery(p)
        df = nasa.build_summary(battery, cell_id)
        all_rows.append(df)
        d = df[df["type"] == "discharge"].dropna(subset=["discharge_capacity_Ah"])
        cap = d["discharge_capacity_Ah"].values
        eol = np.where(cap < 0.8 * cap[0])[0] if len(cap) else np.array([])
        cell_rows.append(
            {
                "cell_id": cell_id,
                "n_cycles_total": int(len(df)),
                "n_discharge": int(len(d)),
                "q0": float(cap[0]) if len(cap) else np.nan,
                "q_end": float(cap[-1]) if len(cap) else np.nan,
                "eol_cycle": int(eol[0] + 1) if len(eol) else np.nan,
                "soh_end": float(cap[-1] / cap[0]) if len(cap) else np.nan,
            }
        )
        print(f"  {cell_id}: {len(df)} cycles, q0={cap[0]:.4f}" if len(cap) else f"  {cell_id}: empty", flush=True)

    cyc = pd.concat(all_rows, ignore_index=True)
    cyc.to_parquet(OUT / "nasa_cycles.parquet", index=False)
    cells = pd.DataFrame(cell_rows)
    cells.to_parquet(OUT / "nasa_cells.parquet", index=False)
    stats = {
        "n_cells": int(len(cells)),
        "n_cycles_rows": int(len(cyc)),
        "total_discharge_cycles": int((cyc["type"] == "discharge").sum()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "phaseB_nasa_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("[NASA] 完成：", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("mit", "all"):
        build_mit()
    if what in ("nasa", "all"):
        build_nasa()
