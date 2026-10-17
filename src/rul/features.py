"""RUL 预测的特征工程。

输入：阶段 B 产出的 ΔQ(V) 立方体 + 放电容量曲线立方体 + 逐循环标量表。
输出：每颗电池一行、可直接喂给模型的宽表（默认使用前 100 圈，即 cycle 2..101）。

特征分五组：
1. 滞回特征（hysteresis）：第 2 / 100 圈 Qd−Qc 曲线及其差值的统计量；
2. 论文口径 ΔQ(V)：Q_100(V) − Q_10(V) 的 log 统计量（xq_*，Severson Table 1）；
3. 定点值：3.0/3.2/3.4/3.5 V 处的滞回容量差；
4. 标量老化特征：容量、内阻、充电时间、温度、库仑效率、衰减斜率与截距；
5. 充电策略参数（15 维，表征快充协议）。

``FEATURE_SETS`` 提供两套口径：
- ``full``   ：全部特征（本项目的扩展）
- ``paper6`` ：论文 Table 1 的六个特征，用于与原文对标
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.dq import dq_cross_cycle, dq_paper_stats

C2, C100 = 2, 101  # 1-based 循环号：第 2 圈与第 100 圈（对应 dq 立方体第 0 / 99 行）
C_EARLY, C_LATE = 10, 100  # 论文 ΔQ(V) 的两端：第 10 圈与第 100 圈
V_POINTS = [3.0, 3.2, 3.4, 3.5]

# 论文 Table 1 的六个特征：3 个形状（ΔQ 曲线）+ 3 个趋势（容量序列）
PAPER6 = ["xq_log_var", "xq_log_min", "xq_log_mean",
          "qd_slope", "qd_intercept", "qd_2"]


def _cube_idx(cycle: int) -> int:
    """1-based 循环号 -> 立方体行索引（立方体第 0 行是第 2 圈）。"""
    return cycle - 2


def _idx(v: float, v_grid: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(v_grid, v), 0, len(v_grid) - 1))


def _curve_feats(curve: np.ndarray, tag: str, v_grid: np.ndarray) -> dict:
    if curve is None or np.isnan(curve).all():
        base = {f"dq_{tag}_min": np.nan, f"dq_{tag}_var": np.nan,
                f"dq_{tag}_logmin": np.nan, f"dq_{tag}_mean": np.nan}
        base.update({f"dq_{tag}_v{int(v * 10)}": np.nan for v in V_POINTS})
        return base
    mn = float(np.nanmin(curve))
    out = {
        f"dq_{tag}_min": mn,
        f"dq_{tag}_var": float(np.nanvar(curve)),
        f"dq_{tag}_logmin": float(np.log10(abs(mn) + 1e-6)),
        f"dq_{tag}_mean": float(np.nanmean(curve)),
    }
    for v in V_POINTS:
        out[f"dq_{tag}_v{int(v * 10)}"] = float(curve[_idx(v, v_grid)])
    return out


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """容量/内阻序列的线性拟合，返回 (slope, intercept)；有效点不足时返回 (nan, nan)。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan
    k, b = np.polyfit(x[m], y[m], 1)
    return float(k), float(b)


def build_features(
    dq_cube: np.ndarray,
    cell_ids: np.ndarray,
    v_grid: np.ndarray,
    cycles_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    n_cycle_window: int = 100,
    qd_cube: np.ndarray | None = None,
) -> pd.DataFrame:
    """组装特征宽表。

    dq_cube: (n_cell, n_cycle, n_v) 滞回曲线；qd_cube 同形状，放电容量曲线。
    qd_cube 为 None 时 xq_* 三个论文特征全部为 NaN（仅用于快速自检）。
    """
    cyc = cycles_df.set_index(["cell_id", "cycle"])
    rows = []
    for i, cid in enumerate(cell_ids):
        cube = dq_cube[i][:n_cycle_window]
        dq2 = cube[0] if len(cube) > 0 else None
        dq100 = cube[-1] if len(cube) > 1 else None
        diff = (dq2 - dq100) if (dq2 is not None and dq100 is not None) else None

        f = {"cell_id": str(cid)}
        f.update(_curve_feats(dq2, "2", v_grid))
        f.update(_curve_feats(dq100, "100", v_grid))
        f.update(_curve_feats(diff, "diff", v_grid))

        # 论文口径 ΔQ(V) = Q_100(V) − Q_10(V)
        if qd_cube is not None:
            qc_i = qd_cube[i]
            ie, il = _cube_idx(C_EARLY), _cube_idx(C_LATE)
            if ie < len(qc_i) and il < len(qc_i):
                f.update(dq_paper_stats(dq_cross_cycle(qc_i[ie], qc_i[il])))
            else:
                f.update(dq_paper_stats(None))
        else:
            f.update(dq_paper_stats(None))

        sub = cycles_df[cycles_df["cell_id"] == cid].sort_values("cycle")
        win = sub[(sub["cycle"] >= C2) & (sub["cycle"] <= C100)]
        cidx = win["cycle"].values.astype(float)
        qd = win["QDischarge"].values.astype(float)
        qc = win["QCharge"].values.astype(float)
        ir = win["IR"].values.astype(float)
        ct = win["chargetime"].values.astype(float)

        f["qd_2"] = float(qd[0]) if len(qd) else np.nan
        f["qd_100"] = float(qd[-1]) if len(qd) else np.nan
        f["qd_ratio"] = f["qd_100"] / f["qd_2"] if f["qd_2"] else np.nan
        f["qd_slope"], f["qd_intercept"] = _linfit(cidx, qd)
        f["qd_std"] = float(np.nanstd(qd)) if len(qd) else np.nan

        f["ir_2"] = float(ir[0]) if len(ir) else np.nan
        f["ir_100"] = float(ir[-1]) if len(ir) else np.nan
        f["ir_ratio"] = f["ir_100"] / f["ir_2"] if f["ir_2"] else np.nan
        f["ir_slope"], _ = _linfit(cidx, ir)

        f["ct_2"] = float(ct[0]) if len(ct) else np.nan
        f["ct_100"] = float(ct[-1]) if len(ct) else np.nan
        f["ct_ratio"] = f["ct_100"] / f["ct_2"] if f["ct_2"] else np.nan

        f["tmax_2"] = float(win["Tmax"].iloc[0]) if len(win) else np.nan
        f["tmax_100"] = float(win["Tmax"].iloc[-1]) if len(win) else np.nan
        f["tavg_2"] = float(win["Tavg"].iloc[0]) if len(win) else np.nan
        f["tavg_100"] = float(win["Tavg"].iloc[-1]) if len(win) else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            ce = qd / qc
        f["ce_mean"] = float(np.nanmean(ce)) if len(ce) else np.nan

        rows.append(f)

    feat = pd.DataFrame(rows)
    pol_cols = [c for c in cells_df.columns if c.startswith("policy_")]
    feat = feat.merge(cells_df[["cell_id", "cycle_life", "split", "batch"] + pol_cols],
                      on="cell_id", how="left")
    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("cell_id", "cycle_life", "split", "batch")]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """两套特征口径，用于消融对比。

    - ``full``   ：全部特征（滞回 + 论文 ΔQ + 标量 + 策略）
    - ``paper6`` ：论文 Table 1 的六个特征，用于与原文 9.1% 对标
    """
    return {
        "full": feature_columns(df),
        "paper6": [c for c in PAPER6 if c in df.columns],
    }
