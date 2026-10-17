"""NASA PCoE battery dataset parser (MATLAB v7, scipy.io.loadmat).

Structure (verified on B0005.mat):
    mat[B0005] -> battery struct (field: cycle)
    cycle[i]    -> struct: type / ambient_temperature / time / data
    data        -> struct: Voltage_measured / Current_measured / Temperature_measured
                           / Current_charge|load / Voltage_charge|load / Time / Capacity
    discharge cycles carry a scalar 'Capacity' (Ah); charge cycles do not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def peel(x):
    """Peel leading singleton dimensions off a numpy array/scalar."""
    if isinstance(x, np.ndarray):
        while x.ndim > 0 and x.shape[0] == 1:
            x = x[0]
        return x
    return x


def load_battery(path):
    """Load a NASA .mat file; returns (cell_id, battery_struct)."""
    mat = sio.loadmat(str(path))
    name = next(k for k in mat if not k.startswith("__"))
    return name, peel(mat[name])


def iter_cycles(battery):
    """Yield (cycle_index, cycle_struct) for every cycle in a battery."""
    cycles = peel(battery["cycle"])
    for i, c in enumerate(cycles):
        yield i, peel(c)


def cycle_type(cc) -> str:
    t = peel(cc["type"])
    if isinstance(t, np.ndarray):
        t = t.ravel()[0]
    return str(t)


def ambient_temperature(cc) -> float:
    return float(np.asarray(peel(cc["ambient_temperature"])).ravel()[0])


def cycle_data(cc) -> dict[str, np.ndarray]:
    """Return the cycle 'data' fields as a dict of 1-D float arrays."""
    d = peel(cc["data"])
    out = {}
    for f in d.dtype.names:
        out[f] = np.asarray(peel(d[f]), dtype=float).ravel()
    return out


def discharge_capacity(cc) -> float:
    """Discharge capacity in Ah (uses 'Capacity' field; falls back to integration)."""
    d = peel(cc["data"])
    if "Capacity" in d.dtype.names:
        cap = np.asarray(peel(d["Capacity"]), dtype=float).ravel()
        if cap.size > 0 and np.isfinite(cap[0]):
            return float(cap[0])
    arr = cycle_data(cc)
    if arr["Time"].size < 2:
        return float("nan")
    return float(np.trapezoid(np.abs(arr["Current_measured"]), arr["Time"]) / 3600.0)


def discharge_curve(cc) -> dict[str, np.ndarray]:
    """Return t/V/I/T arrays for a discharge cycle."""
    arr = cycle_data(cc)
    return {
        "t": arr["Time"],
        "V": arr["Voltage_measured"],
        "I": arr["Current_measured"],
        "T": arr["Temperature_measured"],
    }


def build_summary(battery, cell_id: str) -> pd.DataFrame:
    """Cycle-level summary table for one battery."""
    rows = []
    for i, cc in iter_cycles(battery):
        ctype = cycle_type(cc)
        cap = discharge_capacity(cc) if ctype == "discharge" else None
        rows.append(
            {
                "cell_id": cell_id,
                "cycle_id": i,
                "type": ctype,
                "ambient_T": ambient_temperature(cc),
                "discharge_capacity_Ah": cap,
            }
        )
    df = pd.DataFrame(rows)
    # SOH relative to first discharge capacity (by cycle order)
    d = df[df["type"] == "discharge"].copy()
    if len(d):
        d["SOH"] = d["discharge_capacity_Ah"] / d["discharge_capacity_Ah"].iloc[0]
    else:
        d["SOH"] = np.nan
    return df.merge(
        d[["cell_id", "cycle_id", "SOH"]], on=["cell_id", "cycle_id"], how="left"
    )


def all_nasa_files(root: Path) -> list[Path]:
    """List all NASA .mat battery files.

    Accepts both layouts:
      data/raw/nasa/B0005.mat            (flat)
      data/raw/nasa/extracted/B0005.mat  (after unzipping Battery_Data_Set.zip)
    """
    root = Path(root)
    found: dict[str, Path] = {}
    for p in root.glob("*.mat"):
        found.setdefault(p.name, p)
    for p in (root / "extracted").glob("*.mat"):
        found.setdefault(p.name, p)
    return [found[k] for k in sorted(found)]
