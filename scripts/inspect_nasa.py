"""Inspect NASA .mat battery file structure (first battery) for parser development."""
import argparse
import sys

import numpy as np
import scipy.io as sio


def peel(x):
    """Peel leading singleton dimensions off a numpy array."""
    if isinstance(x, np.ndarray):
        while x.ndim > 0 and x.shape[0] == 1:
            x = x[0]
        return x
    return x


def describe(x, name="", depth=0, maxdepth=3):
    pad = "  " * depth
    if isinstance(x, np.ndarray):
        print(f"{pad}{name}: ndarray shape={x.shape} dtype={x.dtype}")
        if x.dtype.names:
            for f in x.dtype.names:
                describe(x[f], f, depth + 1, maxdepth)
        elif depth < maxdepth and x.size <= 5:
            for i, xi in enumerate(x):
                describe(xi, f"[{i}]", depth + 1, maxdepth)
    elif isinstance(x, np.void):
        print(f"{pad}{name}: void fields={list(x.dtype.names or [])}")
        if x.dtype.names:
            for f in x.dtype.names:
                describe(x[f], f, depth + 1, maxdepth)
    else:
        val = repr(x)
        if len(val) > 60:
            val = val[:60] + "..."
        print(f"{pad}{name}: {type(x).__name__} = {val}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matfile")
    args = ap.parse_args()
    mat = sio.loadmat(args.matfile)
    print("top-level keys:", [k for k in mat if not k.startswith("__")])
    name = next(k for k in mat if not k.startswith("__"))
    battery = peel(mat[name])
    print("\n=== battery struct ===")
    print("battery dtype fields:", list(battery.dtype.names or []))
    cycles = peel(battery["cycle"])
    print("n_cycles:", cycles.size)
    # describe first 3 cycles
    for i in range(min(3, cycles.size)):
        print(f"\n=== cycle[{i}] ===")
        cc = peel(cycles[i])
        print("cycle dtype fields:", list(cc.dtype.names or []))
        for f in cc.dtype.names:
            if f == "data":
                d = peel(cc[f])
                print(f"  data dtype fields: {list(d.dtype.names or [])}")
                for df in d.dtype.names:
                    dv = peel(d[df])
                    print(f"    data.{df}: shape={dv.shape} dtype={dv.dtype}")
                    if dv.size <= 5:
                        print(f"      values={np.ravel(dv)[:5]}")
            else:
                describe(cc[f], f, 1, 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
