"""Inspect MIT (Severson 2019) HDF5 batch .mat file structure.

MATLAB v7.3 = HDF5. Struct arrays appear as groups with reference arrays;
use h5py and dereference. Run this FIRST on a downloaded batch file to learn
the exact layout before finalizing src/parsers/mit.py.
"""
import argparse
import sys

import h5py
import numpy as np


def describe(name, obj, depth=0, maxdepth=4):
    pad = "  " * depth
    if isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        ref = h5py.check_dtype(ref=dtype)
        print(f"{pad}{name}: Dataset shape={shape} dtype={dtype} ref={ref is not None}")
        if obj.size <= 5 and not ref and depth < maxdepth:
            try:
                print(f"{pad}  values={np.asarray(obj).ravel()[:5]}")
            except Exception as e:
                print(f"{pad}  (read err {e})")
    elif isinstance(obj, h5py.Group):
        keys = list(obj.keys())
        print(f"{pad}{name}: Group keys={keys}")
        if depth < maxdepth:
            for k in keys:
                describe(k, obj[k], depth + 1, maxdepth)


def deref(f, ref, depth=0, maxdepth=3):
    pad = "  " * depth
    try:
        obj = f[ref]
        if isinstance(obj, h5py.Dataset):
            print(f"{pad}-> Dataset shape={obj.shape} dtype={obj.dtype} fields={obj.dtype.names}")
            return obj
        elif isinstance(obj, h5py.Group):
            print(f"{pad}-> Group keys={list(obj.keys())}")
            return obj
    except Exception as e:
        print(f"{pad}-> deref err {e}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matfile")
    args = ap.parse_args()

    f = h5py.File(args.matfile, "r")
    print("top-level keys:", list(f.keys()))
    for k in f.keys():
        describe(k, f[k], 0, 3)

    # try to dereference batch/summary
    print("\n=== batch/summary probe ===")
    try:
        s = f["batch"]["summary"]
        print("batch/summary: Dataset shape", s.shape, "dtype", s.dtype,
              "ref", h5py.check_dtype(ref=s.dtype) is not None)
        if s.shape[0] > 0:
            ref0 = s[0]
            print("first ref:", ref0)
            obj = deref(f, ref0)
            if isinstance(obj, h5py.Dataset) and obj.dtype.names:
                print("  summary fields:", obj.dtype.names)
                for fld in obj.dtype.names:
                    try:
                        print(f"    {fld}: {np.asarray(obj[fld]).ravel()[:6]}")
                    except Exception as e:
                        print(f"    {fld}: err {e}")
    except Exception as e:
        print("batch/summary probe err:", type(e).__name__, e)

    print("\n=== batch top keys ===")
    try:
        b = f["batch"]
        if isinstance(b, h5py.Group):
            print("batch keys:", list(b.keys()))
            for k in b.keys():
                describe(k, b[k], 1, 2)
    except Exception as e:
        print("batch err:", e)

    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
