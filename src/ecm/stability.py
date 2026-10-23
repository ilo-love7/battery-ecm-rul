"""ECM 拟合的数值稳定性 / 初值敏感性 / 可辨识性诊断（离线，合成数据）。

为什么单独有这个模块
--------------------
``fit_rc`` 是一个带上下界信赖域最小二乘。代码本身看不出、但必须被固化成
CI 事实的结论有三条，全部由 ``tests/test_ecm_stability.py`` + 已入库基线
``tests/regression_baseline.json`` 守住：

1. **2-RC 的参数向量只在「支路置换」意义下可辨识**：
   ``V = OCV - I*[R0 + R1(1-e^-t/tau1) + R2(1-e^-t/tau2)]`` 对
   ``(R1, tau1) <-> (R2, tau2)`` 不变。两种标注拟合到机器精度、代价完全相同，
   所以 R1/R2 与 tau1/tau2 这两个**标签**在单条恒流放电曲线上不可恢复，
   能恢复的只是**无序列** ``{(R1,tau1), (R2,tau2)}``。这是模型结构决定的，
   不是求解器缺陷，也不该靠"给拟合加个 tau1<tau2 约束"来假装解决。
2. **求解器会对没有信息量的输入报"收敛"**：空数组、零激励（I≡0 或 V≡0 变化）、
   以及样本数 <= 参数个数时，``res.status`` 仍可能是 1（"gtol satisfied"）。
   这类情况不是"求解失败"，而是**问题不可辨识**；由 ``classify_fit`` 分类，
   报告为 ``underdetermined`` / ``no-excitation`` / ``bad-fit``。
3. **内部一致性由残差量级判断**：模型解释不了数据时，残差 RMS 相对量程会很大，
   而 ``status`` 依旧可能是 1 或 4。``classify_fit`` 用
   ``RESID_FRACTION_OF_SPAN_MAX`` 把"收敛但不解释数据"单独标出来。

阈值依据（都在下面的常量注释里，另有 ``tests/regression_baseline.json`` 记录副本）
------------------------------------------------------------------------------
- 合成拟合无噪声、求解器确定性（同一构建重复跑 ``res.x`` 逐位相同，实测漂移 0.0）；
  ``fit_rc`` 用 ``xtol=ftol=gtol=1e-12``，double 的 eps = 2.2e-16。
- 因此 ``PARAM_REL_DRIFT_MAX = 1e-6``：比 eps 高约 10 个数量级、比求解器容差高约
  6 个数量级，留出 scipy/BLAS 在 CI（ubuntu-latest / requirements.txt 轮子）与本地
  Windows 之间的差异余量，同时仍能抓住任何真实变化——对外参数只报到 4–5 位有效数字
  （如 R0 = 107.35 mΩ），1e-6 相对 = 1e-4 mΩ，远低于任何人汇报的精度。
- ``PRED_MAX_ABS_V = 1e-9``（1 nV）：两次收敛拟合的预测差实测约 1e-13 V，
  1 nV 高 4 个数量级；又比 ``tests/test_core.py`` 里用的 1 mV 噪声底低 6 个数量级，
  所以只有"真的收敛到了另一个解"才会违约。
- ``RESID_FRACTION_OF_SPAN_MAX = 0.05``：与本项目 ECM 复测的验收口径同源
  （"≤ 量程 2%"），这里故意放松到 5%，只用于"区分'解释不了数据'与'正常拟合'"，
  不用于评判精度。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ecm import _bounds, fit_rc, model_voltage

# ---------------------------------------------------------------------------
# 阈值（依据见模块 docstring / README「Known limitations」）
# ---------------------------------------------------------------------------
PARAM_REL_DRIFT_MAX = 1e-6      # 相对参数漂移阈值
PRED_MAX_ABS_V = 1e-9           # 预测一致性阈值（V）
RECOVERY_RTOL = 1e-3            # 对合成真值的恢复容差（与 tests/test_core.py 同口径）
RECOVERY_ATOL = 1e-5
RESID_FRACTION_OF_SPAN_MAX = 0.05
# "贴住上下界"的判定容差：求解器的 xtol=1e-12 约束的是**步长**，不是到边界的距离，
# 所以真正撞界时返回值通常在边界上"软着陆"（实测 t 传成小时的病态例子里
# tau1 = 1.0000000042840174 而 lb = 1.0，相对偏差 4.3e-9）。1e-8 相对容差
# 把这类"数值上已与边界不可区分"的情形算作撞界；同时远小于任何物理上有意义的
# 参数变化（最粗的边界是 tau1 的 1 s，1e-8 相对 = 1e-8 s）。
BOUND_REL_TOL = 1e-8

BASELINE_SCHEMA_VERSION = 1
BASELINE_PATH = Path(__file__).resolve().parents[2] / "tests" / "regression_baseline.json"

# ---------------------------------------------------------------------------
# 合成问题（确定性、无随机数）
# ---------------------------------------------------------------------------
# [R0 (Ohm), R1, tau1 (s), R2, tau2 (s)] —— 与 tests/test_core.py 的真值一致
TRUE_PARAMS = [0.030, 0.020, 30.0, 0.010, 300.0]
CAP_AH = 1.1


def ocv_linear(soc):
    """单调、有斜率的 OCV(SOC)，用于把辨识问题与 OCV 假设解耦。"""
    return 3.30 + 0.30 * np.asarray(soc, dtype=float)


def synthetic_discharge(n: int = 4000):
    """一段确定性恒流放电：返回 (t, i_mag, soc)。"""
    t = np.linspace(0.0, 3600.0, n)
    soc = np.linspace(0.0, 1.0, n)
    i_mag = np.full(n, CAP_AH)
    return t, i_mag, soc


def synthetic_voltage(t, i_mag, soc, params=TRUE_PARAMS, order: int = 2):
    """由真值参数生成无噪声端电压。"""
    return model_voltage(t, i_mag, ocv_linear, soc, params, order)


# 多组初值，覆盖"好初值"与"坏初值"（贴上下界、量级错、支路顺序颠倒）
INITIAL_GUESSES = {
    "default": [0.05, 0.05, 30.0, 0.05, 300.0],
    "r0_at_upper_bound": [1.0, 0.5, 30.0, 0.5, 300.0],
    "tiny_resistances": [1e-8, 1e-8, 1.0, 1e-8, 5.0],
    "tau1_at_lower_bound": [0.05, 0.05, 1.0, 0.05, 1000.0],
    "tau2_at_upper_bound": [0.05, 0.05, 30.0, 0.05, 1e5],
    "large_all": [0.9, 0.5, 5.0, 0.5, 10.0],
    # 故意让 tau1 > tau2：进入"支路颠倒"的等价盆地
    "swapped_branch_order": [0.05, 0.010, 300.0, 0.020, 30.0],
}
BAD_GUESSES = ("tiny_resistances", "large_all", "swapped_branch_order")
REFERENCE_GUESS = "default"

PARAM_NAMES = ("R0", "R1", "tau1", "R2", "tau2")


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def rel_drift(x, ref) -> float:
    """相对漂移：max_i |x_i - ref_i| / max(|ref_i|, 1e-12)。

    参数非零时就是普通的最大相对偏差；分母的 1e-12 地板避免"参数被压到 0"
    时出现 0/0。
    """
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if x.shape != ref.shape:
        raise ValueError(f"rel_drift: shape mismatch {x.shape} vs {ref.shape}")
    denom = np.maximum(np.abs(ref), 1e-12)
    return float(np.max(np.abs(x - ref) / denom))


def branch_set(p, order: int = 2):
    """把参数向量压成**无序列**支路集合 {(R, tau), ...}（排序后）。"""
    p = np.asarray(p, dtype=float)
    if order == 1:
        return ((float(p[1]), float(p[2])),)
    branches = ((float(p[1]), float(p[2])), (float(p[3]), float(p[4])))
    return tuple(sorted(branches))


def branch_sets_allclose(a, b, rtol: float = 1e-9) -> bool:
    """无序列支路集合的**容差**比较。

    逐位相等太脆：``fit_rc`` 的返回值在末位浮点上有 1e-16 量级差异，用 ``==``
    会把"恢复成功"误判成失败（实测连默认初值都会被误判成 False）。
    """
    if len(a) != len(b):
        return False
    return all(
        np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                    rtol=rtol, atol=0.0)
        for x, y in zip(a, b)
    )


def param_count(order: int = 2, r0_fixed=None) -> int:
    """自由参数个数（与 ``fit_rc`` 的优化变量一致）。"""
    if r0_fixed is None:
        return 3 if order == 1 else 5
    return 2 if order == 1 else 4


def max_abs_voltage_diff(t, i_mag, soc, p_a, p_b, order: int = 2) -> float:
    """两组参数的预测电压最大绝对差（V）。"""
    va = model_voltage(t, i_mag, ocv_linear, soc, p_a, order)
    vb = model_voltage(t, i_mag, ocv_linear, soc, p_b, order)
    return float(np.max(np.abs(va - vb)))


def residual_rms(res) -> float:
    """残差 RMS（V）；空残差返回 nan（不应再有这种情况，输入校验会先报错）。"""
    fun = np.asarray(res.fun, dtype=float)
    if fun.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(fun ** 2)))


def bound_hits(x, order: int = 2, r0_fixed=None):
    """返回撞到上下界的自由参数名列表。"""
    x = np.asarray(x, dtype=float)
    if r0_fixed is None:
        lb, ub = _bounds(order)
        names = PARAM_NAMES[:3] if order == 1 else PARAM_NAMES
    else:
        if order == 1:
            lb, ub, names = [0.0, 1.0], [1.0, 1e4], ("R1", "tau1")
        else:
            lb, ub, names = ([0.0, 1.0, 0.0, 5.0], [1.0, 1e4, 1.0, 1e5],
                             ("R1", "tau1", "R2", "tau2"))
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    hits = []
    for i, name in enumerate(names):
        if abs(x[i] - lb[i]) <= BOUND_REL_TOL * max(1.0, abs(lb[i])):
            hits.append(name + "=lb")
        elif abs(x[i] - ub[i]) <= BOUND_REL_TOL * max(1.0, abs(ub[i])):
            hits.append(name + "=ub")
    return hits


def solver_failure_reason(res) -> str | None:
    """把"求解失败"翻译成可读原因；``None`` 表示求解器未报失败。

    与 ``fit_rc`` 抛出的 ``ValueError``（问题本身非法）是两个不同类别：
    本函数只看 ``res.status``。
    """
    status = int(getattr(res, "status", -1))
    if status > 0:
        return None
    message = str(getattr(res, "message", ""))
    if status == 0:
        return f"max_nfev reached without convergence ({message})"
    return f"improper input to the solver, status={status} ({message})"


def classify_fit(res, V_meas, i_mag, order: int = 2, r0_fixed=None) -> dict:
    """把一次 ``fit_rc`` 结果分类，暴露"静默成功"的病态情形。

    分类项：
    - ``solver-failure``：``res.status <= 0``
    - ``underdetermined``：样本数 <= 自由参数个数（拟合不唯一，实测 n=5 时
      RMSE 6e-9 V 却把 tau2 报成 5.7e4 s）
    - ``no-excitation``：I≡0 或 V 无变化（参数对电压无影响）
    - ``bad-fit``：残差 RMS 超过量程的 ``RESID_FRACTION_OF_SPAN_MAX``
      （量程为 0 时用 1e-6 V 绝对门槛）
    - ``bound-hit:<name>``：某自由参数贴住上下界
    - ``non-finite-parameters``
    - 以上都没有 → ``ok``
    """
    x = np.asarray(res.x, dtype=float)
    V = np.asarray(V_meas, dtype=float)
    I = np.asarray(i_mag, dtype=float)
    rms = residual_rms(res)
    span = float(np.max(V) - np.min(V)) if V.size else 0.0
    n = int(V.size)
    cats = []

    if not np.all(np.isfinite(x)):
        cats.append("non-finite-parameters")
    if solver_failure_reason(res) is not None:
        cats.append("solver-failure")
    if n <= param_count(order, r0_fixed):
        cats.append("underdetermined")
    if I.size == 0 or float(np.max(np.abs(I))) <= 1e-12 or span <= 1e-12:
        cats.append("no-excitation")
    if span > 1e-12:
        if rms / span > RESID_FRACTION_OF_SPAN_MAX:
            cats.append("bad-fit")
    elif rms > 1e-6:
        cats.append("bad-fit")
    hits = bound_hits(x, order, r0_fixed)
    if hits:
        cats.append("bound-hit:" + ",".join(hits))
    if not cats:
        cats.append("ok")

    return {
        "n_samples": n,
        "n_params": param_count(order, r0_fixed),
        "status": int(getattr(res, "status", -1)),
        "message": str(getattr(res, "message", "")),
        "nfev": int(getattr(res, "nfev", -1)),
        "residual_rms_V": rms,
        "signal_span_V": span,
        "residual_over_span": (rms / span) if span > 1e-12 else None,
        "bound_hits": hits,
        "categories": cats,
        "ok": cats == ["ok"],
    }


# ---------------------------------------------------------------------------
# 基线：生成 / 比较
# ---------------------------------------------------------------------------
def _fit_once(t, i_mag, V, soc, p0, order: int = 2, r0_fixed=None):
    res = fit_rc(t, i_mag, V, ocv_linear, soc, order=order, r0_fixed=r0_fixed, p0=p0)
    return res


def build_baseline(order: int = 2) -> dict:
    """跑完所有初值，生成可提交的回归基线（纯函数，无副作用）。"""
    t, i_mag, soc = synthetic_discharge()
    V = synthetic_voltage(t, i_mag, soc, order=order)

    fits = {}
    for name, p0 in INITIAL_GUESSES.items():
        res = _fit_once(t, i_mag, V, soc, p0, order=order)
        info = classify_fit(res, V, i_mag, order=order)
        fits[name] = {
            "initial_guess": [float(v) for v in p0],
            "x": [float(v) for v in np.asarray(res.x, dtype=float)],
            "branch_set": [[a, b] for a, b in branch_set(res.x, order)],
            "status": info["status"],
            "nfev": info["nfev"],
            "residual_rms_V": info["residual_rms_V"],
            "bound_hits": info["bound_hits"],
            "categories": info["categories"],
        }

    ref = np.asarray(fits[REFERENCE_GUESS]["x"], dtype=float)
    for name, entry in fits.items():
        x = np.asarray(entry["x"], dtype=float)
        entry["param_rel_drift_vs_reference"] = rel_drift(x, ref)
        entry["pred_max_abs_v_vs_reference"] = max_abs_voltage_diff(
            t, i_mag, soc, x, ref, order
        )

    truth = np.asarray(TRUE_PARAMS[: (3 if order == 1 else 5)], dtype=float)
    recovery = {
        name: {
            "param_rel_drift_vs_truth": rel_drift(np.asarray(e["x"], dtype=float), truth),
            "branch_set_matches_truth": branch_sets_allclose(
                branch_set(np.asarray(e["x"], dtype=float), order),
                branch_set(truth, order),
            ),
        }
        for name, e in fits.items()
    }

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "generated_by": "scripts/phase_h_ecm_stability.py",
        "problem": {
            "kind": "synthetic noiseless Thevenin discharge",
            "order": order,
            "n_samples": int(t.size),
            "current_A": float(np.max(i_mag)),
            "soc_span": [float(soc.min()), float(soc.max())],
            "true_params": [float(v) for v in truth],
            "param_names": list(PARAM_NAMES[: (3 if order == 1 else 5)]),
        },
        "tolerances": {
            "param_rel_drift_max": PARAM_REL_DRIFT_MAX,
            "pred_max_abs_v": PRED_MAX_ABS_V,
            "recovery_rtol": RECOVERY_RTOL,
            "recovery_atol": RECOVERY_ATOL,
            "resid_fraction_of_span_max": RESID_FRACTION_OF_SPAN_MAX,
        },
        "reference_guess": REFERENCE_GUESS,
        "fits": fits,
        "truth_recovery": recovery,
        "real_data_artifact": real_data_artifact(),
    }


def real_data_artifact() -> dict:
    """把仓内已提交的真实数据产物快照进基线，供离线回归核对。

    读 ``reports/phaseA2_ecm_results.csv``（NASA PCoE 4 颗电池的 ECM 辨识结果，
    已入库，故不需要 8.6 GB 原始数据）。同时记下 README 对外口径的中位复测 RMSE。
    """
    import csv

    path = Path(__file__).resolve().parents[2] / "reports" / "phaseA2_ecm_results.csv"
    if not path.exists():
        return {"source": str(path), "available": False}
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["cell"]] = {
                "R0_step_mohm": float(row["R0_step_mohm"]),
                "Rdc50_mohm": float(row["Rdc50_mohm"]),
                "Rpol50_mohm": float(row["Rpol50_mohm"]),
                "rmse_test_raw_pct": float(row["rmse_test_raw_pct"]),
                "rmse_test_cal_pct": float(row["rmse_test_cal_pct"]),
            }
    med_raw = float(np.median([r["rmse_test_raw_pct"] for r in rows.values()]))
    med_cal = float(np.median([r["rmse_test_cal_pct"] for r in rows.values()]))
    return {
        "source": "reports/phaseA2_ecm_results.csv",
        "available": True,
        "cells": rows,
        "median_rmse_test_raw_pct": med_raw,
        "median_rmse_test_cal_pct": med_cal,
        # README「Results at a glance」的验收口径：同工况复测 RMSE ≤ 量程 2%
        "acceptance_pct_of_range": 2.0,
    }


def compare_to_baseline(baseline: dict, candidate: dict | None = None,
                        order: int = 2) -> list[str]:
    """把"现在跑出来的"和基线比，返回失败信息列表（空 = 通过）。"""
    candidate = candidate if candidate is not None else build_baseline(order=order)
    failures: list[str] = []

    if candidate["schema_version"] != baseline.get("schema_version"):
        failures.append(
            f"schema_version mismatch: baseline={baseline.get('schema_version')} "
            f"candidate={candidate['schema_version']}"
        )
    tol = baseline.get("tolerances", {})
    drift_max = float(tol.get("param_rel_drift_max", PARAM_REL_DRIFT_MAX))
    pred_max = float(tol.get("pred_max_abs_v", PRED_MAX_ABS_V))

    base_fits = baseline.get("fits", {})
    cand_fits = candidate["fits"]
    for name in sorted(set(base_fits) | set(cand_fits)):
        if name not in base_fits:
            failures.append(f"{name}: new initial guess not present in baseline")
            continue
        if name not in cand_fits:
            failures.append(f"{name}: initial guess missing from candidate run")
            continue
        b = np.asarray(base_fits[name]["x"], dtype=float)
        c = np.asarray(cand_fits[name]["x"], dtype=float)
        drift = rel_drift(c, b)
        if drift > drift_max:
            failures.append(
                f"{name}: parameter drift {drift:.3e} > {drift_max:.3e} "
                f"(baseline={np.array2string(b, precision=6)}, "
                f"now={np.array2string(c, precision=6)})"
            )
        pred = float(cand_fits[name]["pred_max_abs_v_vs_reference"])
        if pred > pred_max:
            failures.append(
                f"{name}: prediction deviates {pred:.3e} V > {pred_max:.3e} V "
                "from the reference-guess fit"
            )
        if cand_fits[name]["categories"] != base_fits[name]["categories"]:
            failures.append(
                f"{name}: category change {base_fits[name]['categories']} -> "
                f"{cand_fits[name]['categories']}"
            )
    return failures


def write_baseline(path: Path | str = BASELINE_PATH, order: int = 2) -> Path:
    """把当前基线写进 JSON（UTF-8，无 BOM，末尾换行）。"""
    path = Path(path)
    payload = build_baseline(order=order)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="ECM 拟合稳定性回归基线（离线，无需原始数据集）"
    )
    parser.add_argument("--out", default=str(BASELINE_PATH), help="基线输出路径")
    parser.add_argument("--check", action="store_true",
                        help="不写文件，只和已提交基线比对（漂移超阈值 → exit 1）")
    args = parser.parse_args(argv)

    if args.check:
        baseline = json.loads(Path(args.out).read_text(encoding="utf-8"))
        failures = compare_to_baseline(baseline)
        if failures:
            print(f"[FAIL] {len(failures)} drift(s) vs {args.out}")
            for f in failures:
                print("  - " + f)
            return 1
        print(f"[PASS] no drift beyond tolerance vs {args.out}")
        return 0

    path = write_baseline(args.out)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"[write] {path}")
    for name, entry in payload["fits"].items():
        print(f"  {name:22s} categories={entry['categories']} "
              f"drift_vs_ref={entry['param_rel_drift_vs_reference']:.3e}")
    art = payload["real_data_artifact"]
    if art.get("available"):
        print(f"  real-data artifact: median re-test RMSE raw "
              f"{art['median_rmse_test_raw_pct']:.4f}% / cal "
              f"{art['median_rmse_test_cal_pct']:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())