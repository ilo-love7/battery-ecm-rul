"""ECM 拟合的数值稳定性 / 初值敏感性 / 可辨识性回归测试（离线，合成数据）。

守住三条结论（完整推导见 ``src/ecm/stability.py`` 的模块 docstring）：

1. **2-RC 的参数向量只在「支路置换」意义下可辨识。**
   ``V = OCV - I·[R0 + R1(1-e^-t/tau1) + R2(1-e^-t/tau2)]`` 对
   ``(R1, tau1) <-> (R2, tau2)`` 不变，所以带标签的 R1/R2、tau1/tau2 无法从单条
   恒流放电曲线恢复，能恢复的只是**无序列**支路集合 ``{(R1,tau1),(R2,tau2)}``。
   这不是求解器缺陷，也不该靠"给拟合加 tau1<tau2 约束"来假装解决。

2. **求解器会对没有信息量的输入报"收敛"。**
   空数组、零激励（I≡0）以及样本数 <= 自由参数个数时，``res.status`` 仍可能是 1。
   这类不是"求解失败"而是**问题不可辨识**，由 ``classify_fit`` 分类为
   ``underdetermined`` / ``no-excitation``。

3. **内部一致性由残差量级判断**：模型解释不了数据时残差 RMS 相对量程会很大，
   而 ``status`` 依旧可能是 1 或 4。

另外在 CI 里守住**已入库的回归基线** ``tests/regression_baseline.json``：今天跑出来
的拟合若相对基线漂移超过 tolerance，本文件就会红（这就是"参数漂移超阈值即失败"）。

运行：``python -m pytest tests/ -q``（合成数据，不需要 NASA / MIT 原始数据集）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ecm.ecm import fit_rc
from src.ecm.stability import (
    BASELINE_PATH,
    BOUND_REL_TOL,
    INITIAL_GUESSES,
    PARAM_REL_DRIFT_MAX,
    PRED_MAX_ABS_V,
    RECOVERY_RTOL,
    TRUE_PARAMS,
    bound_hits,
    branch_set,
    branch_sets_allclose,
    classify_fit,
    compare_to_baseline,
    ocv_linear,
    rel_drift,
    synthetic_discharge,
    synthetic_voltage,
)

ORDER = 2
T, I_MAG, SOC = synthetic_discharge()
V_MEAS = synthetic_voltage(T, I_MAG, SOC, order=ORDER)
TRUE_BRANCHES = branch_set(TRUE_PARAMS, ORDER)


def _load_baseline() -> dict:
    assert BASELINE_PATH.exists(), f"缺少已入库基线 {BASELINE_PATH}"
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _fit(p0=None, order=ORDER, t=None, i_mag=None, v=None, soc=None):
    """在无噪声合成曲线上跑一次拟合；默认复用模块级的确定性数据。"""
    return fit_rc(
        T if t is None else t,
        I_MAG if i_mag is None else i_mag,
        V_MEAS if v is None else v,
        ocv_linear,
        SOC if soc is None else soc,
        order=order,
        p0=p0,
    )


# ---------------------------------------------------------------------------
# 已入库基线：schema + 无漂移（本文件的核心闸门）
# ---------------------------------------------------------------------------
def test_committed_baseline_exists_and_has_schema():
    """基线必须随仓库提交，且覆盖全部已登记初值。"""
    baseline = _load_baseline()
    assert baseline["schema_version"] == 1
    assert set(baseline["fits"]) == set(INITIAL_GUESSES), (
        "基线里的初值集合与 INITIAL_GUESSES 不一致："
        f"{sorted(set(baseline['fits']) ^ set(INITIAL_GUESSES))}"
    )
    tol = baseline["tolerances"]
    assert tol["param_rel_drift_max"] == PARAM_REL_DRIFT_MAX
    assert tol["pred_max_abs_v"] == PRED_MAX_ABS_V
    for name, entry in baseline["fits"].items():
        assert len(entry["x"]) == len(TRUE_PARAMS), f"{name}: 参数向量长度不对"
        assert entry["categories"], f"{name}: 缺少分类标记"


def test_current_code_does_not_drift_from_committed_baseline():
    """重新拟合一遍，任何参数/预测/分类相对基线的漂移都判失败。

    阈值依据（见 stability.py 常量注释）：求解器收敛到 xtol=ftol=gtol=1e-12，
    double eps≈2.2e-16，因此 1e-6 的相对漂移比"单次求解器噪声"高 6 个数量级、
    比任何真实参数变化低若干个数量级——既能容忍 CI 与本机 BLAS 差异，又能抓住
    任何真实的行为改变。
    """
    failures = compare_to_baseline(_load_baseline())
    assert failures == [], (
        "拟合相对已入库基线发生漂移（若这是有意的行为变更，"
        "请重跑 scripts/phase_h_ecm_stability.py 更新基线并说明原因）：\n  - "
        + "\n  - ".join(failures)
    )


# ---------------------------------------------------------------------------
# 初值敏感性：好初值与坏初值都应落到同一个"等价盆地"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(INITIAL_GUESSES))
def test_initial_guess_fits_are_finite_and_converged(name):
    """任何已登记初值都不得产生 NaN/inf，也不得是求解器失败。"""
    res = _fit(p0=INITIAL_GUESSES[name])
    assert np.all(np.isfinite(res.x)), f"{name}: 拟合出现非有限参数 {res.x}"
    assert int(res.status) > 0, f"{name}: 求解器失败 status={res.status}"


@pytest.mark.parametrize("name", sorted(INITIAL_GUESSES))
def test_initial_guess_recovers_the_true_branch_set(name):
    """关键结论：无论初值好坏，恢复出的**无序列支路集合**都是真值。"""
    res = _fit(p0=INITIAL_GUESSES[name])
    recovered = branch_set(res.x, ORDER)
    assert branch_sets_allclose(recovered, TRUE_BRANCHES, rtol=RECOVERY_RTOL), (
        f"{name}: 支路集合未恢复到真值\n  真值={TRUE_BRANCHES}\n  实测={recovered}"
    )


def test_swapped_branch_order_is_a_permutation_not_a_wrong_answer():
    """tau1 > tau2 的初值进的是等价盆地：标签变了，物理答案没变。

    这条测试把"可辨识性边界"写死：带标签的参数向量差异巨大（>1 相对漂移），
    但无序列支路集合与预测电压都与真值一致。因此对外报告只能报**支路集合**。
    """
    name = "swapped_branch_order"
    res = _fit(p0=INITIAL_GUESSES[name])
    assert branch_sets_allclose(branch_set(res.x, ORDER), TRUE_BRANCHES,
                                rtol=RECOVERY_RTOL), "等价盆地内的解必须与真值同集合"
    assert rel_drift(res.x, TRUE_PARAMS) > 1.0, (
        "这一初值本应落在标签颠倒的等价解上；若相对漂移变小，"
        "说明拟合被某个约束强行拉回了标签顺序，可辨识性结论需要重新评估"
    )


def test_noiseless_fit_residual_is_machine_level():
    """无噪声合成问题上残差应是机器量级——否则问题/求解器配置已变。"""
    res = _fit()
    rms = float(np.sqrt(np.mean(np.asarray(res.fun, dtype=float) ** 2)))
    span = float(np.max(V_MEAS) - np.min(V_MEAS))
    assert rms < 1e-12, f"无噪声问题的残差 RMS={rms:.3e} V，远高于机器量级"
    assert span > 0.1, "合成曲线的量程过小，测试失去意义"


# ---------------------------------------------------------------------------
# 数值稳定性：畸形输入必须报错，且必须能和"求解器失败"区分开
# ---------------------------------------------------------------------------
def test_malformed_input_raises_value_error_instead_of_returning_garbage():
    """空 / NaN / 形状不符 / 坏 order / 坏 p0 一律 ValueError。

    改动前这些输入会**静默**返回初值本身并报 status=1（"收敛"），
    rmse = nan —— 这是本模块存在的首要原因。
    """
    cases = [
        ("empty t", dict(t=np.array([]), i_mag=np.array([]), v=np.array([]), soc=np.array([]))),
        ("NaN in V", dict(v=np.where(np.arange(T.size) == 10, np.nan, V_MEAS))),
        ("inf in i_mag", dict(i_mag=np.where(np.arange(T.size) == 5, np.inf, I_MAG))),
        ("length mismatch", dict(soc=SOC[:-1])),
        ("2-D t", dict(t=T.reshape(-1, 1))),
    ]
    for label, kwargs in cases:
        with pytest.raises(ValueError):
            _fit(**kwargs)
    # 坏 order 与坏 p0 走各自的守卫
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        _fit(order=3)
    with pytest.raises(ValueError, match="p0 must have"):
        _fit(p0=[0.05, 0.05])


def test_scalar_constant_current_is_still_accepted():
    """标量电流是既有公开用法（tests/test_core.py 与早期脚本如此调用）。

    纯粹的 0-D 校验会把它误判为畸形输入并让 3 个既有测试变红；这条测试把
    "标量 i_mag 必须继续可用，且与数组写法给出同一答案"钉住。
    """
    scalar_current = float(I_MAG[0])
    res = fit_rc(T, scalar_current, V_MEAS, ocv_linear, SOC, order=ORDER)
    assert np.all(np.isfinite(res.x)), res.x
    assert int(res.status) > 0, f"标量电流不应导致求解器失败：status={res.status}"
    assert branch_sets_allclose(branch_set(res.x, ORDER), TRUE_BRANCHES,
                                rtol=RECOVERY_RTOL)
    assert rel_drift(res.x, _fit().x) <= PARAM_REL_DRIFT_MAX, (
        "标量电流与数组电流应给出同一个拟合结果（标量需先广播成常数数组）"
    )


def test_empty_or_nonfinite_input_says_invalid_problem_not_solver_failure():
    """错误信息要能区分「问题非法」与「求解器没收敛」。"""
    with pytest.raises(ValueError) as exc:
        _fit(t=np.array([]), i_mag=np.array([]), v=np.array([]), soc=np.array([]))
    assert "not a solver failure" in str(exc.value), (
        f"空输入的报错应说明这是非法输入而非求解器失败，实际：{exc.value}"
    )
    with pytest.raises(ValueError) as exc:
        _fit(v=np.where(np.arange(T.size) == 3, np.nan, V_MEAS))
    assert "not a solver failure" in str(exc.value), (
        f"含 NaN 的报错应说明这是非法数据而非求解器失败，实际：{exc.value}"
    )


# ---------------------------------------------------------------------------
# 静默成功：分类器必须把"没有信息量"的病态情形标出来
# ---------------------------------------------------------------------------
def test_no_excitation_is_classified_not_silently_ok():
    """I≡0 时参数对电压没有影响：status 可能仍是 1，但分类不得是 ok。"""
    i_zero = np.zeros_like(I_MAG)
    v_zero = synthetic_voltage(T, i_zero, SOC, order=ORDER)
    res = _fit(i_mag=i_zero, v=v_zero)
    cls = classify_fit(res, v_zero, i_zero, order=ORDER)
    assert "no-excitation" in cls["categories"], cls
    assert cls["ok"] is False, f"I≡0 不应被判为 ok：{cls}"


def test_underdetermined_is_classified_not_silently_ok():
    """样本数 <= 自由参数个数（order=2 有 5 个）时拟合不唯一。"""
    n = 5
    t, i_mag, soc = T[:n], I_MAG[:n], SOC[:n]
    v = synthetic_voltage(t, i_mag, soc, order=ORDER)
    res = _fit(t=t, i_mag=i_mag, v=v, soc=soc)
    cls = classify_fit(res, v, i_mag, order=ORDER)
    assert "underdetermined" in cls["categories"], cls
    assert cls["ok"] is False, f"n=5 且 5 个自由参数不应被判为 ok：{cls}"


def test_bad_fit_is_classified_not_silently_ok():
    """模型解释不了数据时，残差相对量程会很大——必须被标出来。"""
    period = T[-1]
    v_bad = V_MEAS + 0.5 * np.sin(40.0 * np.pi * T / period)
    res = _fit(v=v_bad)
    cls = classify_fit(res, v_bad, I_MAG, order=ORDER)
    assert "bad-fit" in cls["categories"], cls
    assert cls["ok"] is False, f"明显拟合不上数据的模型不应被判为 ok：{cls}"


def test_bound_hits_flags_parameters_sitting_on_a_bound():
    """贴界判定（BOUND_REL_TOL）必须是"真贴住"才算，不能把正常解误报成贴界。"""
    on_bounds = np.array([1.0, 1.0, 1.0, 1.0, 1.0e5])  # R0/tau1/R2/tau2 恰好落在界上
    hits = bound_hits(on_bounds, ORDER)
    assert hits, "恰好落在上下界的参数向量必须报出贴界"
    assert all(isinstance(h, str) for h in hits)
    # 正常解（基线里的参考拟合）不应报贴界
    assert bound_hits(TRUE_PARAMS, ORDER) == [], (
        "真值参数不在任何界上，不得误报贴界"
    )
    assert BOUND_REL_TOL > 0