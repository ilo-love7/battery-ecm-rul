"""核心算法的合成数据测试 —— 不需要 NASA / MIT 原始数据。

这些测试的目的不是覆盖全部代码，而是守住三条最容易被无声破坏的约定：

1. ``fit_rc`` 在无噪声合成曲线上必须能**精确恢复**已知参数（辨识链条真的在工作）；
2. ``dq_curve`` / ``dq_cross_cycle`` 的**符号与量纲**必须符合其文档定义；
3. ``metrics`` 的中位 APE 必须与我们对外报告的数字同一口径。

运行：``pytest tests/ -q``（或 ``python -m pytest tests/ -q``）
"""
from __future__ import annotations

import numpy as np
import pytest

from src.ecm.ecm import fit_rc, model_voltage, params_to_dict
from src.features.dq import N_GRID, V_GRID, dq_cross_cycle, dq_curve, dq_paper_stats
from src.rul.models import metrics

# 合成曲线的真值参数（欧姆：Ohm，时间：s）
P_TRUE = [0.030, 0.020, 30.0, 0.010, 300.0]
CAP_AH = 1.1


def _synthetic_discharge(soc0: float = 0.0, soc1: float = 1.0, n: int = 4000):
    """一段恒流放电：SOC 线性下降，返回 (t, i_mag, V, soc)。"""
    t = np.linspace(0.0, 3600.0, n)
    soc = np.linspace(soc0, soc1, n)
    i_mag = CAP_AH * 3600.0 / 3600.0  # 1C
    return t, i_mag, soc


def _ocv_linear(soc):
    """OCV(SOC)：单调、有斜率，用来把辨识问题与 OCV 假设解耦。"""
    return 3.30 + 0.30 * np.asarray(soc, dtype=float)


class TestEcmIdentification:
    def test_recovers_known_parameters_without_noise(self):
        t, i_mag, soc = _synthetic_discharge()
        V = model_voltage(t, i_mag, _ocv_linear, soc, P_TRUE, order=2)

        res = fit_rc(t, i_mag, V, _ocv_linear, soc, order=2)

        assert np.allclose(res.x, P_TRUE, rtol=1e-3, atol=1e-5), (
            f"辨识结果偏离真值：{res.x} vs {P_TRUE}"
        )

    def test_robust_to_1mv_measurement_noise(self):
        t, i_mag, soc = _synthetic_discharge()
        V = model_voltage(t, i_mag, _ocv_linear, soc, P_TRUE, order=2)
        rng = np.random.default_rng(42)
        V_noisy = V + rng.normal(0.0, 1e-3, size=V.shape)  # 1 mV 量测噪声

        res = fit_rc(t, i_mag, V_noisy, _ocv_linear, soc, order=2)

        rmse_mv = float(np.sqrt(np.mean(res.fun ** 2))) * 1e3
        assert rmse_mv < 1.5, f"1 mV 噪声下复算 RMSE 应仍在 mV 量级，实际 {rmse_mv:.2f} mV"
        assert 0.9 * P_TRUE[0] < res.x[0] < 1.1 * P_TRUE[0]

    def test_r0_fixed_variant_returns_full_vector(self):
        t, i_mag, soc = _synthetic_discharge()
        V = model_voltage(t, i_mag, _ocv_linear, soc, P_TRUE, order=2)

        res = fit_rc(t, i_mag, V, _ocv_linear, soc, order=2, r0_fixed=P_TRUE[0])

        assert len(res.x) == 5
        assert res.x[0] == pytest.approx(P_TRUE[0])  # 被钉住
        d = params_to_dict(res.x, order=2)
        assert set(d) == {"R0", "R1", "C1", "R2", "C2"}
        assert d["C1"] == pytest.approx(30.0 / 0.020, rel=1e-3)


class TestDqCurve:
    def test_hysteresis_gap_spans_both_endpoints(self):
        """滞回口径 ΔQ = Qd(V) − Qc(V) 的量纲与端点方向。

        两条容量曲线按 _q_of_v 的约定**都以低压端为零点**：
        充电累计容量 Qc 在 V=2.0 处为 0、V=3.5 处为 1.05 Ah；
        放电累计容量 Qd 换算后在 V=2.0 处为 0、V=3.5 处为 1.10 Ah。
        于是 ΔQ(2.0 V) = 0（共同起点），ΔQ(3.5 V) = 1.10 − 1.05 = **+0.05 Ah**，
        即这一圈的净容量损失（库仑效率损失），中间线性过渡。
        """
        n = 200
        v = np.concatenate([np.linspace(3.5, 2.0, n), np.linspace(2.0, 3.5, n)])
        qc = np.maximum(v - 2.0, 0.0) / 1.5 * 1.05            # 充电：V 升、Q 增
        qd = (3.5 - v) / 1.5 * 1.10                           # 放电：V 降、Q 增
        cur = np.concatenate([np.full(n, -2.0), np.full(n, 2.0)])

        dq = dq_curve(v, cur, qc, qd, v_grid=V_GRID)

        assert dq.shape == (N_GRID,)
        assert not np.isnan(dq).any(), "覆盖完整的循环不应出现 NaN"
        assert dq[0] == pytest.approx(0.0, abs=2e-3), f"2.0 V 端点应为 0（共同起点），实际 {dq[0]}"
        assert dq[-1] == pytest.approx(0.05, abs=2e-3), f"3.5 V 端点应为净容量损失 0.05 Ah，实际 {dq[-1]}"
        assert np.all(np.diff(dq) > -1e-9), "两条容量曲线的斜率差恒定，ΔQ 应单调上升"

    def test_insufficient_coverage_returns_nan(self):
        v = np.linspace(3.4, 3.5, 50)           # 只覆盖栅格一角
        cur = np.full(50, -2.0)
        qc = np.zeros(50)
        qd = np.linspace(0, 0.1, 50)

        assert np.isnan(dq_curve(v, cur, qc, qd, v_grid=V_GRID)).all()

    def test_cross_cycle_is_late_minus_early(self):
        early = np.zeros(N_GRID)
        late = -np.linspace(0.0, 0.4, N_GRID)
        dq = dq_cross_cycle(early, late)
        assert (dq <= 0).all(), "论文口径 ΔQ 应恒为负"
        assert dq[-1] == pytest.approx(-0.4)
        assert np.isnan(dq_cross_cycle(None, late)).all()
        assert np.isnan(dq_cross_cycle(np.full(N_GRID, np.nan), late)).all()

    def test_paper_stats_depth_tracks_valley(self):
        flat = np.full(N_GRID, -0.2)
        deeper = np.full(N_GRID, -0.2)
        deeper[N_GRID // 2] = -0.6              # 挖一个更深的谷
        # xq_log_min = log10(谷深)：谷越深取值越大
        assert dq_paper_stats(deeper)["xq_log_min"] > dq_paper_stats(flat)["xq_log_min"]
        assert dq_paper_stats(deeper)["xq_log_min"] == pytest.approx(np.log10(0.6))


class TestMetrics:
    def test_median_ape_matches_reported_definition(self):
        y_true = np.array([100.0, 200.0, 400.0])
        y_pred = np.array([110.0, 180.0, 404.0])
        m = metrics(y_true, y_pred)
        # 中位 APE = median(10%, 10%, 1%) = 10%
        assert m["median_ape"] == pytest.approx(10.0, abs=1e-6)
        assert m["mean_ape"] == pytest.approx(np.mean([10.0, 10.0, 1.0]))
        assert m["max_ape"] == pytest.approx(10.0)

    def test_perfect_prediction_gives_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert metrics(y, y.copy())["median_ape"] == pytest.approx(0.0)
        assert metrics(y, y.copy())["r2"] == pytest.approx(1.0)
