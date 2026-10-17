"""PyBaMM 模型与参数集装配。

三套模型对应三种物理保真度，构成本项目的「模型谱系」：

===========  ==========================================  ==============
模型          假设                                        典型单次求解
===========  ==========================================  ==============
Thevenin     一阶 RC 等效电路，无电化学过程                 ~0.05 s
SPM          单粒子模型：球形颗粒扩散 + Butler-Volmer     ~0.1  s
DFN          全阶 Doyle-Fuller-Newman：电解液输运 + 颗粒    ~0.3  s
===========  ==========================================  ==============

参数集选择（**化学体系必须与数据对齐**）：

- ``lfp`` → ``Prada2013``（A123 2.3 Ah LFP/石墨 18650）。与 NASA PCoE、
  MIT-Stanford 两套数据集同为磷酸铁锂体系，用于电压曲线复现与 ΔQ(V) 对照。
  该参数集**不含 SEI 参数**，做老化仿真时由 :func:`_with_sei` 从 Chen2020
  补齐（SEI 的本征参数如摩尔体积、电阻率、溶剂扩散系数与正极化学无关）。
- ``nmc`` → ``Chen2020``（5 Ah NMC811/SiOx-石墨）。SEI、热参数、集流体几何
  一应俱全，用于产热与快充协议的老化对比。

⚠ 化学体系不可混用：NMC 的电压窗口是 3.0–4.2 V，LFP 是 2.0–3.5 V，
两者的 ΔQ(V) 曲线不能直接逐点相减，只能比较形态（见 ``scripts/phase_d2``）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 化学体系 -> PyBaMM 参数集名
PARAM_SETS = {"lfp": "Prada2013", "nmc": "Chen2020", "nmc_plate": "OKane2022"}

#: 老化参数（SEI）的借用来源：Chen2020 的 SEI 参数最完整
SEI_SOURCE = "Chen2020"

#: 各化学体系的电压窗口（V）
V_WINDOW = {"lfp": (2.0, 3.5), "nmc": (3.0, 4.2), "nmc_plate": (2.5, 4.2)}

#: 阶段 A 在 NASA B0005 上辨识出的一阶 RC 参数（50% SOC，25 ℃）
STAGE_A_ECM = {"R0_ohm": 0.1073, "R1_ohm": 0.0211, "tau_s": 30.0}


def _pb():
    """延迟导入 pybamm，未安装时给出可操作的提示。"""
    try:
        import pybamm
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "本模块需要 pybamm。请执行：\n"
            "    pip install -r requirements-full.txt\n"
            "或单独安装：pip install pybamm"
        ) from exc
    return pybamm


def _with_sei(params, scale: float = 1.0):
    """给缺少 SEI 参数的参数集（如 Prada2013）补齐 SEI 参数。

    ``scale`` 缩放「SEI 溶剂扩散系数」以标定老化速率：默认参数下
    SEI 增长极慢（300 圈容量损失 <0.1%），需放大才能落在真实电池的
    寿命量级（150–2200 圈）。标定方法见 ``scripts/phase_d2_sim_rul.py``。
    """
    pb = _pb()
    src = pb.ParameterValues(SEI_SOURCE)
    keys = [k for k in src.keys() if "SEI" in k or "solvent" in k.lower()]
    for k in keys:
        if k not in params.keys():
            params[k] = src[k]
    if scale != 1.0:
        params["SEI solvent diffusivity [m2.s-1]"] = (
            params["SEI solvent diffusivity [m2.s-1]"] * scale
        )
    return params


def load_params(chemistry: str = "lfp", sei_scale: float | None = None):
    """载入参数集。

    Parameters
    ----------
    chemistry : {"lfp", "nmc"}
    sei_scale : float, optional
        给定时补 SEI 参数并按此倍数缩放 SEI 扩散系数（开启老化仿真）。
        ``None`` 表示不引入 SEI，纯电化学/等效电路仿真。
    """
    pb = _pb()
    if chemistry not in PARAM_SETS:
        raise ValueError(f"chemistry 只能是 {list(PARAM_SETS)}，收到 {chemistry!r}")
    params = pb.ParameterValues(PARAM_SETS[chemistry])
    if sei_scale is not None:
        params = _with_sei(params, sei_scale)
    return params


def build_model(kind: str = "spm", sei: bool = False, thermal: bool = False,
                plating: bool = False):
    """构建 PyBaMM 模型。

    Parameters
    ----------
    kind : {"spm", "dfn", "thevenin"}
    sei : bool
        是否加入 SEI 生长（仅 spm/dfn 有效）。
    thermal : bool
        是否使用集总热模型（需参数集含集流体几何，LFP 参数集不支持）。
    plating : bool
        是否加入不可逆析锂。需要含析锂参数的参数集（``nmc_plate`` / OKane2022），
        这是让「充电协议」真正影响寿命的关键机制——纯 SEI 生长由时间主导，
        协议差异几乎体现不出来（见 ``docs/sim.md``）。
    """
    pb = _pb()
    options: dict[str, str] = {}
    if sei:
        options["SEI"] = "solvent-diffusion limited"
    if plating:
        options["lithium plating"] = "irreversible"
    if thermal:
        options["thermal"] = "lumped"

    kind = kind.lower()
    if kind == "spm":
        return pb.lithium_ion.SPM(options or None)
    if kind == "dfn":
        return pb.lithium_ion.DFN(options or None)
    if kind == "thevenin":
        return pb.equivalent_circuit.Thevenin(options or None)
    raise ValueError(f"kind 只能是 spm/dfn/thevenin，收到 {kind!r}")


class OCVTable:
    """把 (SOC, OCV) 表包装成 PyBaMM 可用的参数函数。

    PyBaMM 的 ``Open-circuit voltage [V]`` 参数在符号代入阶段会拿到一个
    ``pybamm.Symbol``，而不是 float，因此不能直接 ``np.interp``；
    这里按输入类型分流：数值走插值，符号走 :class:`pybamm.Interpolant`。
    """

    def __init__(self, soc: np.ndarray, ocv: np.ndarray):
        self.soc = np.asarray(soc, dtype=float)
        self.ocv = np.asarray(ocv, dtype=float)

    def __call__(self, x):
        if isinstance(x, (int, float, np.ndarray)):
            return np.interp(x, self.soc, self.ocv)
        pb = _pb()
        return pb.Interpolant(self.soc, self.ocv, x, name="ocv")


_OCV_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def ocv_from_spm(chemistry: str = "lfp") -> tuple[np.ndarray, np.ndarray]:
    """用 SPM 的 C/50 准平衡放电反推 OCV(SOC)。

    ECM 需要一条 OCV-SOC 曲线，而 ``Prada2013`` 只给端点值。与其凭空拟合，
    不如让高保真模型自己给出：C/50 放电的端电压已非常接近开路电压，
    由此得到的 OCV 表既保留了 LFP 的电压平台，也让「SPM → ECM」的降阶
    链路有据可依（见 ``docs/sim.md``）。
    """
    if chemistry in _OCV_CACHE:
        return _OCV_CACHE[chemistry]
    pb = _pb()
    params = load_params(chemistry)
    v_min = V_WINDOW[chemistry][0]
    exp = pb.Experiment([f"Discharge at C/50 until {v_min:g} V"], period="5 minutes")
    sim = pb.Simulation(build_model("spm"), experiment=exp, parameter_values=params)
    sol = sim.solve()
    V = np.asarray(sol["Voltage [V]"].entries, dtype=float)
    Q = np.abs(np.asarray(sol["Throughput capacity [A.h]"].entries, dtype=float))
    soc = 1.0 - Q / Q.max()
    order = np.argsort(soc)
    soc, V = soc[order], V[order]
    keep = np.concatenate([[True], np.diff(soc) > 1e-6])
    soc, V = soc[keep], V[keep]
    _OCV_CACHE[chemistry] = (soc, V)
    return soc, V


def thevenin_params(
    chemistry: str = "lfp",
    r0: float | None = None,
    r1: float | None = None,
    tau: float = STAGE_A_ECM["tau_s"],
):
    """用阶段 A 辨识的 R0/R1/τ 装配 Thevenin 参数。

    PyBaMM 的 Thevenin 需要 ``R0 [Ohm]``、``R1 [Ohm]``、``C1 [F]``、
    ``Cell capacity [A.h]`` 与 ``Open-circuit voltage [V]``。前三者取自
    ``reports/phaseA2_ecm_results.csv``（NASA B0005，50% SOC 工况），
    OCV 由 :func:`ocv_from_spm` 给出，容量沿用参数集默认值。

    ``C1`` 由 ``τ / R1`` 反算：阶段 A 的拟合直接给出 τ，未单独存 C1。
    """
    pb = _pb()
    params = load_params(chemistry)
    model = pb.equivalent_circuit.Thevenin()
    # Thevenin 自带一组通用默认参数（热质量、熵热项等），化学参数集里没有，
    # 缺哪个补哪个，避免逐个硬编码。
    for k, v in model.default_parameter_values.items():
        if k not in params.keys():
            params[k] = v

    soc, ocv = ocv_from_spm(chemistry)
    params["Open-circuit voltage [V]"] = OCVTable(soc, ocv)
    params["Cell capacity [A.h]"] = params["Nominal cell capacity [A.h]"]
    r0 = STAGE_A_ECM["R0_ohm"] if r0 is None else r0
    r1 = STAGE_A_ECM["R1_ohm"] if r1 is None else r1
    params["R0 [Ohm]"] = r0
    params["R1 [Ohm]"] = r1
    params["C1 [F]"] = tau / r1
    # 初始 SOC 取 0.99：1.0 会直接触发 "Maximum SoC" 事件导致求解失败；
    # 熵热项置 0（恒温假设，ECM 不含热耦合）。
    params["Initial SoC"] = 0.99
    params["Entropic change [V/K]"] = 0.0
    params["Element-1 initial overpotential [V]"] = 0.0
    return params


_CAP_CACHE: dict[tuple[str, float], float] = {}


def delivered_capacity(chemistry: str = "lfp", c_rate: float = 1.0) -> float:
    """某一倍率下满充后放电到下限实际放出的容量（Ah）。

    参数集的 ``Nominal cell capacity`` 往往大于实际放电容量
    （Prada2013 额定 2.3 Ah，1C 放电只放出 1.94 Ah）。协议按「占额定容量的
    比例」定义，若不折算，充电段会在首段就充过头、触发上限电压。
    """
    key = (chemistry, float(c_rate))
    if key in _CAP_CACHE:
        return _CAP_CACHE[key]
    pb = _pb()
    params = load_params(chemistry)
    v_min = V_WINDOW[chemistry][0]
    exp = pb.Experiment([f"Discharge at {c_rate:g}C until {v_min:g} V"], period="5 minutes")
    sol = pb.Simulation(build_model("spm"), experiment=exp, parameter_values=params).solve()
    cap = float(np.abs(np.asarray(sol["Throughput capacity [A.h]"].entries)[-1]))
    _CAP_CACHE[key] = cap
    return cap


def capacity_scale(chemistry: str = "lfp", c_rate: float = 1.0,
                   safety: float = 0.97) -> float:
    """额定容量 → 该倍率实际放电容量的折算系数（留 ``safety`` 余量）。"""
    pb = _pb()
    nominal = float(pb.ParameterValues(PARAM_SETS[chemistry])["Nominal cell capacity [A.h]"])
    return safety * delivered_capacity(chemistry, c_rate) / nominal


@dataclass(frozen=True)
class SimResult:
    """一次实验仿真的抽取结果。"""

    name: str
    cap: np.ndarray            # 每圈容量 / Ah
    lli: np.ndarray            # 每圈锂库存损失 / %
    t_charge: np.ndarray       # 每圈充电总时长（含恒压段）/ s
    t80: np.ndarray            # 每圈充到 80% 荷电量的时长 / s
    tmax: np.ndarray           # 每圈最高温度 / K（无热模型时为 NaN）
    qd_early: np.ndarray       # 第 10 圈放电容量曲线 Q(V)，自高压端起算
    qd_late: np.ndarray        # 第 100 圈（不足时取末圈）同口径曲线
    life: float                # 至 80% 容量的圈数；未衰减到则为 NaN
    n_cycles: int
    seconds: float             # 墙钟耗时
