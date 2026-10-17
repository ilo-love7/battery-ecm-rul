"""快充协议族定义。

Severson et al. 2019 的充电协议是「多步恒流 + 恒压收尾」：

    C1(Q1) - C2(Q2) - C3(Q3)     例如 4.8C(80%)-3.6C(10%)-3.2C(10%)

含义是「先以 C1 充到 Q1 的荷电量，再以 C2 充 Q2，最后 C3 充 Q3，
末端恒压至截止电流」。本模块用 ``(C 倍率, 荷电比例)`` 的元组序列表达，
充电时长由 ``Δt = Q_i / (C_i · Q_nom)`` 直接换算，因此**与电池容量无关**。

协议族里特意放了一对「充电时间完全相同、快充段位置相反」的协议
（``place_low`` / ``place_high``），用于检验一个关键问题：

    充电时间是否足以决定寿命？

MIT 实测数据给出的答案是「否」——124 颗电池早期平均充电时长与循环寿命的
Spearman 相关仅 −0.05（见 ``scripts/phase_d2_sim_rul.py``）。若仿真也能在
充电时间相同的条件下复现出显著的寿命差，就说明真正起作用的不是「快」，
而是「快在哪一段」，这正是 ΔQ(V) 按电压分辨所编码的信息。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    """一条多步恒流充电协议。"""

    key: str
    label: str
    segments: tuple[tuple[float, float], ...]  # (C 倍率, 荷电比例)，比例之和应为 1
    note: str = ""

    @property
    def nominal_minutes(self) -> float:
        """名义充电时长（分钟），仅计恒流段。"""
        return sum(frac / c_rate for c_rate, frac in self.segments) * 60.0

    @property
    def max_c(self) -> float:
        return max(c for c, _ in self.segments)

    @property
    def mean_c(self) -> float:
        total = sum(frac / c for c, frac in self.segments)
        return 1.0 / total if total > 0 else float("nan")

    def charge_steps(self, chemistry: str = "lfp", scale: float = 1.0) -> list[str]:
        """转成 PyBaMM Experiment 的充电步骤字符串。

        **每一段恒流都带电压上限**（``for X minutes or until V_max``），
        而不是只有末段带。原因有三条，都由实测逼出来：

        1. 容量会随老化下降。若全程按时长充，电池衰减后必然过充，
           而真实充电器的末段本来就是「充到截止电压」；
        2. 激进协议在给定电芯上**根本执行不完**。``6C(60%)`` 在 Chen2020
           （5 Ah NMC811）上从空电开始充，端电压在第 5.4 min 前就冲到
           **4.68 V**，远超 4.2 V 上限，下一步直接 ``EmptySolution``；
        3. 电压上限本身就是真实充电器的行为，不是数值技巧。

        加上限之后，协议是否被截断、截断在多少分钟，本身就是结论：
        ``6C(60%)`` 在 5 Ah 电芯上只能执行到约 2.7 min（≈30% 荷电量），
        名义上的「6C 充到 60%」是一个**不可执行的协议**。

        ``scale`` 把「占额定容量的比例」折算成「占实际放电容量（1C）的比例」：
        参数集的额定容量大于 1C 放电实际放出的容量（Prada2013 额定 2.3 Ah、
        1C 放电 1.94 Ah），不折算会让首段就充过头。
        """
        from .models import V_WINDOW

        v_max = V_WINDOW[chemistry][1]
        out = []
        last = len(self.segments) - 1
        for j, (c_rate, frac) in enumerate(self.segments):
            if j == last:
                out.append(f"Charge at {c_rate:g}C until {v_max:g} V")
            else:
                minutes = frac * scale / c_rate * 60.0
                out.append(f"Charge at {c_rate:g}C for {minutes:.2f} minutes "
                           f"or until {v_max:g} V")
        return out

    def describe(self) -> str:
        return "-".join(f"{c:g}C({frac * 100:.0f}%)" for c, frac in self.segments)


#: 协议族：覆盖「恒流基线 → 超快充 → Severson 多步 → 等时长换位」四类
PROTOCOLS: tuple[Protocol, ...] = (
    Protocol("base_1c", "1C 恒流（基线）", ((1.0, 1.0),),
             "不施加任何快充，作为寿命参照"),
    Protocol("cc_3c", "3C 恒流", ((3.0, 1.0),),
             "单段快充，验证「越快越伤」的单调性"),
    Protocol("sev_3_6", "3.6C(80%)-3.6C(10%)-3.6C(10%)", ((3.6, 0.8), (3.6, 0.1), (3.6, 0.1)),
             "Severson 数据集里的恒 3.6C 基准协议"),
    Protocol("sev_4_8", "4.8C(80%)-3.6C(10%)-3.2C(10%)", ((4.8, 0.8), (3.6, 0.1), (3.2, 0.1)),
             "Severson 论文的激进快充原型"),
    Protocol("sev_6_0", "6C(60%)-3C(40%)", ((6.0, 0.6), (3.0, 0.4)),
             "低 SOC 段超快充 + 高 SOC 段降速，Severson 的最优类协议"),
    Protocol("place_low", "3C(20%)-1C(80%)", ((3.0, 0.2), (1.0, 0.8)),
             "快充放在低 SOC 段；名义时长 52 min"),
    Protocol("place_high", "1C(80%)-3C(20%)", ((1.0, 0.8), (3.0, 0.2)),
             "快充放在高 SOC 段；名义时长同为 52 min，与 place_low 严格等时长"),
)

BY_KEY = {p.key: p for p in PROTOCOLS}

#: 严格等时长的协议对（名义充电时长完全相同）
EQUAL_TIME_PAIRS = (("place_low", "place_high"),)


def discharge_step(chemistry: str = "lfp", c_rate: float = 1.0) -> str:
    """统一放电步骤：恒流放到下截止电压。"""
    from .models import V_WINDOW

    v_min = V_WINDOW[chemistry][0]
    return f"Discharge at {c_rate:g}C until {v_min:g} V"


def cv_step(chemistry: str = "lfp", c_rate: float = 0.05) -> str:
    """恒压收尾：充到上截止电压后恒压至 C/20。"""
    from .models import V_WINDOW

    v_max = V_WINDOW[chemistry][1]
    return f"Hold at {v_max:g} V until C/{1 / c_rate:.0f}"


def experiment_steps(protocol: Protocol, chemistry: str = "lfp",
                     rest: str = "Rest for 5 minutes",
                     scale: float = 1.0) -> list[str]:
    """一个完整循环的步骤序列：放电 → 多步充电 → 恒压收尾 → 静置。"""
    return ([discharge_step(chemistry)]
            + protocol.charge_steps(chemistry, scale)
            + [cv_step(chemistry), rest])
