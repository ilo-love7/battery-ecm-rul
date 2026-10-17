"""PyBaMM 物理仿真（阶段 D / 候选清单 P2）。

本子包把 PyBaMM 封装成项目内的可复用组件，与阶段 A（ECM 辨识）、
阶段 C（RUL 预测）共用同一套数据口径：

- ``models``    模型与参数集装配（SPM / DFN / Thevenin-ECM）
- ``protocols`` 快充协议族定义与 Severson 风格的多步 CC 结构
- ``runner``    单次实验的运行、曲线抽取、寿命判定

**pybamm 是可选依赖**（``pip install -r requirements-full.txt``）。
所有入口都在函数内部延迟导入，未安装时导入本包不会报错，
只有真正调用仿真函数才会抛出带安装提示的 ImportError。
"""
from __future__ import annotations

__all__ = ["models", "protocols", "runner"]
