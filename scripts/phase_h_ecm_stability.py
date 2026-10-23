"""阶段 H（E1）：ECM 拟合稳定性回归基线的生成 / 校验入口。

离线可跑：只用合成数据 + 仓内已提交的 `reports/phaseA2_ecm_results.csv`，
**不需要** NASA / MIT 原始数据集。

用法::

    python scripts/phase_h_ecm_stability.py            # 重新生成 tests/regression_baseline.json
    python scripts/phase_h_ecm_stability.py --check     # 只比对，漂移超阈值 → 退出码 1

本脚本只写 `tests/regression_baseline.json`（或 --out 指定的路径），
不触碰任何既有脚本的产物路径（reports/、figures/、data/processed/）。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.ecm import stability  # noqa: E402


if __name__ == "__main__":
    sys.exit(stability.main())