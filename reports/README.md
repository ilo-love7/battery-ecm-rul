# reports/

脚本产出的指标表与小结。所有数字由 `scripts/` 下的脚本在种子 42 下生成，可重跑复现。

| 文件 | 内容 | 生成脚本 |
|---|---|---|
| `phaseA2_ecm_results.csv` | ECM 辨识参数与复测 RMSE（4 颗 NASA 电池） | `phase_a2_ecm.py` |
| `phaseA2_aging.csv` | 逐圈 R0 / R_dc(50%) / 容量 / SOH 老化轨迹 | `phase_a2_ecm.py` |
| `phaseA_ecm_results.csv` | 初版 ECM 结果（仅 B0005 / B0006，保留作对照） | `phase_a_ecm.py` |
| `phaseC_metrics.csv` | 五类模型 × 两个测试集的完整指标 | `phase_c_rul.py` |
| `phaseC_cv.csv` | 全量 124 颗 5 折交叉验证逐折结果 | `phase_c_rul.py` |
| `phaseC_predictions.csv` | 每颗电池 5 个模型的预测值 | `phase_c_rul.py` |
| `phaseC_learning_curve.csv` | 训练电池数 10→41 的学习曲线 | `phase_c_rul.py` |
| `phaseC_results.json` | 上述指标的汇总快照（含特征数、ElasticNet alpha） | `phase_c_rul.py` |
| `phaseC_summary.md` | RUL 方法与结果小结 | 手写 |

## 快速查看

```bash
python -c "import pandas as pd; print(pd.read_csv('reports/phaseC_metrics.csv').round(3))"
```

完整的表格与解读见 [`../docs/results.md`](../docs/results.md)。
