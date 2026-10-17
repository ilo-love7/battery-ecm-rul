.PHONY: help all test data ecm rul sim audit dashboard app check clean

PY ?= python

help:
	@echo "Battery ECM-RUL — 常用命令"
	@echo ""
	@echo "  make data       原始 .mat -> 统一 Parquet / NPZ（需先放好 data/raw/）"
	@echo "  make ecm        ECM 参数辨识 + 内阻老化轨迹"
	@echo "  make rul        RUL 建模、评估、出图（约 4 分钟）"
	@echo "  make sim        PyBaMM 物理仿真 + 与 RUL 的结合（约 2 分钟，需 pybamm）"
	@echo "  make test       单元测试（合成数据，约 2 秒，需 pytest）"
	@echo "  make audit      P3 辨识审计 + 三线桥接（约 2 分钟，需 pybamm）"
	@echo "  make all        data + ecm + rul 全流程"
	@echo "  make dashboard  生成离线 HTML 数据看板"
	@echo "  make app        启动 Streamlit 小工具"
	@echo "  make check      语法自检（不依赖数据）"
	@echo "  make clean      清理中间产物"

all: data ecm rul

data:
	$(PY) scripts/phase_b_build_dataset.py all

ecm:
	$(PY) scripts/phase_a2_ecm.py

rul:
	$(PY) scripts/phase_c_rul.py

sim:
	$(PY) scripts/phase_d1_pybamm_sim.py
	$(PY) scripts/phase_d2_sim_rul.py

test:
	$(PY) -m pytest tests/ -q

audit:
	$(PY) scripts/phase_e_p3_audit.py
	$(PY) scripts/phase_f_bridge.py

dashboard:
	$(PY) scripts/build_dashboard.py

app:
	streamlit run src/app/app.py

check:
	$(PY) -m compileall -q src scripts
	$(PY) -c "from src.parsers import nasa, mit; from src.features.dq import dq_curve; from src.rul import models; from src.sim import models as sim; from src.sim.protocols import PROTOCOLS; print('imports OK', len(PROTOCOLS), 'protocols')"

clean:
	rm -rf data/processed/*.parquet data/processed/*.npz
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
