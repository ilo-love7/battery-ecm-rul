"""阶段 E：Streamlit 数据小工具（P7）。

功能：
1. 数据集概览：MIT 124 颗 / NASA 34 颗，寿命分布、SOH 衰减族谱；
2. 单电池体检：SOH 曲线、容量/内阻/充电时间演化；
3. ECM 参数：四颗 NASA 电池的 R0/R_dc/R_pol 与复测 RMSE；
4. RUL 预测：主测试集预测 vs 真实，可查看每颗电池的误差。

运行：
    streamlit run src/app/app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"

st.set_page_config(page_title="锂电池 ECM + RUL 工具", page_icon="🔋", layout="wide")


@st.cache_data
def load_mit():
    cells = pd.read_parquet(PROC / "mit_cells.parquet")
    cycles = pd.read_parquet(PROC / "mit_cycles.parquet")
    return cells, cycles


@st.cache_data
def load_nasa():
    return pd.read_parquet(PROC / "nasa_cycles.parquet"), pd.read_parquet(PROC / "nasa_cells.parquet")


@st.cache_data
def load_preds():
    p = REP / "phaseC_predictions.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_ecm():
    p = REP / "phaseA2_ecm_results.csv"
    return pd.read_csv(p) if p.exists() else None


def main() -> None:
    st.title("🔋 锂电池等效电路建模 + RUL 寿命预测")
    st.caption("NASA PCoE / MIT-Stanford（Severson 2019）· 数据小工具 P7")

    mit_cells, mit_cycles = load_mit()
    nasa_cycles, nasa_cells = load_nasa()

    with st.sidebar:
        st.header("数据选择")
        dataset = st.radio("数据集", ["MIT-Stanford（124 颗）", "NASA PCoE（34 颗）"])
        if dataset.startswith("MIT"):
            cell = st.selectbox("电池", mit_cells["cell_id"].tolist())
            info = mit_cells.set_index("cell_id").loc[cell]
            st.metric("循环寿命", f"{int(info['cycle_life'])} 圈")
            st.metric("划分", {"train": "训练集", "test": "主测试集",
                               "secondary_test": "二级测试集"}[info["split"]])
        else:
            cell = st.selectbox("电池", nasa_cells["cell_id"].tolist())
            info = nasa_cells.set_index("cell_id").loc[cell]
            st.metric("放电循环数", f"{int(info['n_discharge'])}")
            st.metric("首圈容量", f"{info['q0']:.3f} Ah")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据集概览", "🔍 单电池体检", "⚙️ ECM 参数", "📈 RUL 预测"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("MIT 电池数", len(mit_cells))
        c2.metric("MIT 循环记录", f"{len(mit_cycles):,}")
        c3.metric("NASA 放电循环", f"{int((nasa_cycles['type'] == 'discharge').sum()):,}")
        st.subheader("循环寿命分布（MIT）")
        hist, edges = np.histogram(mit_cells["cycle_life"], bins=25)
        st.bar_chart(pd.DataFrame({"寿命区间": [f"{int(e)}" for e in edges[:-1]], "电池数": hist})
                     .set_index("寿命区间"))
        st.subheader("SOH 衰减族谱（MIT，按寿命着色）")
        pivot = mit_cycles.pivot_table(index="cycle", columns="cell_id", values="SOH")
        st.line_chart(pivot.iloc[:800])

    with tab2:
        st.subheader(f"{cell} — 老化轨迹")
        if dataset.startswith("MIT"):
            d = mit_cycles[mit_cycles["cell_id"] == cell].sort_values("cycle")
            st.line_chart(d.set_index("cycle")[["SOH"]])
            st.line_chart(d.set_index("cycle")[["IR", "chargetime"]])
        else:
            d = nasa_cycles[(nasa_cycles["cell_id"] == cell) & (nasa_cycles["type"] == "discharge")]
            st.line_chart(d.set_index("cycle_id")[["SOH"]])
            st.line_chart(d.set_index("cycle_id")[["discharge_capacity_Ah"]])
        st.dataframe(d.tail(20), use_container_width=True, height=260)

    with tab3:
        ecm = load_ecm()
        if ecm is None:
            st.info("尚未运行 scripts/phase_a2_ecm.py")
        else:
            st.subheader("ECM 参数辨识结果")
            st.dataframe(ecm, use_container_width=True)
            st.caption("R0 由放电电流阶跃压降求得；R_dc(SOC) 由恒流区逐点求取；"
                       "复测 RMSE 为「同工况第 3 次放电」验证结果。")

    with tab4:
        preds = load_preds()
        if preds is None:
            st.info("尚未运行 scripts/phase_c_rul.py")
        else:
            st.subheader("主测试集 / 二级测试集预测")
            cols = [c for c in preds.columns if c.startswith("pred_")]
            show = preds[["cell_id", "split", "cycle_life"] + cols].copy()
            show["最佳预测"] = show[cols].median(axis=1)
            show["误差 %"] = (show["最佳预测"] - show["cycle_life"]).abs() / show["cycle_life"] * 100
            show = show.rename(columns={"cycle_life": "真实寿命", "split": "划分"})
            st.dataframe(show.round(2), use_container_width=True, height=380)
            st.download_button("下载预测明细 CSV", show.to_csv(index=False).encode("utf-8-sig"),
                               "rul_predictions.csv", "text/csv")
            test = show[show["划分"] == "test"]
            st.metric("主测试集中位误差", f"{test['误差 %'].median():.1f}%")


if __name__ == "__main__":
    main()
