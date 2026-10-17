"""数据自检：读取 NASA B0005，打印循环类型统计并绘制首次放电的 V-t 曲线。

用途：确认原始数据已正确落盘、解析器可正常读取。
    python scripts/validate_nasa.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsers import nasa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def find_cell(cell_id: str = "B0005"):
    nasa_dir = os.environ.get("NASA_DIR") or os.path.join(ROOT, "data", "raw", "nasa")
    for p in nasa.all_nasa_files(nasa_dir):
        if p.stem == cell_id:
            return p
    raise SystemExit(
        f"[错误] 找不到 {cell_id}.mat（搜索目录：{nasa_dir}）。\n"
        f"       请先按 data/README.md 下载 NASA PCoE 数据集。"
    )


def main() -> int:
    path = find_cell("B0005")
    name, battery = nasa.load_battery(path)
    print(f"cell: {name}")

    # cycle-type census
    from collections import Counter

    census = Counter(nasa.cycle_type(cc) for _, cc in nasa.iter_cycles(battery))
    print("cycle-type census:", dict(census))

    # first discharge cycle
    for i, cc in nasa.iter_cycles(battery):
        if nasa.cycle_type(cc) == "discharge":
            curve = nasa.discharge_curve(cc)
            cap = nasa.discharge_capacity(cc)
            t, V, I = curve["t"], curve["V"], curve["I"]
            print(f"first discharge: cycle_id={i}, points={len(V)}, capacity={cap:.4f} Ah")
            print(f"  V range = [{V.min():.3f}, {V.max():.3f}] V, t range = [{t.min():.1f}, {t.max():.1f}] s")

            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax[0].plot(t, V, color="tab:blue", lw=1.2)
            ax[0].set_ylabel("Voltage (V)")
            ax[0].set_title(f"NASA {name} — discharge cycle #{i} (capacity {cap:.4f} Ah)")
            ax[0].grid(alpha=0.3)
            ax[1].plot(t, I, color="tab:red", lw=1.2)
            ax[1].set_ylabel("Current (A)")
            ax[1].set_xlabel("Time (s)")
            ax[1].grid(alpha=0.3)
            fig.tight_layout()
            out = os.path.join(FIG_DIR, "fig_a0_discharge_vt.png")
            fig.savefig(out, dpi=130)
            print(f"saved figure: {out}")
            break

    # summary table sanity
    summ = nasa.build_summary(battery, name)
    n_discharge = int((summ["type"] == "discharge").sum())
    print(f"summary rows={len(summ)}, discharge cycles={n_discharge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
