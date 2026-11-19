"""
DRT 结果对比图: DearEIS / DRTtools / EIS Analysis Tool

读取 tests/Result compared 下 3 个 CSV (tau, g), 以 semilogx 叠加绘制,
输出 PNG 至同目录。
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Result compared")

FILES = [
    ("DearEIS.csv", "DearEIS", "#2962ff"),
    ("DRTtools.csv", "DRTtools", "#c62828"),
    ("EIS Analysis Tool.csv", "EIS Analysis Tool", "#00897b"),
]


def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    for fname, label, color in FILES:
        path = os.path.join(DATA_DIR, fname)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        tau, g = data[:, 0], data[:, 1]
        ax.semilogx(tau, g, "-", color=color, lw=2, label=label)

    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(r"$g(\tau)$ ($\Omega$/ln$\tau$)")
    ax.set_title("DRT Result Comparison (DearEIS / DRTtools / EIS Analysis Tool)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(framealpha=0.8)

    out = os.path.join(DATA_DIR, "DRT_Result_Comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
