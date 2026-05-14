from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_result(points: np.ndarray, median: np.ndarray, output_path: str) -> None:
    p: np.ndarray = np.ascontiguousarray(points, dtype=np.float64)
    m: np.ndarray = np.ascontiguousarray(median, dtype=np.float64)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(p[:, 0], p[:, 1], color="blue", label="Source points")
    plt.scatter(
        m[0],
        m[1],
        color="red",
        s=180,
        marker="o",
        edgecolors="red",
        linewidths=0.8,
        label="Geometric median",
        zorder=3,
    )
    plt.title("Geometric Median")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
