from __future__ import annotations

import numpy as np

from geometric_median_math import find_geometric_median
from visualizer import plot_result


def main() -> None:
    points: np.ndarray = np.ascontiguousarray(
        [
            (0.7621083180927701, -0.14055282440529338),
            (0.11712191696366236, 0.24064185122105997),
            (0.19210823899069684, -0.4415048868269591),
            (0.1638517587277829, 0.8691878070815509),
        ],
        dtype=np.float64,
    )

    median: np.ndarray = find_geometric_median(points)
    print(f"Geometric median: ({median[0]:.10f}, {median[1]:.10f})")
    plot_result(points, median, output_path="data/output/result.png")


if __name__ == "__main__":
    main()
