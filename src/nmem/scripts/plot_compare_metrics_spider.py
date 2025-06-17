import matplotlib.pyplot as plt

from nmem.analysis.memory_data import (
    axis_max,
    axis_min,
    datasets,
    labels,
    metrics,
    normalizers,
    styles,
    units,
)
from nmem.analysis.spider_plots import (
    plot_radar,
)

# --- Call the updated plot function ---
plot_radar(
    metrics,
    units,
    axis_min,
    axis_max,
    normalizers,
    datasets,
    labels,
    styles,
)
plt.show()