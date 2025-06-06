import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis_cmos_comparison.memory_data import (
    cap_colors,
    cap_labels,
    cap_vals,
    den_colors,
    den_labels,
    den_vals,
    tech_cap,
    tech_den,
)
from nmem.analysis.bar_plots import (
    draw_extruded_barh,
)
from nmem.analysis.plotting import set_inter_font, set_pres_style

set_inter_font()
set_pres_style()

# ---------------------- Create Figure ----------------------
fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=False)
fig.suptitle(
    "Superconducting vs. Semiconducting Memory Scaling Comparison",
    weight="bold",
    y=0.96,
)

# Plot 1: Capacity
draw_extruded_barh(
    axs[1],
    tech_cap,
    cap_vals,
    cap_colors,
    cap_labels,
    xlabel="Memory Capacity (log scale, bits/chip)",
    xticks=[np.log10(2**10), np.log10(2**20), np.log10(2**30), np.log10(2**40)],
    xticklabels=["1 kb", "1 Mb", "1 Gb", "1 Tb"],
)

# Plot 2: Density
draw_extruded_barh(
    axs[0],
    tech_den,
    den_vals,
    den_colors,
    den_labels,
    xlabel="Functional Density (log scale, bits/cm²)",
    xticks=[np.log10(1e6), np.log10(1e9), np.log10(1e12)],
    xticklabels=["1 Mb/cm²", "1 Gb/cm²", "1 Tb/cm²"],
)

plt.show()
