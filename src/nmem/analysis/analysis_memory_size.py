import matplotlib.pyplot as plt
import numpy as np
from nmem.analysis.analysis import set_inter_font
set_inter_font()
# ---------------------- Presentation Style Settings ----------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
})
import matplotlib.font_manager as fm

matches = [f.name for f in fm.fontManager.ttflist if "Helvetica" in f.name]

# ---------------------- Data: Memory Capacity ----------------------
technologies_cap = ["CMOS DRAM", "CMOS SRAM", "Josephson RAM", "This Work (SNM)"]
capacities_bits = [2**43, 2**26, 2**12, 64]
log_capacities = np.log10(capacities_bits)
colors_cap = ["gray", "gray", "red", "royalblue"]
labels_cap = ["~8 TB", "~64 MB", "4 kbit", "64 bit"]

# ---------------------- Data: Functional Density ----------------------
technologies_density = ["CMOS DRAM", "CMOS SRAM", "Josephson RAM", "This Work (SNM)"]
densities = [1e11, 1e9, 1e6, 2.6e6]
log_densities = np.log10(densities)
colors_density = ["gray", "gray", "red", "royalblue"]
labels_density = ["~100 Gbit/cm²", "~1 Gbit/cm²", "1 Mbit/cm²", "2.6 Mbit/cm²"]

# ---------------------- Create Plot ----------------------
fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=False)
fig.suptitle("Cryogenic-Compatible Memory Scaling Comparison", weight="bold", y=0.9)

# ---------------------- Plot 1: Memory Capacity ----------------------
ax = axs[0]
bars = ax.barh(technologies_cap, log_capacities, color=colors_cap, height=0.7)

# Invert for descending order
ax.invert_yaxis()

# Add text labels inside bars
for bar, label, tech in zip(bars, labels_cap, technologies_cap):
    width = bar.get_width()
    ha = "right"
    color = "white"
    ax.text(width - 0.2, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha=ha, fontsize=12, color=color)
    if tech == "This Work (SNM)":
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

ax.set_xlabel("Memory Capacity (log scale, bits)")
ticks = [np.log10(2**10), np.log10(2**20), np.log10(2**30), np.log10(2**40)]
ticklabels = ["1 kbit", "1 Mbit", "1 Gbit", "1 Tbit"]
ax.set_xticks(ticks, ticklabels)
ax.grid(axis="x", linestyle="--", linewidth=0.5)

# ---------------------- Plot 2: Functional Density ----------------------
ax = axs[1]
bars = ax.barh(technologies_density, log_densities, color=colors_density, height=0.7)

ax.invert_yaxis()

for bar, label, tech in zip(bars, labels_density, technologies_density):
    width = bar.get_width()
    ha = "right"
    color = "white"
    ax.text(width - 0.2, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha=ha, fontsize=12, color=color)
    if tech == "This Work (SNM)":
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

ax.set_xlabel("Functional Density (log scale, bits/cm$^2$)")
ticks = [np.log10(1e6), np.log10(1e9), np.log10(1e12)]
ticklabels = ["1 Mbit/cm²", "1 Gbit/cm²", "1 Tbit/cm²"]
ax.set_xticks(ticks, ticklabels)
ax.grid(axis="x", linestyle="--", linewidth=0.5)

# ---------------------- Final Adjustments ----------------------
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
