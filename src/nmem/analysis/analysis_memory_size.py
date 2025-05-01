import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import to_rgb
from nmem.analysis.analysis import set_inter_font, set_pres_style

# Apply custom styles
set_inter_font()
set_pres_style()

# ---------------------- Data (sorted largest to smallest) ----------------------
tech_cap = ["CMOS NAND Flash", "CMOS DRAM", "CMOS SRAM", "Josephson RAM", "This Work (SNM)"]
cap_vals = np.log10([1e12, 24e9, 64e6, 4e3, 64])
cap_labels = ["~1 Tb", "~24 Gb", "~64 Mb", "4 kb", "64 b"]
cap_colors = ["gray", "gray", "gray", "darkred", "royalblue"]

tech_den = ["CMOS NAND Flash", "CMOS DRAM", "CMOS SRAM", "Josephson RAM", "This Work (SNM)"]
den_vals = np.log10([96e9, 50e9, 8e9, 1e6, 2.6e6])
den_labels = ["~96 Gbit/cm²", "~50 Gbit/cm²", "~8 Gbit/cm²", "1 Mbit/cm²", "2.6 Mbit/cm²"]
den_colors = ["gray", "gray", "gray", "darkred", "royalblue"]

# ---------------------- Helper Functions ----------------------
def darken(color, factor=0.6):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))

def lighten(color, factor=1.1):  # less aggressive than before
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))

def draw_extruded_barh(ax, y_labels, values, colors, labels, xlabel, xticks, xticklabels):
    bar_height = 0.6
    depth = 0.15

    for i, (val, label, base_color) in enumerate(zip(values, labels, colors)):
        y = i

        # Color shading
        front_color = base_color
        top_color = lighten(base_color, 1.1)
        side_color = darken(base_color, 0.6)

        # Front face
        rect = Rectangle((0, y - bar_height/2), val, bar_height,
                         facecolor=front_color, edgecolor='none')
        ax.add_patch(rect)

        # Top face
        top = Polygon([
            (0, y + bar_height/2),
            (depth, y + bar_height/2 + depth),
            (val + depth, y + bar_height/2 + depth),
            (val, y + bar_height/2)
        ], closed=True, facecolor=top_color, edgecolor='none')
        ax.add_patch(top)

        # Side face
        side = Polygon([
            (val, y - bar_height/2),
            (val, y + bar_height/2),
            (val + depth, y + bar_height/2 + depth),
            (val + depth, y - bar_height/2 + depth)
        ], closed=True, facecolor=side_color, edgecolor='none')
        ax.add_patch(side)

        # Label inside bar
        ax.text(val - 0.2, y, label, va="center", ha="right", fontsize=13,
                color="white" if base_color != "royalblue" else "black")

    # Axes settings
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()  # larger bars on top
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.set_xlim(0, max(values) + 1.5)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)

# ---------------------- Create Figure ----------------------
fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=False)
fig.suptitle("Cryogenic-Compatible Memory Scaling Comparison", weight="bold", y=0.96)

# Plot 1: Capacity
draw_extruded_barh(
    axs[0], tech_cap, cap_vals, cap_colors, cap_labels,
    xlabel="Memory Capacity (log scale, bits/chip)",
    xticks=[np.log10(2**10), np.log10(2**20), np.log10(2**30), np.log10(2**40)],
    xticklabels=["1 kb", "1 Mb", "1 Gb", "1 Tb"]
)

# Plot 2: Density
draw_extruded_barh(
    axs[1], tech_den, den_vals, den_colors, den_labels,
    xlabel="Functional Density (log scale, bits/cm²)",
    xticks=[np.log10(1e6), np.log10(1e9), np.log10(1e12)],
    xticklabels=["1 Mb/cm²", "1 Gb/cm²", "1 Tb/cm²"]
)

# Final layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("memory_scaling_comparison_3dextruded_lit_sorted.png", dpi=600)
plt.show()
