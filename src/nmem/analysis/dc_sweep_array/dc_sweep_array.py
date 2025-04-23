import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from nmem.analysis.analysis import set_plot_style, import_directory
import os
set_plot_style()

# Set global plot style for publication
# mpl.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.size": 12,
#         "axes.labelsize": 12,
#         "axes.titlesize": 12,
#         "legend.fontsize": 10,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "axes.linewidth": 1,
#     }
# )

# Load the .mat file
data = import_directory(os.getcwd())[0]
ic_vs_ih = data["ic_vs_ih_data"]

# Extract the fields
heater_currents = ic_vs_ih["heater_currents"][0, 0]
avg_current = ic_vs_ih["avg_current"][0, 0]
ystd = ic_vs_ih["ystd"][0, 0]
cell_names = ic_vs_ih["cell_names"][0, 0]

# Define color and line style maps
row_colors = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
    "G": "#e377c2",
}
col_linestyles = {
    "1": "-",
    "2": "--",
    "3": "-.",
    "4": ":",
    "5": (0, (3, 1, 1, 1)),
    "6": (0, (5, 2)),
    "7": (0, (1, 1)),
}

# Initialize plot
fig, ax = plt.subplots(figsize=(3, 3))

x_intercepts = []
y_intercepts = []

# Initialize a set to track rows already added to the legend
rows_in_legend = set()

# Iterate over each cell
for j in range(heater_currents.shape[1]):
    ih = np.squeeze(heater_currents[0, j]) * 1e6
    ic = np.squeeze(avg_current[0, j])
    err = np.squeeze(ystd[0, j])

    cell_name = str(cell_names[0, j][0])
    row, col = cell_name[0], cell_name[1]

    color = row_colors.get(row, "black")
    linestyle = col_linestyles.get(col, "-")

    # Add to legend only if the row is not already included
    label = f"Row {row}" if row not in rows_in_legend else None
    if label:
        rows_in_legend.add(row)

    # Plot with specific style
    ax.errorbar(
        ih,
        ic,
        yerr=err,
        label=label,
        color=color,
        linestyle=linestyle,
        marker="o",  # Slightly larger marker
        markersize=5,
        linewidth=1.2,  # Main line width
        elinewidth=1.75,  # Thinner error bar lines
        capsize=2,  # No caps for error bars
        alpha=0.9,  # Slight transparency for overlap
    )


    # Linear fit for intercepts (200-600 µA)
    valid_indices = (ih >= 200) & (ih <= 550)
    ih_filtered = ih[valid_indices]
    ic_filtered = ic[valid_indices]

    if len(ih_filtered) > 1:
        z = np.polyfit(ih_filtered, ic_filtered, 1)
        x_intercept = -z[1] / z[0]
        y_intercept = z[1]

        x_intercepts.append(x_intercept)
        y_intercepts.append(y_intercept)

# Average fit line
filtered_x = np.array(x_intercepts)
filtered_y = np.array(y_intercepts)
valid_avg = (filtered_x > 0) & (filtered_x < 1e3)
avg_x_intercept = np.mean(filtered_x[valid_avg])
avg_y_intercept = np.mean(filtered_y[valid_avg])


def avg_line(x):
    slope = avg_y_intercept / avg_x_intercept
    return -slope * x + avg_y_intercept


fit_range = np.linspace(0, 800, 100)
ax.plot(
    fit_range,
    avg_line(fit_range),
    color="black",
    linestyle="-",
    linewidth=2,
    label="Average Fit",
)

# Final touches
ax.set_xlabel(r"$I_h$ (µA)")
ax.set_ylabel(r"$I_c$ (µA)")
# ax.set_title(r"$I_c$ vs. $I_h$ Across Array Cells")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(ncol=2, frameon=False, loc="upper right")
ax.set_ybound(lower=0)
ax.set_xlim(0, 800)
plt.tight_layout()
plt.savefig("ic_vs_ih_array.pdf", dpi=300, bbox_inches="tight")
plt.show()
