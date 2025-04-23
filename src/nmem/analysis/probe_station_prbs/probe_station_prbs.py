import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.cm import get_cmap
from nmem.analysis.analysis import import_directory, set_plot_style

set_plot_style()

# Load data
data_list = import_directory(".")
N = len(data_list)
trim = 4500
W1R0_error = 0
W0R1_error = 0

# Colormap and colors
cmap = get_cmap("Reds")
colors = cmap(np.linspace(0.5, 1.0, N))

fig, ax = plt.subplots(figsize=(6, 2.5))

for i, data in enumerate(data_list):
    x = data["trace_chan"][0] * 1e6  # µs
    y = data["trace_chan"][1] * 1e3  # mV
    x_trimmed = x[trim:-trim]
    y_trimmed = y[trim:-trim] + (i * 20)

    ax.plot(x_trimmed, y_trimmed, color="black", linewidth=0.75)

    bit_write = "".join(data["bit_string"].flatten())
    bit_read = "".join(data["byte_meas"].flatten())
    errors = [bw != br for bw, br in zip(bit_write, bit_read)]

    # Add bit sequence text near the right edge of each trace
    text_x = x_trimmed[-1] + 1  # Offset slightly to the right
    text_y = y_trimmed[len(y_trimmed) // 2]  # Middle of the trace
    ax.text(text_x, text_y, f"Write: {bit_write}", fontsize=6, va="center", ha="left")

    for j, error in enumerate(errors):
        if error:
            ex = 0.4 + j * 1
            ey = -5 + i * 20
            exw = 0.5
            eyw = 15
            px = [ex, ex + exw, ex + exw, ex]
            py = [ey, ey, ey + eyw, ey + eyw]
            polygon = Polygon(
                xy=list(zip(px, py)), color=colors[-1], alpha=0.5, linewidth=0
            )
            ax.add_patch(polygon)

            if bit_write[j] == "1":
                W1R0_error += 1
            elif bit_write[j] == "0":
                W0R1_error += 1

total_error = W1R0_error + W0R1_error
ax.set_title(
    f"Total: {total_error}, W1R0: {W1R0_error}, W0R1: {W0R1_error}", fontsize=7
)
ax.set_xlabel("Time (µs)", fontsize=7)
ax.set_ylabel("Voltage (mV)", fontsize=7)

ax.tick_params(direction="in", length=3, width=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("probe_station_prbs.pdf", bbox_inches="tight")
plt.show()
