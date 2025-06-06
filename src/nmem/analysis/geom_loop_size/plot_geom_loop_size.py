
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import set_plot_style

set_plot_style()

loop_sizes = np.arange(1.7, 5.2, 0.5)


# Load data
data = import_directory("data")
N = len(data)
NMEAS = 1000

# First plot: Vch vs. ber_est
fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
ax = axs[0]
for i in range(N):
    Vch = np.ravel(data[i]["Vch"]) * 1e3
    ber_est = np.ravel(data[i]["ber_est"])
    err = np.sqrt(ber_est * (1 - ber_est) / NMEAS)
    ax.plot(Vch, ber_est, label=f"$w_{{5}}$ = {loop_sizes[i]:.1f} µm")
ax.set_yscale("log")
ax.set_xlabel("channel voltage [mV]")
ax.set_ylabel("estimated BER")
ax.legend(loc="lower left", labelspacing=0.1, handlelength=1.5, fontsize=7)

# Second plot: best BER vs loop size
best_ber = [np.min(np.ravel(d["ber_est"])) for d in data]

ax = axs[1]
ax.plot(loop_sizes, best_ber, "-o")
ax.set_yscale("log")
ax.set_xlabel("loop size [µm]")
ax.set_ylabel("minimum BER")
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.2)

axs[2].axis("off")
save_fig = False
if save_fig:
    plt.savefig(
        "geom_loop_size.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
plt.show()
