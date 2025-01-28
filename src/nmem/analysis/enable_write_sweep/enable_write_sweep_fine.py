import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MultipleLocator

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
)


def plot_enable_write_sweep_single(ax: Axes, data_dict: dict, index: int) -> Axes:
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, 51))
    colors = np.flipud(colors)
    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        enable_write_currents = data["x"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        write_current = int(data["write_current"][0, 0, 0] * 1e6)

        ax.plot(
            enable_write_currents,
            ber,
            label=f"$I_{{W}}$ = {data['write_current'][0,0,0]*1e6:.1f} $\mu$A",
            color=colors[write_current],
            marker=".",
            markeredgecolor="k",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Enable Write Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.legend(frameon=True, loc=3)
    ax.grid(True, which="both", axis="x", linestyle="--")
    ax.hlines(4e-2, ax.get_xlim()[0], ax.get_xlim()[1], linestyle="--", color="k")

    return ax


def plot_write_sweep_fine(ax: Axes, data_dict: dict) -> Axes:
    for key in data_dict.keys():
        ax = plot_enable_write_sweep_single(ax, data_dict, key)
    return ax


if __name__ == "__main__":
    data_list2 = import_directory("data2")
    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_list2)
    plt.show()