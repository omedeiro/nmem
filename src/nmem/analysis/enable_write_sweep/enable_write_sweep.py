import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep,
    plot_state_currents,
    plot_state_separation,
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


def plot_enable_write_sweep_grid(data_list: list[dict], save: bool = False) -> None:
    fig, ax = plt.subplot_mosaic(
        [["A", "B", "C", "D"], ["E", "E", "E", "E"]],
        figsize=(16, 9),
        tight_layout=True,
        sharex=False,
        sharey=False,
    )

    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_list)))
    for i, j in zip(["A", "B", "C", "D"], [2, 6, 7, 10]):
        ax[i] = plot_read_sweep(
            ax[i],
            data_list[j],
            "bit_error_rate",
            "enable_write_current",
            color=colors[j],
        )
        ax[i].set_xlabel("Read Current ($\mu$A)")
        if i == "A":
            ax[i].set_ylabel("Bit Error Rate")

    ax["E"] = plot_state_currents(ax["E"], data_list)
    fig.tight_layout()
    if save:
        fig.savefig("enable_write_sweep_grid.png", dpi=300)

    return


if __name__ == "__main__":

    data_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_state_currents(ax, data_list)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_state_separation(ax, data_list)
    plt.show()

    plot_enable_write_sweep_grid(data_list)