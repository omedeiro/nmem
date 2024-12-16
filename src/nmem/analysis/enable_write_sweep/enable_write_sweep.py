import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from nmem.analysis.analysis import find_state_currents, plot_read_sweep

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


def plot_enable_write_sweep_grid(data_dict: dict, save: bool = False) -> None:
    fig, ax = plt.subplot_mosaic(
        [["A", "B", "C", "D"], ["E", "E", "E", "E"]],
        figsize=(16, 9),
        tight_layout=True,
        sharex=False,
        sharey=False,
    )

    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))
    for i, j in zip(["A", "B", "C", "D"], [2, 6, 7, 10]):
        ax[i] = plot_read_sweep(
            ax[i],
            data_dict[j],
            "bit_error_rate",
            "enable_write_current",
            color=colors[j],
        )
        ax[i].set_xlabel("Read Current ($\mu$A)")
        if i == "A":
            ax[i].set_ylabel("Bit Error Rate")

    ax["E"] = plot_state_currents(ax["E"], data_dict)
    fig.tight_layout()
    if save:
        fig.savefig("enable_write_sweep_grid.png", dpi=300)

    return


def plot_state_separation(ax: Axes, data_dict: dict) -> Axes:
    state_separation = []
    for key, data in data_dict.items():
        state0_current, state1_current = find_state_currents(data)
        state_separation.append(state1_current - state0_current)

    ax.bar(
        [
            data.get("enable_write_current")[0, 0, 0] * 1e6
            for data in data_dict.values()
        ],
        state_separation,
        width=2.5,
    )
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.grid(True, axis="both", which="both")
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("Diff. Between State Currents ($\mu$A)")
    return ax


def plot_state_currents(ax: Axes, data_dict: dict):
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))
    enable_write_currents = []
    state0_currents = []
    state1_currents = []
    for key, data in data_dict.items():
        state0_current, state1_current = find_state_currents(data)

        enable_write_currents.append(data["enable_write_current"][0, 0, 0] * 1e6)
        state0_currents.append(state0_current)
        state1_currents.append(state1_current)

        ax.plot(
            data["enable_write_current"][0, 0, 0] * 1e6,
            state0_current,
            "D",
            color=colors[key],
            markeredgecolor="k",
        )
        ax.plot(
            data["enable_write_current"][0, 0, 0] * 1e6,
            state1_current,
            "P",
            color=colors[key],
            markeredgecolor="k",
            markersize=10,
            markeredgewidth=1,
        )

    ax.plot(
        enable_write_currents,
        state0_currents,
        label="State 0",
        marker="D",
        color="grey",
        markeredgecolor="k",
        zorder=0,
    )
    ax.plot(
        enable_write_currents,
        state1_currents,
        label="State 1",
        marker="P",
        color="grey",
        markeredgecolor="k",
        markersize=10,
        zorder=0,
    )
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("State Current ($\mu$A)")
    ax.grid(True, which="both", axis="both")
    ax.legend(frameon=False)

    return ax


if __name__ == "__main__":
    data_dict = {
        0: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-51-36.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-45-47.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-43-16.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-40-40.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-37-15.mat"
        ),
        5: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-54-34.mat"
        ),
        6: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-25-14.mat"
        ),
        7: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-28-20.mat"
        ),
        8: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-31-44.mat"
        ),
        9: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-34-33.mat"
        ),
        10: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-04-25.mat"
        ),
        11: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-19-21.mat"
        ),
        12: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-22-26.mat"
        ),
        13: sio.loadmat(
            "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-26-08.mat"
        ),
    }
    plot_enable_write_sweep_grid(data_dict)

    fig, ax = plt.subplots()
    plot_state_currents(ax, data_dict)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_state_separation(ax, data_dict)
