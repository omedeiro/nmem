from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


def plot_enable_write_sweep_single(ax: Axes, data_dict: dict, index: int) -> Axes:
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        ax.plot(
            read_currents,
            ber,
            label=f"$I_{{EW}}$ = {data['enable_write_current'][0,0,0]*1e6:.1f} $\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

        state0_current, state1_current = find_state_currents(data)
        ax.plot(
            read_currents[read_currents == state0_current],
            ber[read_currents == state0_current],
            color=colors[key],
            marker="D",
            markerfacecolor=colors[key],
            markeredgecolor="k",
            linewidth=1.5,
            label="_state0",
        )
        ax.plot(
            read_currents[read_currents == state1_current],
            ber[read_currents == state1_current],
            color=colors[key],
            marker="P",
            markerfacecolor=colors[key],
            markeredgecolor="k",
            markersize=10,
            linewidth=1.5,
            label="_state1",
        )

    ax.set_ylim(0, 1)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.legend(frameon=True, loc=2)
    return ax


def plot_enable_write_sweep_grid(data_dict: dict, save: bool = False) -> None:
    fig, ax = plt.subplot_mosaic(
        [["A", "B", "C", "D"], ["E", "E", "E", "E"]],
        figsize=(16, 9),
        tight_layout=True,
        sharex=False,
        sharey=False,
    )
    for i, j in zip(["A", "B", "C", "D"], [2, 6, 7, 10]):
        ax[i] = plot_enable_write_sweep_single(ax[i], data_dict, j)
        if i == "A":
            ax[i].set_ylabel("Bit Error Rate")
            ax[i].legend(loc=2)
        if i == "C":
            ax[i].legend(loc=3)
        if i == "D":
            ax[i].legend(loc=3)

        ax[i].set_xlabel("Read Current ($\mu$A)")

    ax["E"] = plot_state_currents(ax["E"], data_dict)
    plt.tight_layout()
    if save:
        plt.savefig("enable_write_sweep_grid.png", dpi=300)
    plt.show()
    return


def plot_persistent_currents(ax: Axes, data_dict: dict) -> None:
    persistent_currents = []
    for key, data in data_dict.items():
        state0_current, state1_current = find_state_currents(data)
        persistent_currents.append(state1_current - state0_current)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    plt.sca(ax)

    plt.bar(
        [data["enable_write_current"][0, 0, 0] * 1e6 for data in data_dict.values()],
        persistent_currents,
        width=2.5,
    )
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    plt.grid(True, axis="both", which="both")
    plt.xlabel("Enable Write Current ($\mu$A)")
    plt.ylabel("Diff. Between State Currents ($\mu$A)")
    return


def find_state_currents(data_dict: dict) -> Tuple[float, float]:
    read_currents = data_dict["y"][:, :, 0].flatten() * 1e6
    ber = data_dict["bit_error_rate"].flatten()

    if np.max(ber) > 0.6:
        max_ber = np.max(ber)
        ber_threshold = 0.5 + (max_ber - 0.5) / 2
        state1_current = read_currents[ber > ber_threshold][0]
        state0_current = read_currents[ber > ber_threshold][-1]

    else:
        min_ber = np.min(ber)
        ber_threshold = 0.5 - (0.5 - min_ber) / 2
        state0_current = read_currents[ber < ber_threshold][0]
        state1_current = read_currents[ber < ber_threshold][-1]

    return state0_current, state1_current


def plot_state_currents(ax: Axes, data_dict: dict):
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))
    enable_write_currents = []
    state0_currents = []
    state1_currents = []
    for key, data in data_dict.items():
        state0_current, state1_current = find_state_currents(data)
        # print(f"State 0 Current: {state0_current:.2f} µA")
        # print(f"State 1 Current: {state1_current:.2f} µA")

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

