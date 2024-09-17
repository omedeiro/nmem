import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.ticker import MultipleLocator

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


def plot_enable_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        plt.plot(
            read_currents,
            ber,
            label=f"$I_{{EW}}$ = {data['enable_write_current'][0,0,0]*1e6:.1f} $\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

        state0_current, state1_current = find_state_currents(data)
        plt.plot(
            read_currents[read_currents == state0_current],
            ber[read_currents == state0_current],
            color=colors[key],
            marker="D",
            markerfacecolor=colors[key],
            markeredgecolor="k",
            linewidth=1.5,
        )
        plt.plot(
            read_currents[read_currents == state1_current],
            ber[read_currents == state1_current],
            color=colors[key],
            marker="P",
            markerfacecolor=colors[key],
            markeredgecolor="k",
            markersize=10,
            linewidth=1.5,
        )

    ax = plt.gca()

    plt.ylim(0, 1)
    # plt.yscale("log")
    # plt.xlabel("Read Current ($\mu$A)")
    # plt.ylabel("Bit Error Rate")
    plt.grid(True)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.legend(frameon=True, loc=2)
    # plt.title("Enable Write Sweep")
    return ax


def plot_enable_write_sweep(data_dict: dict, index: int = None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    if index is not None:
        data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        ax = plt.plot(
            read_currents,
            ber,
            label=f"$I_{{EW}}$ = {data['enable_write_current'][0,0,0]*1e6:.1f} $\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

        state0_current, state1_current = find_state_currents(data)
        plt.plot(
            read_currents[read_currents == state0_current],
            ber[read_currents == state0_current],
            color=colors[key],
            marker="D",
            markerfacecolor=colors[key],
            markeredgecolor=colors[key],
        )
        plt.plot(
            read_currents[read_currents == state1_current],
            ber[read_currents == state1_current],
            color=colors[key],
            marker="P",
            markerfacecolor=colors[key],
            markeredgecolor=colors[key],
            markersize=10,
        )
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1.25))
    plt.title("Enable Write Sweep")
    return ax


def plot_enable_write_sweep_grid(data_dict: dict):
    fig, ax = plt.subplot_mosaic(
        [["A", "B", "C", "D"], ["E", "E", "E", "E"]],
        figsize=(16, 9),
        tight_layout=True,
        sharex=False,
        sharey=False,
    )
    for i, j in zip(["A", "B", "C", "D"], [2, 6, 7, 10]):
        ax[i] = plot_enable_write_sweep_single(data_dict, j, ax=ax[i])
        if i == "A":
            print(ax[i])
            ax[i].set_ylabel("Bit Error Rate")
            plt.legend(loc=2)
        if i == "C":
            plt.legend(loc=3)
        if i == "D":
            plt.legend(loc=3)

        ax[i].set_xlabel("Read Current ($\mu$A)")

    ax["E"] = plot_state_currents(data_dict, ax=ax["E"])
    plt.tight_layout()
    plt.show()


def plot_persistent_currents(data_dict: dict, ax=None):
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


def find_state_currents(data_dict: dict):
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


def plot_state_currents(data_dict: dict, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
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

        plt.plot(
            data["enable_write_current"][0, 0, 0] * 1e6,
            state0_current,
            "D",
            color=colors[key],
            markeredgecolor="k",
        )
        plt.plot(
            data["enable_write_current"][0, 0, 0] * 1e6,
            state1_current,
            "P",
            color=colors[key],
            markeredgecolor="k",
            markersize=10,
            markeredgewidth=1,
        )

    plt.plot(
        enable_write_currents,
        state0_currents,
        label="State 0",
        marker="D",
        color="grey",
        markeredgecolor="k",
        zorder=0,
    )
    plt.plot(
        enable_write_currents,
        state1_currents,
        label="State 1",
        marker="P",
        color="grey",
        markeredgecolor="k",
        markersize=10,
        zorder=0,
    )
    plt.xlabel("Enable Write Current ($\mu$A)")
    plt.ylabel("State Current ($\mu$A)")
    plt.grid(True)
    plt.legend(frameon=False)
    # plt.title("Enable Write Sweep")

    return ax


if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-51-36.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-48-50.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-45-47.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-43-16.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-40-40.mat"
    )
    data5 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-37-15.mat"
    )
    data6 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-54-34.mat"
    )
    data7 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-25-14.mat"
    )
    data8 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-28-20.mat"
    )
    data9 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-31-44.mat"
    )
    data10 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 14-34-33.mat"
    )
    data11 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-04-25.mat"
    )
    data12 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-19-21.mat"
    )
    data13 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-22-26.mat"
    )
    data14 = sio.loadmat(
        "SPG806_20240916_nMem_parameter_sweep_D6_A4_C1_2024-09-16 15-26-08.mat"
    )

    data_dict = {
        0: data0,
        1: data1,
        2: data2,
        3: data3,
        4: data4,
        5: data5,
        6: data6,
        7: data7,
        8: data8,
        9: data9,
        10: data10,
        11: data11,
        12: data12,
        13: data13,
        14: data14,
    }
    plot_enable_write_sweep_grid(data_dict)
    plot_persistent_currents(data_dict)
