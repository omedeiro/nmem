import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes

plt.rcParams["figure.figsize"] = [5, 3.5]
plt.rcParams["font.size"] = 14


def plot_enable_write_sweep_single(ax: Axes, data_dict: dict, **kwargs) -> Axes:
    read_currents = data_dict.get("y")[:, :, 0].flatten() * 1e6
    ber = data_dict.get("bit_error_rate").flatten()
    enable_write_width = data_dict.get("enable_write_width").flatten()[0]
    ax.plot(
        read_currents,
        ber,
        label=f"{enable_write_width:.1f}",
        marker=".",
        markeredgecolor="k",
        **kwargs,
    )

    ax.set_ylim(1e-4, 1)
    ax.set_xticks(np.linspace(570, 680, 5))
    ax.set_yscale("log")
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Write Enable Width Sweep")
    ax.grid(True)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

    return ax


def plot_enable_write_sweep_multiple(ax: Axes, data_dict: dict) -> Axes:
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    for key in data_dict.keys():
        color = colors[key]
        ax = plot_enable_write_sweep_single(ax, data_dict[key], color=color)

    return ax


if __name__ == "__main__":
    data5 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-11-16.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-13-59.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-54-27.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-16-57.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-28-34.mat"
    )
    data0 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-25-56.mat"
    )

    data6 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-37-37.mat"
    )
    data7 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-35-00.mat"
    )
    data8 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 14-32-15.mat"
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
    }
    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_dict)

    data_dict = {
        0: data0,
        1: data2,
        2: data3,
        3: data4,
        4: data5,
        5: data8,
    }

    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_dict)
