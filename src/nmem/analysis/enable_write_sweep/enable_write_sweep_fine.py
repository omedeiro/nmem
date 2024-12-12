import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MultipleLocator


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


def plot_write_sweep_fine(ax:Axes, data_dict: dict) -> Axes:
    for key in data_dict.keys():
        ax = plot_enable_write_sweep_single(ax, data_dict, key)
    return ax


if __name__ == "__main__":
    data_dict = {
        0: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 14-54-42.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-02-05.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-12-02.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-19-02.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-29-42.mat"
        ),
        5: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-37-52.mat"
        ),
        6: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-44-49.mat"
        ),
        7: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-51-58.mat"
        ),
        8: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-59-19.mat"
        ),
        9: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 16-10-26.mat"
        ),
        10: sio.loadmat(
            "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 16-33-07.mat"
        ),
    }

    fig, ax = plt.subplots()
    plot_write_sweep_fine(ax, data_dict)

    data_dict_top = {
        0: data_dict[0],
        1: data_dict[1],
        2: data_dict[2],
        3: data_dict[3],
    }

    fig, ax = plt.subplots()
    plot_write_sweep_fine(ax, data_dict_top)
