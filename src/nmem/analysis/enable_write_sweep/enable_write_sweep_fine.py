import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def plot_enable_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, 51))
    colors = np.flipud(colors)
    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        enable_write_currents = data["x"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        write_current = int(data["write_current"][0, 0, 0] * 1e6)

        plt.plot(
            enable_write_currents,
            ber,
            label=f"$I_{{W}}$ = {data['write_current'][0,0,0]*1e6:.1f} $\mu$A",
            color=colors[write_current],
            marker=".",
            markeredgecolor="k",
        )

    ax = plt.gca()

    # plt.ylim(0, 1)
    plt.yscale("log")
    plt.xlabel("Enable Write Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.legend(frameon=True, loc=3)
    plt.grid(True, which="both", axis="x", linestyle="--")
    plt.hlines(4e-2, ax.get_xlim()[0], ax.get_xlim()[1], linestyle="--", color="k")

    return ax


def plot_write_sweep_fine(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_enable_write_sweep_single(data_dict, key, ax)
    return ax


if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 14-54-42.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-02-05.mat"
    )

    data2 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-12-02.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-19-02.mat"
    )

    data4 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-29-42.mat"
    )

    data5 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-37-52.mat"
    )
    data6 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-44-49.mat"
    )
    data7 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-51-58.mat"
    )

    data8 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 15-59-19.mat"
    )
    data9 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 16-10-26.mat"
    )
    data10 = sio.loadmat(
        "SPG806_20240919_nMem_parameter_sweep_D6_A4_C1_2024-09-19 16-33-07.mat"
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
    }
    plot_write_sweep_fine(data_dict)

    data_dict = {
        0: data0,
        1: data1,
        2: data2,
        3: data3,
    }
    plot_write_sweep_fine(data_dict)
