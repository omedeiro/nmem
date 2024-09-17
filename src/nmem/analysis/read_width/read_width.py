import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [5, 3.5]
plt.rcParams["font.size"] = 14


def plot_enable_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        write_width = data["read_width"].flatten()[0]
        plt.plot(
            read_currents,
            ber,
            label=f"{write_width:.1f}",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

    ax = plt.gca()

    plt.ylim(1e-4, 1)
    plt.xticks(np.linspace(600, 650, 5))
    plt.yscale("log")
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    plt.title("Read Width Sweep")
    return ax


def plot_enable_write_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_enable_write_sweep_single(data_dict, key, ax)
    return ax


if __name__ == "__main__":
    data6 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-53-11.mat"
    )
    data7 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-57-39.mat"
    )
    data5 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-38-59.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-33-56.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-31-07.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-28-29.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 15-25-09.mat"
    )
    data0 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 16-49-00.mat"
    )
    data8 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 17-00-35.mat"
    )
    data9 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 17-04-51.mat"
    )

    data10 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 17-08-52.mat"
    )
    data11 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 17-11-51.mat"
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
    }

    data_dict = {
        0: data1,
        1: data3,
        2: data5,
        3: data7,
        4: data11,
    }
    plot_enable_write_sweep_multiple(data_dict)
    plt.show()
