import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


def plot_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("Greens")
    colors = cmap(np.linspace(0.3, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        write_current = data["write_current"].flatten()[0] * 1e6
        plt.plot(
            read_currents,
            ber,
            label=f"$I_W$ = {write_current:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

    ax = plt.gca()
    plt.xlim(570, 650)
    plt.hlines([0.5], ax.get_xlim()[0], ax.get_xlim()[1], linestyle=":", color="lightgray")
    plt.ylim(1e-4, 1)
    plt.xticks(np.linspace(570, 650, 5))
    plt.yscale("log")
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    plt.title("Write Current Sweep")
    return ax


def plot_enable_write_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_write_sweep_single(data_dict, key, ax)
    return ax


if __name__ == "__main__":
    
    data0 = sio.loadmat(
        "SPG806_20240920_nMem_parameter_sweep_D6_A4_C1_2024-09-20 16-43-58.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240920_nMem_parameter_sweep_D6_A4_C1_2024-09-20 16-58-16.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240920_nMem_parameter_sweep_D6_A4_C1_2024-09-20 17-11-39.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240920_nMem_parameter_sweep_D6_A4_C1_2024-09-20 17-25-49.mat"
    )
    data5 = sio.loadmat(
        "SPG806_20240920_nMem_parameter_sweep_D6_A4_C1_2024-09-20 16-28-10.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240920_nMem_parameter_sweep_D6_A4_C1_2024-09-20 17-45-14.mat"
    )
    data_dict = {0: data0, 1: data1, 2: data2, 3: data3, 4: data4, 5: data5}
    plot_enable_write_sweep_multiple(data_dict)