import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14




def load_data(file_path: str):
    data = sio.loadmat(file_path)
    return data


def find_edge(bit_error_rate: np.ndarray):
    ber_diff = np.abs(np.diff(np.log10(bit_error_rate)))
    edge = np.where(ber_diff > 0.99)[0]
    return edge

def plot_read_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.3, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["write_1_read_0_norm"].flatten()
        write_current = data["enable_read_current"].flatten()[0] * 1e6
        plt.plot(
            read_currents,
            ber,
            label=f"$I_{{ER}}$ = {write_current:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

    ax = plt.gca()
    # plt.xlim(570, 650)
    # plt.hlines([0.5], ax.get_xlim()[0], ax.get_xlim()[1], linestyle=":", color="lightgray")
    # plt.ylim(1e-4, 1)
    # plt.xticks(np.linspace(570, 650, 5))
    # plt.yscale("log")
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.8, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        if any(ber > 0.9):
            ber = 1-ber
            line_style = "--"
        else:
            line_style = "-"
        enable_write_current = data["enable_write_current"].flatten()[0] * 1e6
        plt.plot(
            read_currents,
            ber,
            label=f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
            linestyle=line_style,
        )
        edge = find_edge(ber)
        if len(edge) > 0:
            plt.vlines(
                read_currents[edge[0]],
                1e-4,
                1,
                linestyle=":",
                color=colors[key],
                label=f"Edge {read_currents[edge[0]]:.1f}$\mu$A",
            )
            print(f"Edge {read_currents[edge[0]]:.1f}$\mu$A")
    ax = plt.gca()
    # plt.xlim(570, 650)
    # plt.hlines([0.5], ax.get_xlim()[0], ax.get_xlim()[1], linestyle=":", color="lightgray")
    # plt.ylim(1e-4, 1)
    # plt.xticks(np.linspace(570, 650, 5))
    # plt.yscale("log")
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax

def plot_enable_read_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_read_sweep_single(data_dict, key, ax)
    return ax

def plot_enable_write_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_write_sweep_single(data_dict, key, ax)
    return ax

if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 15-10-41.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 15-17-47.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-11-46.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-18-32.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-25-36.mat"
    )


    read_dict = {
        0: load_data("SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"),
        1: load_data("SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"),
    }

    data_dict = {0: data0, 1: data1, 2: data2, 3: data3, 4: data4}

    plot_enable_read_sweep_multiple(data_dict)
    plot_enable_write_sweep_multiple(read_dict)
