import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


def plot_read_delay(data_dict: dict):
    cmap = plt.get_cmap("Reds")
    colors = cmap(np.linspace(0.2, 1, len(data_dict)))
    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        ax = plt.plot(
            read_currents,
            ber,
            label=f"+{key}$\mu$s",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )
        plt.xlim(read_currents[0], read_currents[-1])
        plt.xlabel("Read Current ($\mu$A)")
        plt.ylabel("Bit Error Rate")
        plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1))


if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 13-30-06.mat"
    )

    data1 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 13-33-45.mat"
    )

    data2 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 13-26-26.mat"
    )

    data3 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 13-37-24.mat"
    )

    data_dict = {
        0: data0,
        1: data1,
        2: data2,
        3: data3,
    }

    fig, ax = plt.subplots()
    plot_read_delay(data_dict)
