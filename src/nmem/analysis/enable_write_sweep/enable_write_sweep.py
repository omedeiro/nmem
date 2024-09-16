import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


def plot_enable_write_sweep(data_dict: dict):
    cmap = plt.get_cmap("Blues")
    colors = cmap(np.linspace(0.2, 1, len(data_dict)))
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
        plt.xlim(read_currents[0], read_currents[-1])
        plt.xlabel("Read Current ($\mu$A)")
        plt.ylabel("Bit Error Rate")
        plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1))
    plt.title("Enable Write Sweep")


if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 15-44-44.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 15-50-02.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 16-00-08.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240913_nMem_parameter_sweep_D6_A4_C1_2024-09-13 16-04-27.mat"
    )

    data_dict = {
        0: data0,
        1: data1,
        2: data2,
        3: data3,
    }
    fig, ax = plt.subplots()
    plot_enable_write_sweep(data_dict)
