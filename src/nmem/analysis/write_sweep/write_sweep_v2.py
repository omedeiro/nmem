import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240918_nMem_parameter_sweep_D6_A4_C1_2024-09-18 13-15-55.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240918_nMem_parameter_sweep_D6_A4_C1_2024-09-18 13-22-08.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240918_nMem_parameter_sweep_D6_A4_C1_2024-09-18 13-28-30.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240918_nMem_parameter_sweep_D6_A4_C1_2024-09-18 13-42-42.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240918_nMem_parameter_sweep_D6_A4_C1_2024-09-18 13-55-18.mat"
    )
    data5 = sio.loadmat(
        "SPG806_20240918_nMem_parameter_sweep_D6_A4_C1_2024-09-18 14-13-05.mat"
    )

    data_dict = {0: data0, 1: data1, 2: data2, 3: data3}
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)

    data_dict_fine = {0: data5, 1: data4}
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict_fine, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
