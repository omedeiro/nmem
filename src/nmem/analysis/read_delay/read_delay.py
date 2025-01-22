import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import plot_read_delay

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


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
    plot_read_delay(ax, data_dict)
