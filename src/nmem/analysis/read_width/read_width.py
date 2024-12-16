import matplotlib.pyplot as plt
import scipy.io as sio
from nmem.analysis.analysis import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [5, 3.5]
plt.rcParams["font.size"] = 14


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

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "read_width")
    ax.set_yscale("log")
    ax.set_ylim([1e-4, 1])
