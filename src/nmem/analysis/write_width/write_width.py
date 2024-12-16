import matplotlib.pyplot as plt
import scipy.io as sio
from nmem.analysis.analysis import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [5, 3.5]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data5 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-04-50.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-12-42.mat"
    )

    data3 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-15-24.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-18-06.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-21-23.mat"
    )
    data0 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-57-17.mat"
    )

    data9 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 12-47-07.mat"
    )
    data8 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 12-53-16.mat"
    )
    data7 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 12-58-46.mat"
    )
    data6 = sio.loadmat(
        "SPG806_20240917_nMem_parameter_sweep_D6_A4_C1_2024-09-17 13-01-39.mat"
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
    }
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_width")
    plt.show()

    data_dict = {
        0: data0,
        1: data5,
        2: data6,
        3: data8,
    }

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_width")
