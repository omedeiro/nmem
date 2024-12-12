import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


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
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)