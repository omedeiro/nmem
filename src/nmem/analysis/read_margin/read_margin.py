import matplotlib.pyplot as plt
import scipy.io as sio
from nmem.analysis.analysis import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240912_nMem_parameter_sweep_D6_A4_C1_2024-09-12 14-32-14.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240912_nMem_parameter_sweep_D6_A4_C1_2024-09-12 14-27-55.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240912_nMem_parameter_sweep_D6_A4_C1_2024-09-12 16-58-03.mat"
    )

    data3 = sio.loadmat(
        "SPG806_20240912_nMem_parameter_sweep_D6_A4_C1_2024-09-12 17-00-28.mat"
    )

    data4 = sio.loadmat(
        "SPG806_20240912_nMem_parameter_sweep_D6_A4_C1_2024-09-12 17-02-51.mat"
    )

    data_dict = {
        0: data0,
        1: data1,
        2: data2,
        3: data3,
        4: data4,
    }

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_current")
    ax.set_ylim(0, 0.5)
