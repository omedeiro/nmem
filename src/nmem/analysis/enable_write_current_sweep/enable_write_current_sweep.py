
import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import (
    plot_enable_write_sweep_multiple,
    # plot_peak_distance,
    plot_operating_points,
    plot_operating_margins,
    plot_waterfall,
    import_directory,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 12



if __name__ == "__main__":
    # data_dict = {
    #     0: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-52-33.mat"
    #     ),
    #     1: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-45-20.mat"
    #     ),
    #     2: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-33-31.mat"
    #     ),
    #     3: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-26-47.mat"
    #     ),
    #     4: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-20-06.mat"
    #     ),
    #     5: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-22-38.mat"
    #     ),
    #     6: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-41-06.mat"
    #     ),
    #     7: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-09-12.mat"
    #     ),
    #     8: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-14-52.mat"
    #     ),
    #     9: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-49-37.mat"
    #     ),
    #     10: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-41-43.mat"
    #     ),
    #     11: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-49-23.mat"
    #     ),
    #     12: sio.loadmat(
    #         "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-19-05.mat"
    #     ),
    # }

    # data_dict1 = {
    #     0: data_dict[5],
    #     1: data_dict[12],
    # }
    # data_dict2 = {
    #     0: data_dict[1],
    #     1: data_dict[3],
    #     2: data_dict[5],
    #     3: data_dict[6],
    #     4: data_dict[7],
    #     5: data_dict[8],
    #     6: data_dict[9],
    #     7: data_dict[10],
    #     8: data_dict[11],
    #     9: data_dict[12],
    # }

    data_list = import_directory("data")
    ALPHA = 0.612
    WIDTH_RATIO = 1.8
    IRETRAP = 0.82
    IREAD = 630
    CURRENT_CELL = "C1"
    ICHL = 150
    ICHR = ICHL * WIDTH_RATIO

    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_list)
    plt.show()

    # fig, ax = plt.subplots()
    # plot_operating_points(ax, data_list)
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_operating_margins(ax, data_list)
    # plt.show()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    # plot_waterfall(ax, data_list)
    # plt.show()