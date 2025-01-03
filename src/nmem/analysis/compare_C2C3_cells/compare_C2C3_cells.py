import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from nmem.analysis.analysis import get_fitting_points
from nmem.measurement.functions import build_array, plot_fitting

if __name__ == "__main__":
    data_dict = sio.loadmat(
        r"SPG806_20241220_nMem_measure_enable_response_D6_A4_C2_2024-12-20 13-28-02.mat"
    )
    data_dict2 = sio.loadmat(
        r"SPG806_20241220_nMem_measure_enable_response_D6_A4_C3_2024-12-20 17-28-57.mat"
    )

    split_idx = 10

    fig, axs = plt.subplots()
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)
    axs.plot(xfit, yfit, label="C2", linestyle="-")
    split_idx = 7
    plot_fitting(axs, xfit[split_idx + 1 :], yfit[split_idx + 1 :])

    split_idx = 10
    x2, y2, ztotal2 = build_array(data_dict2, "total_switches_norm")

    xfit, yfit = get_fitting_points(x2, y2, ztotal2)
    axs.plot(xfit, yfit, label="C3", linestyle="-")
    axs.legend()
    axs.set_ylim([0, 1000])
    axs.set_xlim([0, 500])

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    plot_fitting(
        axs[0], xfit[split_idx + 1 :], yfit[split_idx + 1 :], label="C3", linestyle="-"
    )
    axs[0].set_ylim([0, 1000])
    axs[0].set_xlim([0, 500])
    axs[0].plot(xfit, yfit, label="C2", linestyle="-")

    plot_fitting(axs[1], xfit[:split_idx], yfit[:split_idx], label="C3", linestyle="-")
    axs[1].plot(xfit, yfit, label="C2", linestyle="-")
    axs[1].set_ylim([0, 1000])
    axs[1].set_xlim([0, 500])
