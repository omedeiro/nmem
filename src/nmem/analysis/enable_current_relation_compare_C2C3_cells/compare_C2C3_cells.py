import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    build_array,
    get_fitting_points,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_fitting,
    plot_channel_temperature,
)

if __name__ == "__main__":
    data_list = import_directory("data")
    data_dict = data_list[0]
    data_dict2 = data_list[1]

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
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    plot_fitting(axs[1], xfit[:split_idx], yfit[:split_idx], label="C3", linestyle="-")
    axs[1].plot(xfit, yfit, label="C2", linestyle="-")
    axs[1].set_ylim([0, 1000])
    axs[1].set_xlim([0, 500])
    axs[1].set_xlabel("Enable Current ($\mu$A)")

    # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)
    # axs[0].plot(xfit, yfit, label="C2", linestyle="-")


    save = False
    if save:
        plt.savefig("enable_current_relation_compare_C2C3.png", dpi=300, bbox_inches="tight")
    plt.show()

