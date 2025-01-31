import matplotlib.pyplot as plt
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.markeredgewidth"] = 0.5

plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1


if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    # fig, ax = plt.subplots()
    # plot_read_sweep_array(
    #     ax, enable_read_290_list, "bit_error_rate", "enable_read_current"
    # )
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_read_sweep_array(
    #     ax, enable_read_300_list, "bit_error_rate", "enable_read_current"
    # )
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_read_sweep_array(
    #     ax, enable_read_310_list, "bit_error_rate", "enable_read_current"
    # )
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_read_sweep_array(
    #     ax, enable_read_310_C4_list, "bit_error_rate", "enable_read_current"
    # )
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_read_sweep_array(ax, data_inverse, "bit_error_rate", "enable_write_current")
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_read_sweep_array(ax, data, "bit_error_rate", "enable_read_current")
    # plt.show()


    fig, axs = plt.subplots(1, 3, figsize=(7, 4.3), sharey=True)
    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]
    for i in range(3):
        plot_read_sweep_array(axs[i], dict_list[i], "bit_error_rate", "enable_read_current")
        axs[i].set_xlim(400, 1000)
    axs[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1), title="Enable Read Current [$\mu$A]")

    axs[0].set_xlabel("Read Current [$\mu$A]")
    axs[0].set_ylabel("Normalized Bit Error Rate")
    plt.savefig("enable_read_current_sweep.pdf", bbox_inches="tight")
    # plt.savefig("enable_read_current_sweep.png", dpi=300, bbox_inches="tight")
    plt.show()