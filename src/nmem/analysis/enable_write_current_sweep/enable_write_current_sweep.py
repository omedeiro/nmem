import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
    plot_enable_write_sweep_single,
    plot_state_current_markers,
    get_state_current_markers_list,
    plot_waterfall,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, ax = plt.subplots()
    ax, ax2 = plot_enable_write_sweep_multiple(ax, dict_list[3:-1:2])
    plt.savefig("enable_write_sweep.pdf", bbox_inches="tight")
    plt.show()
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    
    
    fig, ax = plt.subplots()
    for i in range(0, len(dict_list)):
        plot_enable_write_sweep_single(ax, dict_list[i])
        plot_state_current_markers(ax, dict_list[i], "enable_write_current")
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("Write Current ($\mu$A)")
    plt.show()
 
    fig, ax = plt.subplots()
    state_current_marker_list = get_state_current_markers_list(dict_list, "enable_write_current")
    for state_current_marker in state_current_marker_list:
        ax.plot(state_current_marker[0], state_current_marker[1], marker="o")
    
    # fig, ax = plt.subplots()
    # plot_operating_points(ax, dict_list, "write_current")
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_operating_margins(ax, dict_list, "write_current")
    # plt.show()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    # plot_waterfall(ax, dict_list[::2])
    # ax.view_init(45, 90)

    # plt.show()
