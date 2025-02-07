import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
    plot_enable_write_sweep_single,
    plot_state_current_markers,
    get_state_current_markers_list,
    plot_waterfall,
    get_state_current_markers,
    get_write_current,
)


if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, ax = plt.subplots()
    ax, ax2 = plot_enable_write_sweep_multiple(ax, dict_list[0:7])
    plt.savefig("enable_write_sweep.pdf", bbox_inches="tight")
    plt.show()
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    
    colors = {
        0: "red",
        1: "red",
        2: "blue",
        3: "blue",
    }
    fig, ax = plt.subplots()
    for data_dict in dict_list:
       state_current_markers = get_state_current_markers(data_dict, "enable_write_current")
       write_current = get_write_current(data_dict)
       for i, state_current in enumerate(state_current_markers[0, :]):
           if state_current > 0:
               ax.plot(
                   write_current,
                   state_current,
                   "o",
                   label=f"{write_current} $\mu$A",
                   markerfacecolor=colors[i],
                   markeredgecolor="none",
               )

    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$I_{\mathrm{enable}}$ [$\mu$A]")