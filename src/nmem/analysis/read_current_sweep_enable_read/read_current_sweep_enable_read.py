import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
    get_write_temperature,
    get_write_temperatures,
    get_read_temperatures_array,
    get_enable_read_currents_array,
    get_enable_write_current,
    get_enable_write_current_array,
    get_write_current,
    get_write_temperatures_array,
    plot_fill_between,
    plot_read_sweep,
    plot_fill_between_array,
    CMAP,
)

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams.update(
    {
        "figure.figsize": [3.5, 3.5],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "Inter",
        "lines.markersize": 2,
        "lines.linewidth": 1.2,
        "legend.fontsize": 5,
        "legend.frameon": False,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)
if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]
    dict_list = dict_list[2]



    read_temperatures = get_read_temperatures_array(dict_list)
    enable_read_currents = get_enable_read_currents_array(dict_list)


    fig, axs = plt.subplots(2,2, figsize=(8.3, 4), constrained_layout=True, width_ratios=[1, .25])
    
    ax = axs[0,0]
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "enable_read_current")
    plot_fill_between_array(ax, dict_list)
    ax.axvline(910, color="black", linestyle="--")
    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(400, 1000)


    ax = axs[0,1]
    ax.plot(
        enable_read_currents,
        read_temperatures,
        marker="o",
        color="black",
    )
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{read}}$ [K]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(.2))
    
 

    data_list = import_directory(r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_enable_write\data")
    data_list2 = [data_list[0], data_list[3], data_list[-6], data_list[-1]]
    colors = CMAP(np.linspace(0, 1, len(data_list2)))

    ax = axs[1,0]
    for j, data_dict in enumerate(data_list2):
        plot_read_sweep(
            ax, data_dict, "bit_error_rate", "enable_write_current", color=colors[j]
        )
        plot_fill_between(ax, data_dict, fill_color=colors[j])
        enable_write_current = get_enable_write_current(data_dict)

    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlim(400, 1000)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))

    ax = axs[1,1]
    # for i, data_dict in enumerate(data_list):
    #     state_currents = get_state_currents_measured(data_dict, "enable_write_current")
    #     for j, state in enumerate(state_currents[1]):
    #         if state > 0:
    #             ax.plot(
    #                 state_currents[0],
    #                 state,
    #                 marker="o",
    #                 color=colors[j],
    #             )
    write_current = get_write_current(data_dict)
    enable_write_currents = get_enable_write_current_array(data_list)
    # ax.set_xlabel("Write Temperature [K]")
    # ax.set_ylabel("Read Current [$\mu$A]")

    write_temperatures = get_write_temperatures_array(data_list)
    ax.plot(
        enable_write_currents,
        write_temperatures,
        marker="o",
        color="black",
    )

    for i, idx in enumerate([0, 3, -6, -1]):
        ax.plot(
            enable_write_currents[idx],
            write_temperatures[idx],
            marker="o",
            markersize=6,
            markeredgecolor="none",
            markerfacecolor=colors[i],
        )
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(.2))
    
    
    plt.savefig("read_current_sweep_enable_read3.pdf", bbox_inches="tight")