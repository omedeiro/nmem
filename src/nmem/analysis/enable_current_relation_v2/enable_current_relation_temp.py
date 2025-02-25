import matplotlib.pyplot as plt
import numpy as np
from nmem.analysis.analysis import (
    import_directory,
    plot_column,
    plot_full_grid,
    plot_grid,
    plot_row,
    get_current_cell,
    convert_cell_to_coordinates,
    get_fitting_points,
    filter_plateau,
    plot_linear_fit,
    plot_optimal_enable_currents,
    CMAP,
    get_max_enable_current, 
    calculate_channel_temperature,
    CRITICAL_TEMP,
    SUBSTRATE_TEMP,
)


if __name__ == "__main__":
    dict_list = import_directory("data")

    colors = CMAP(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]

    fig, axs = plt.subplots(
        1, 1, figsize=(120 / 25.4, 90 / 25.4), sharex=True, sharey=True
    )
    axs2 = axs.twinx()

    for data_dict in dict_list:
        cell = get_current_cell(data_dict)

        column, row = convert_cell_to_coordinates(cell)
        enable_currents = data_dict["x"][0]
        switching_current = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(enable_currents, switching_current, ztotal)
        axs.plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)

        # plot_linear_fit(
        #     axs,
        #     xfit,
        #     yfit,
        # )
        # plot_optimal_enable_currents(axs, data_dict)

        max_enable_current = get_max_enable_current(data_dict)
        channel_temperature = calculate_channel_temperature(
            CRITICAL_TEMP, SUBSTRATE_TEMP, enable_currents, max_enable_current
        )
        axs2.plot(enable_currents, channel_temperature, color="grey", marker="o")
        axs2.set_ybound(lower=0)
        axs.legend(loc="upper right")
        axs.set_xlim(0, 600)
        axs.set_ylim(0, 1000)
    axs.set_xlabel("Enable Current ($\mu$A)")
    axs.set_ylabel("Critical Current ($\mu$A)")
    axs2.set_ylabel("Channel Temperature (K)")
    plt.show()
