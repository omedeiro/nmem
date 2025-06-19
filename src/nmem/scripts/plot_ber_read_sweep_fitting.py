import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from nmem.analysis.constants import CRITICAL_TEMP
from nmem.analysis.core_analysis import fit_state_currents, prepare_state_current_data
from nmem.analysis.currents import calculate_state_currents
from nmem.analysis.state_currents_plots import (
    plot_calculated_filled_region,
    plot_state_current_fit,
)


def main():

    ALPHA = 0.612
    WIDTH = 0.3
    persistent_current = 30
    critical_current_zero = 1240
    data_dict1 = sio.loadmat("../data/ber_sweep_enable_write_current/persistent_current/measured_state_currents_290.mat"),
    data_dict2 = sio.loadmat("../data/ber_sweep_enable_write_current/persistent_current/measured_state_currents_300.mat")
    data_dict3 = sio.loadmat("../data/ber_sweep_enable_write_current/persistent_current/measured_state_currents_310.mat")
    dict_list = [data_dict1, data_dict2, data_dict3]
    colors = {0: "blue", 1: "blue", 2: "red", 3: "red"}
    fit_results = []
    for data_dict in [dict_list[2]]:
        fig, ax = plt.subplots()
        x_list, y_list = prepare_state_current_data(data_dict)
        p0 = [ALPHA, persistent_current]
        fit = fit_state_currents(x_list, y_list, p0, WIDTH, critical_current_zero)
        fit_results.append(fit.x)
        x_list_full = np.linspace(0, CRITICAL_TEMP, 100)

        # Model for plotting
        def model_function(x0, x1, x2, x3, alpha, persistent):
            retrap = 1
            i0, _, _, _ = calculate_state_currents(
                x0,
                CRITICAL_TEMP,
                retrap,
                WIDTH,
                alpha,
                persistent,
                critical_current_zero,
            )
            _, i1, _, _ = calculate_state_currents(
                x1,
                CRITICAL_TEMP,
                retrap,
                WIDTH,
                alpha,
                persistent,
                critical_current_zero,
            )
            _, _, i2, _ = calculate_state_currents(
                x2,
                CRITICAL_TEMP,
                retrap,
                WIDTH,
                alpha,
                persistent,
                critical_current_zero,
            )
            _, _, _, i3 = calculate_state_currents(
                x3,
                CRITICAL_TEMP,
                retrap,
                WIDTH,
                alpha,
                persistent,
                critical_current_zero,
            )
            return [i0, i1, i2, i3]

        model = model_function(
            x_list_full, x_list_full, x_list_full, x_list_full, *fit.x
        )
        plot_state_current_fit(ax, x_list, y_list, x_list_full, model, colors)
        f = fit.x
        plot_calculated_filled_region(
            ax,
            x_list_full,
            data_dict,
            f[1],
            CRITICAL_TEMP,
            1,
            WIDTH,
            f[0],
            critical_current_zero,
        )
        ax_inset = fig.add_axes([0.15, 0.15, 0.35, 0.35])
        for i in range(4):
            ax_inset.plot(x_list[i], y_list[i], "o", color=colors[i])
            plot_calculated_filled_region(
                ax_inset,
                x_list_full,
                data_dict,
                f[1],
                CRITICAL_TEMP,
                1,
                WIDTH,
                f[0],
                critical_current_zero,
            )
            ax_inset.plot(x_list_full, model[i], "--", color=colors[i])
        ax_inset.set_xlim([6, 8.5])
        ax_inset.set_ylim([500, 950])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
    for f in fit_results:
        print(f"Alpha: {f[0]:.2f}, Persistent: {f[1]:.2f}")


if __name__ == "__main__":
    main()
