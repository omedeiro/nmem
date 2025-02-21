import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import least_squares
from nmem.analysis.analysis import (
    calculate_state_currents,
    get_critical_current_intercept,
    import_directory,
    CRITICAL_TEMP,
    CMAP,
    plot_calculated_filled_region,
)


def filter_nan(x, y):
    mask = np.isnan(y)
    x = x[~mask]
    y = y[~mask]
    return x, y


def residuals(p, x0, y0, x1, y1, x2, y2, x3, y3) -> float:
    alpha, persistent = p
    model = model_function(
        x0, x1, x2, x3, alpha, persistent
    )
    residuals = np.concatenate(
        [
            y0 - model[0],
            y1 - model[1],
            y2 - model[2],
            y3 - model[3],
        ]
    )
    return residuals


def model_function(x0, x1, x2, x3, alpha, persistent):
    width = WIDTH
    critical_current_zero = 1240
    retrap = 1
    i0, _, _, _ = calculate_state_currents(
        x0, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
    )
    _, i1, _, _ = calculate_state_currents(
        x1, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
    )
    _, _, i2, _ = calculate_state_currents(
        x2, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
    )
    _, _, _, i3 = calculate_state_currents(
        x3, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
    )
    model = [i0, i1, i2, i3]
    return model


if __name__ == "__main__":

    data = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_enable_read\data"
    )

    ALPHA = 0.612
    WIDTH = 0.3
    persistent_current = 30
    # critical_current_zero = get_critical_current_intercept(data[0]) * 0.88
    critical_current_zero = 1240
    data_dict1 = sio.loadmat("measured_state_currents_290.mat")
    data_dict2 = sio.loadmat("measured_state_currents_300.mat")
    data_dict3 = sio.loadmat("measured_state_currents_310.mat")

    dict_list = [data_dict1, data_dict2, data_dict3]
    # colors = CMAP(np.linspace(0.1, 1, 4))
    colors = {0: "blue", 1: "blue", 2: "red", 3: "red"}
    fit_results = []
    for data_dict in [dict_list[2]]:
        fig, ax = plt.subplots()
        temp = data_dict["measured_temperature"].flatten()
        state_currents = data_dict["measured_state_currents"]
        x_list = []
        y_list = []
        for i in range(4):
            x = temp
            y = state_currents[:, i]
            x, y = filter_nan(x, y)
            ax.plot(x, y, "-o", color=colors[i], label=f"State {i}")
            if len(x) > 0:
                x_list.append(x)
                y_list.append(y)
            else:
                x_list.append(None)
                y_list.append(None)
        print(critical_current_zero)
        p0 = [ALPHA, persistent_current]
        fit = least_squares(
            residuals,
            p0,
            args=(
                x_list[0],
                y_list[0],
                x_list[1],
                y_list[1],
                x_list[2],
                y_list[2],
                x_list[3],
                y_list[3],
            ),
            bounds=([0, -100], [1, 100]),
        )
        fit_results.append(fit.x)
        x_list_full = np.linspace(0, CRITICAL_TEMP, 100)
        model = model_function(
            x_list_full, x_list_full, x_list_full, x_list_full, *fit.x
        )
        for i in range(4):
            ax.plot(x_list_full, model[i], "--", color=colors[i])

        ax.legend()

        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Current [$\mu$A]")
        ax.grid()
        # ax.set_xlim([6, 8.5])
        # ax.set_ylim([500, 1000])


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
            # ax_inset.plot(x_list_full, model[i], "--", color=colors[i])
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
                critical_current_zero
            )
            ax_inset.plot(x_list_full, model[i], "--", color=colors[i])

        ax_inset.set_xlim([6, 8.5])
        ax_inset.set_ylim([500, 950])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

    ax.set_ybound(lower=0)   
    ax.set_xbound(lower=0) 
    for f in fit_results:
        print(
            f"Alpha: {f[0]:.2f}, Persistent: {f[1]:.2f}"
        )

