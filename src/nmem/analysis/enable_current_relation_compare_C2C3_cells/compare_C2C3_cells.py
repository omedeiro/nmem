import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    build_array,
    get_fitting_points,
    import_directory,
    plot_fitting,
)


def plot_fit(ax, xfit, yfit, **kwargs):
    plot_fitting(ax, xfit, yfit, **kwargs)
    ax.legend()
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 1000])
    ax.set_xlabel("Enable Current ($\mu$A)")
    ax.set_ylabel("Critical Current ($\mu$A)")

    return ax


def get_fit_points(data_dict: dict):
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)

    return xfit, yfit


IDX_C2 = 5
IDX_C3 = 10

if __name__ == "__main__":
    # Import
    data_list = import_directory("data")
    data_dict_C2 = data_list[0]
    data_dict_C3 = data_list[1]

    # Preprocess
    xfit_c2, yfit_c2 = get_fit_points(data_dict_C2)
    xfit_c3, yfit_c3 = get_fit_points(data_dict_C3)

    # Plot
    fig, ax = plt.subplots()
    ax = plot_fitting(ax, xfit_c2[IDX_C2 + 1 :], yfit_c2[IDX_C2 + 1 :], linestyle="-")
    ax.plot(xfit_c2, yfit_c2, label="C2", linestyle="-")
    ax.plot(xfit_c3, yfit_c3, label="C3", linestyle="-")
    ax.legend()

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    axs[0].plot(xfit_c3, yfit_c3, label="C3", linestyle="-")
    axs[0] = plot_fit(axs[0], xfit_c3[:IDX_C3], yfit_c3[:IDX_C3], linestyle="-")
    
    axs[1].plot(xfit_c3, yfit_c3, label="C3", linestyle="-")
    axs[1] = plot_fit(
        axs[1], xfit_c3[IDX_C3 + 1 :], yfit_c3[IDX_C3 + 1 :], linestyle="-"
    )

    save = True
    if save:
        fig.savefig(
            "enable_current_relation_compare_C2C3.png", dpi=300, bbox_inches="tight"
        )
    plt.show()
