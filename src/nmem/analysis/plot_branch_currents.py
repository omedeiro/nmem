import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    CRITICAL_TEMP,
    RETRAP,
    SUBSTRATE_TEMP,
    WIDTH,
    get_critical_current_intercept,
    import_directory,
    plot_branch_currents,
)

N = 100
YMAX = 2000
experiment_marker = (7, 800)


def create_branch_current_plot(
    ax: plt.Axes, temp_array: np.ndarray, critical_current_zero: float
) -> plt.Axes:
    ax = plot_branch_currents(
        ax, temp_array, CRITICAL_TEMP, RETRAP, WIDTH, critical_current_zero
    )

    ax.axvline(SUBSTRATE_TEMP, color="black", linestyle=":", label="Substrate Temp")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [$\mu$A]")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    ax.set_xlim(0, CRITICAL_TEMP)
    ax.set_ylim(0, YMAX)
    ax.plot([0], [critical_current_zero], marker="x", color="black", markersize=10)
    ax.plot(
        experiment_marker[0],
        experiment_marker[1],
        marker="x",
        color="black",
        markersize=10,
    )

    return ax


if __name__ == "__main__":
    # Import
    data_dict = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_enable_read\data"
    )[0]

    # Preprocess
    critical_current_zero = get_critical_current_intercept(data_dict)
    temps = np.linspace(0, CRITICAL_TEMP, N)

    # Plot
    fig, ax = plt.subplots()
    ax = create_branch_current_plot(ax, temps, critical_current_zero)
    fig.savefig("plot_branch_currents.pdf", bbox_inches="tight")
    plt.show()
