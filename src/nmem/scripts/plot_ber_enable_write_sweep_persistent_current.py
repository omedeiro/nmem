import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.constants import CRITICAL_TEMP
from nmem.analysis.data_import import import_directory
from nmem.analysis.state_currents_plots import (
    plot_measured_state_currents,
    plot_persistent_current,
)


def main(
    data_dir="../data/ber_sweep_enable_write_current/data1",
    mat_files=None,
    persistent_current=75,
    critical_current_zero=1250,
    save_dir=None,
):
    """
    Main function to plot persistent current and measured state currents.
    """
    dict_list = import_directory(data_dir)
    data_dict = dict_list[0]
    fig, ax = plot_persistent_current(
        data_dict, persistent_current, critical_current_zero
    )
    if mat_files is None:
        mat_files = [
            sio.loadmat(
                "../data/ber_sweep_enable_write_current/persistent_current/measured_state_currents_290.mat"
            ),
            sio.loadmat(
                "../data/ber_sweep_enable_write_current/persistent_current/measured_state_currents_300.mat"
            ),
            sio.loadmat(
                "../data/ber_sweep_enable_write_current/persistent_current/measured_state_currents_310.mat"
            ),
        ]
    colors = {0: "blue", 1: "blue", 2: "red", 3: "red"}
    plot_measured_state_currents(ax, [mat_files[1]], colors)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [au]")
    ax.grid()
    ax.set_xlim(0, CRITICAL_TEMP)
    ax.plot([7], [800], marker="x", color="black", markersize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlim(6, 9)
    ax.set_ylim(500, 900)
    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_persistent_current.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
