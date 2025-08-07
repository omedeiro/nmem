"""
Script to plot the critical current vs enable current.
Enable current was swept through the linear region of the hTron reponse.
"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.htron_plotting import plot_enable_current_relation
from nmem.analysis.styles import apply_global_style

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    dict_list = import_directory("../data/enable_current_relation/data1")
    fig, ax = plt.subplots()
    plot_enable_current_relation(ax, dict_list)

    ax.set_xlabel("Enable Current ($\mu$A)")
    ax.set_ylabel("Critical Current ($\mu$A)")
    ax.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")

    if save_dir:
        plt.savefig(
            f"{save_dir}/ic_temperature_sweep_array.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
